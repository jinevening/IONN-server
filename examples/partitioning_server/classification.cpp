#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <boost/thread.hpp>

#define FEATURE_BUFF_SIZE 64*1024*1024
#define BUFF_SIZE 256*1024*1024
#define PORT 7675
#define PARAM_BUFF_SIZE 8

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

using boost::asio::ip::tcp;

// Saved network parameter
// We will keep received network parameter here for incremental offloading
NetParameter saved_net_param;
Net<float>* saved_net=NULL;
int total_bytes_model = 0;
int transmitted_bytes_inuse = 0;
bool firstmodelmade = false;
bool model_data_ready = false;

boost::condition_variable cond_first_model;
boost::condition_variable cond_model_data;
boost::mutex mutex_first_model;
boost::mutex mutex_model_data;

boost::mutex mutex_net_inuse;
boost::mutex mutex_saved_net;

boost::mutex mutex_net_parameter;
std::vector<NetParameter* > proto_param;
std::vector<NetParameter* > front_net_param;
std::vector<NetParameter* > rear_net_param;
std::vector<int> proto_size;
std::vector<int> offloading_point;
std::vector<int> resume_point;
std::vector<int> prototxt_end;
std::vector<int> front_model_size;
std::vector<int> rear_model_size;
int net_param_index = 0;

bool push_back_net_param(unsigned char* buffer_ptr){ //called by model_upload_server
  double timechk;
  struct timeval start;
  struct timeval finish;
  gettimeofday(&start, NULL);
  NetParameter* proto_param_pointer = new NetParameter();
  NetParameter* front_net_param_pointer = new NetParameter();
  NetParameter* rear_net_param_pointer = new NetParameter();
  gettimeofday(&finish, NULL);
  timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
          (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
  cout << "Initializing 3 NetParameter in heap took " << timechk << " s" << endl;


  boost::lock_guard<boost::mutex> guard(mutex_net_parameter);

  proto_size.push_back(0);
  offloading_point.push_back(0);
  resume_point.push_back(0);
  prototxt_end.push_back(0);
  front_model_size.push_back(0);
  rear_model_size.push_back(0);

  CHECK_EQ(sizeof(int), 4);
  memcpy(&proto_size.back(), buffer_ptr + 4, 4);
  memcpy(&front_model_size.back(), buffer_ptr + 8, 4);
  memcpy(&rear_model_size.back(), buffer_ptr + 12, 4);
  memcpy(&offloading_point.back(), buffer_ptr + 16, 4);
  memcpy(&resume_point.back(), buffer_ptr + 20, 4);
  memcpy(&prototxt_end.back(), buffer_ptr + 24, 4);


  cout << "proto_size "<< proto_size.back() << " bytes " << "from 1 to " << prototxt_end.back() << endl;
  gettimeofday(&start, NULL);
  int sizeofindexes = 28;
// Decode received data
  if (!(proto_param_pointer->ParseFromArray(buffer_ptr + sizeofindexes, proto_size.back()))) {
    perror("protobuf prototxt decoding failed");
    exit(EXIT_FAILURE);
  }
  if (front_model_size.back() > 0) {
    if (!(front_net_param_pointer->ParseFromArray(buffer_ptr + sizeofindexes + proto_size.back(), front_model_size.back() ))) {
      perror("Protobuf front network decoding failed");
      exit(EXIT_FAILURE);
    }
  }
  if (rear_model_size.back() > 0) {
    if (!(rear_net_param_pointer->ParseFromArray(buffer_ptr + sizeofindexes + proto_size.back() + front_model_size.back(), rear_model_size.back() ))) {
      perror("Protobuf rear network decoding failed");
      exit(EXIT_FAILURE);
    }
  }

  CHECK(!(front_model_size.back() == 0 && rear_model_size.back() == 0));

  proto_param.push_back(proto_param_pointer);
  front_net_param.push_back(front_net_param_pointer);
  rear_net_param.push_back(rear_net_param_pointer);

  if(model_data_ready == false){
    {
      boost::lock_guard<boost::mutex> lock(mutex_model_data);
      model_data_ready = true;
    }
    cond_model_data.notify_all();
  }


  gettimeofday(&finish, NULL);
  timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
            (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
  cout << "Server-side decode time of model data: " << timechk << " s" << endl;
  return true;
}


void execution_server(unsigned short port){
  Caffe::set_mode(Caffe::GPU);
  boost::asio::io_service io_service;
  tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
  cout << "Execution Server Started on Port " << port << endl;

  // Huge buffer. We will use this again and again.
  unsigned char* buffer = new unsigned char[FEATURE_BUFF_SIZE];

  for (;;)
  {
    tcp::socket sock(io_service);
    a.accept(sock);

    Net<float>* net;


    for (;;) {
      memset(buffer, 0, FEATURE_BUFF_SIZE);
      unsigned char* buffer_ptr = buffer;

      int feature_size = 0;	// feature size
      double timechk;
      struct timeval start;
      struct timeval finish;
      gettimeofday(&start, NULL);

      boost::system::error_code error;

      
      try {
        // Receive Data
        do {
          size_t length = sock.read_some(boost::asio::buffer(buffer_ptr, FEATURE_BUFF_SIZE), error);
          if (error == boost::asio::error::eof)
            break; // Connection closed cleanly by peer.
          else if (error)
            throw boost::system::system_error(error); // Some other error.

          if (buffer == buffer_ptr) {
            memcpy(&feature_size, buffer, 4);
            cout << "Total size of feature data " << feature_size << " bytes" << endl;
          }
          buffer_ptr += length;
        	//cout << "Execution thread" << endl;
        //	cout << "Received data so far : " << buffer_ptr - buffer << endl;
        } while ((buffer_ptr - buffer) < feature_size + 4 );
      }
      catch (std::exception& e) {
        std::cerr << "Exception in thread: " << e.what() << "\n";
        return;
      }

      if (error == boost::asio::error::eof)
        break;



        

//      boost::asio::read(sock, boost::asio::buffer(buffer_ptr, 4), boost::asio::transfer_all(), error);
//      memcpy(&feature_size, buffer_ptr, 4);

//      availBytes = sock.available();
//      while(availBytes < total_size){
//        availBytes = sock.available();
//      }

//      size_t n = boost::asio::read(sock, boost::asio::buffer(buffer_ptr+4, feature_size), boost::asio::transfer_all(), error);

//      if (error == boost::asio::error::eof)
//        break;


      gettimeofday(&finish, NULL);
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
                (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
      cout << "Received " << buffer_ptr-buffer << " bytes for forward. Time to receive: " << timechk << "s" << endl;

      CHECK_EQ(sizeof(int), 4);

      gettimeofday(&start, NULL);
      // Decode received data
      BlobProto feature;
      if (!(feature.ParseFromArray(buffer + 4, feature_size))) {
        perror("Protobuf feature decoding failed");
        exit(EXIT_FAILURE);
      }
      gettimeofday(&finish, NULL);
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
                (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
      cout << "Server-side decode time of feature data: " << timechk << " s" << endl;

      // If more bytes have been uploaded,
      // then we update the model to the latest one

      {
        boost::lock_guard<boost::mutex> _(mutex_saved_net);
        if (transmitted_bytes_inuse != total_bytes_model) {
          transmitted_bytes_inuse = total_bytes_model;
//          net.reset(saved_net);
          {
            boost::lock_guard<boost::mutex> _(mutex_net_inuse);
            saved_net->inUse = true;
          }
          net = saved_net;
        }
        else if(firstmodelmade == true) {
          boost::lock_guard<boost::mutex> _(mutex_net_inuse);
          net->inUse = true;
        }
      }
      
      if(firstmodelmade == false){
        cout << "net not created, going into wait"<<endl;
        boost::unique_lock<boost::mutex> lock(mutex_first_model);
        while(firstmodelmade == false){
          cond_first_model.wait(lock);
        }
         cout <<"woke from wait"<<endl;
         boost::lock_guard<boost::mutex> _(mutex_saved_net);
         transmitted_bytes_inuse = total_bytes_model;
         saved_net->inUse = true;
         net = saved_net;
      }

      cout << "net used for forward has layers from "<<net->getLayerIDLeft() << ", " << net->getLayerIDRight()<< endl;
      Blob<float>* input_layer;
      if(net->getLayerIDLeft()==1){
        input_layer = net->input_blobs()[0];
        input_layer->FromProto(feature, false);
      }
      else{
//        net->bottom_vecs()[net->getLayerIDLeft()][0]->FromProto(feature, false);
        cout <<"test, input_blobs()[7].size : " <<net->input_blobs().size()<<endl;
        net->input_blobs()[net->getLayerIDLeft()]->FromProto(feature, false);
      }
      gettimeofday(&start, NULL);
      // Run forward
//      cout <<"test: server layer numb= " << net->layer_names().size() - 1<< endl;
      if(net->getLayerIDLeft()==1){
        net->Forward();
      }
      else{
        cout<<"test, bottomvecs size " <<net->bottom_vecs()[7].size()<<endl;
        net->ForwardFromTo(net->getLayerIDLeft(), net->getLayerIDRight());
      }
      int computed_upto = net->getLayerIDRight(); 
      gettimeofday(&finish, NULL);
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
                (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
      cout << "Server-side forward time : " << timechk << "s" << endl;

      // Get output data
      Blob<float>* output_layer = net->output_blobs()[0];
      BlobProto output_proto;
      output_layer->ToProto(&output_proto);
      {
        boost::lock_guard<boost::mutex> _(mutex_net_inuse);
        net->inUse = false;
      }
      int output_size = output_proto.ByteSize();
      memcpy(buffer, &output_size, 4);
      memcpy(buffer+4, &computed_upto, 4);
      output_proto.SerializeWithCachedSizesToArray(buffer + 8);
      // send(new_socket , buffer , output_size , 0 );
      cout << "Output blob size : " << output_size << " bytes" << endl;

      // Send output data to the client
      int sent_bytes = boost::asio::write(sock, boost::asio::buffer(buffer, output_size + 8));
      cout << "Sent " << sent_bytes << " bytes of forward result upto layer num " << computed_upto<< endl;
    }

    // Connection closed by client
    cout << "Client closed the connection" << endl;
  }
}

void model_upload_server(int port){


  boost::asio::io_service io_service;
  tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
  cout << "Upload Server Started on Port " << port << endl;
  // Huge buffer. We will use this again and again.
  unsigned char* buffer = new unsigned char[BUFF_SIZE];

//  Net<float>* net_test = NULL;

  for (;;)
  {
    tcp::socket sock(io_service);
    a.accept(sock);
    std::cout << "a client connected" << std::endl;

    bool is_first_net =true;
    size_t n;
    int next_total_size = 0;
    size_t data_to_read_sum = 0;
    size_t data_already_read_sum = 0;
    int increase_in_model_size = 0;

    for (;;) {
      memset(buffer, 0, BUFF_SIZE);
      unsigned char* buffer_ptr = buffer;

      int curr_total_size = 0;	// proto + front_model + rear_model
      double timechk;
      struct timeval start;
      struct timeval finish;

      cout << "partitions waiting to be created: " << proto_param.size() <<endl;

      boost::system::error_code error;
      size_t availBytes = 0;

      do{
        if (is_first_net) {
          is_first_net = false;
          do{
            n = boost::asio::read(sock, boost::asio::buffer(buffer_ptr, 4), boost::asio::transfer_all(), error);
          }
          while(n < 4);
          memcpy(&curr_total_size, buffer_ptr, 4);
        }
        else {
          curr_total_size = next_total_size;
        }
        increase_in_model_size += curr_total_size;
        int temp = sock.available();
        data_already_read_sum += temp;
        data_to_read_sum += curr_total_size-temp;
        cout << "Total size " << curr_total_size << " bytes of model partition to read, "<< temp << " bytes available in socket" << endl;
        cout << "Already read data size sum: "<<data_already_read_sum<<". Data to read sum: "<< data_to_read_sum << endl;

        do {
          n  = boost::asio::read(sock, boost::asio::buffer(buffer_ptr+4, curr_total_size+24), boost::asio::transfer_all(), error);
        }
        while(n < curr_total_size + 24);

        if (error == boost::asio::error::eof)
          break; //for some reason a cout<<...; code in this block causes error in client
        gettimeofday(&finish, NULL);
        timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
            (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;


        cout << "Received " << n -24 << " bytes of model. Time to recieve : " << timechk << "s" << endl;

        push_back_net_param(buffer_ptr); //decode received data and store to global std::vector variables
        
        do{
          n = boost::asio::read(sock, boost::asio::buffer(buffer_ptr, 4), boost::asio::transfer_all(), error);
        }
        while(n < 4);
        memcpy(&next_total_size, buffer_ptr, 4);
        availBytes = 0;
        availBytes = sock.available();
        cout << "next partition size: " << next_total_size << " and bytes available in socket: " << availBytes << endl;
        if (next_total_size == 0) {
          is_first_net = true;
          cout << "model upload complete" << endl;
        }
      }while(availBytes >= next_total_size and availBytes != 0);

    }
  }
}

void model_create_server(){
  double timechk_model_make = 0;
  int num_model_made = 0;
  std::list<Net<float>* > nets_inuse;
  int increase_in_model_size = 0;

  for (;;)
  {
    if(model_data_ready == false){
      boost::unique_lock<boost::mutex> lock(mutex_model_data);
      while(model_data_ready == false){
  //      boost::unique_lock<boost::mutex> lock=boost::unique_lock<boost::mutex>(mutex_model_data);
        cond_model_data.wait(lock);
      }
    }

    mutex_net_parameter.lock();
    try{
      net_param_index = proto_param.size() - 1;
    }
    catch (...){
      mutex_net_parameter.unlock();
      throw;
    }
    mutex_net_parameter.unlock();
    struct timeval start;
    struct timeval finish;
    double timechk;
//    struct timeval start2;
//    struct timeval finish2;
//    double timechk2;
    gettimeofday(&start, NULL);
    Net<float>* new_net = new Net<float>( *(proto_param[net_param_index]) );
//      cout << "breakpoint model2" <<endl;
//    gettimeofday(&finish, NULL);
//      new_net->Reshape();
//      net.reset(new_net);
//    timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
//    (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
//    timechk_model_make += timechk;
    num_model_made++;
    increase_in_model_size += front_model_size[net_param_index] + rear_model_size[net_param_index];

//    cout << "Remaking of protoparam time of "<< num_model_made <<"th model : " << timechk << " s, total "<< timechk_model_make << " s" << endl;
//    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
    new_net ->inUse = false;

    new_net->setLayerIDLeft(offloading_point[net_param_index]);
    new_net->setLayerIDRight(resume_point[net_param_index]);
//    gettimeofday(&start2, NULL);
    new_net->CopyTrainedLayersFrom(saved_net_param);
//    gettimeofday(&finish2, NULL);
    for(size_t i = 0; i < net_param_index + 1; i++){
      if (front_model_size[i] > 0) {
        new_net->CopyTrainedLayersFrom((*front_net_param[i]));
        if (rear_model_size[i] > 0)  // both front model and rear model arrived
          new_net->CopyTrainedLayersFrom((*rear_net_param[i]));
        }
      else {  // only rear model arrived
        new_net->CopyTrainedLayersFrom((*rear_net_param[i]));

      }
    }
    mutex_net_parameter.lock();
    try{
      for(int i = 0; i < net_param_index + 1; i++){
        proto_param[i]->~NetParameter();
        front_net_param[i]->~NetParameter();
        rear_net_param[i]->~NetParameter();
      }
      proto_param.erase(proto_param.begin(), proto_param.begin() + net_param_index + 1);
      front_net_param.erase(front_net_param.begin(), front_net_param.begin() + net_param_index + 1);
      rear_net_param.erase(rear_net_param.begin(), rear_net_param.begin() + net_param_index + 1);
      proto_size.erase(proto_size.begin(), proto_size.begin()+ net_param_index + 1); 
      offloading_point.erase(offloading_point.begin(), offloading_point.begin() + net_param_index + 1);
      resume_point.erase(resume_point.begin(), resume_point.begin()+ net_param_index + 1);
      prototxt_end.erase(prototxt_end.begin(), prototxt_end.begin() + net_param_index + 1);
      front_model_size.erase(front_model_size.begin(), front_model_size.begin() + net_param_index + 1);
      rear_model_size.erase(rear_model_size.begin(), rear_model_size.begin() + net_param_index + 1);

      if (proto_param.size() == 0) {
        model_data_ready = false;
      }
    }
    catch(...) {
      mutex_net_parameter.unlock();
      throw;
    }
    mutex_net_parameter.unlock();

    gettimeofday(&finish, NULL);

    cout << "Layers in "<<num_model_made <<"th model = "<<new_net->getLayerIDLeft() << ", " << new_net->getLayerIDRight() <<" num of layers = "<<new_net->layer_names().size() << " number of partitions added = " << net_param_index + 1<< endl;
    timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
              (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
    timechk_model_make += timechk;
    cout << "Server-side Net creation time : " << timechk << " s. Total " << timechk_model_make << " s."<< endl;

    {
      boost::lock_guard<boost::mutex> _(mutex_saved_net);
      total_bytes_model += increase_in_model_size;
      saved_net = new_net;
    }
    nets_inuse.push_back(new_net);
    if (nets_inuse.size() > 1 ){
      std::list<Net<float>* >::iterator i = nets_inuse.begin();
//      auto i = nets_inuse.begin();
      int model_freed = 0;

//      gettimeofday(&start, NULL);

      while(i!= (--nets_inuse.end()) ){
        boost::lock_guard<boost::mutex> _(mutex_net_inuse);
        if ((*i)->inUse == false){
//          cout << "before deconstructor in list is needed ";
//          (*i)->~Net();
          boost::thread t([](Net<float>* net_pointer){net_pointer->~Net();}, (*i));
          t.detach();
          model_freed += 1;
          i = nets_inuse.erase(i);
//          cout << "after deconstructor is performed ";
        }
        else {
          i++;
        }
      }
//      gettimeofday(&finish, NULL);
//      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
//      (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
//      cout << "Freeing "<< model_freed << " models took " << timechk << "s" << endl;
    }

    if(firstmodelmade == false){
      {
        boost::lock_guard<boost::mutex> lock(mutex_first_model);
        firstmodelmade = true;
      }
      cond_first_model.notify_all();
    }
    // Save current net parameter for later use (incremental offloading)
    new_net->ToProto(&saved_net_param, false);
    
  }
}

int main(int argc, char** argv) {
  string net_file;
  if (argc == 2){
    net_file = argv[1];
  }

#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
  cout << "CPU mode" << endl;
#else
  Caffe::set_mode(Caffe::GPU);
  cout << "GPU mode" << endl;
#endif

  google::InitGoogleLogging(argv[0]);
  try {
//    boost::asio::io_service io_service;
//    server(io_service, PORT);
    if(argc == 2){
      struct timeval start;
      struct timeval finish;
      double timechk;
      cout <<"Performing warmup" <<endl;
      gettimeofday(&start, NULL);
      Net<float>* warmup_net = new Net<float>(net_file, TEST);
      warmup_net->~Net();
      gettimeofday(&finish, NULL);
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
                (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
      cout << "Warmup took " << timechk << " s" << endl;
    }
    boost::thread_group threads;
    threads.create_thread(boost::bind(model_upload_server, 7675));
    threads.create_thread(boost::bind(model_create_server));
    threads.create_thread(boost::bind(execution_server, 7676));
    threads.join_all();
  }
  catch (std::exception& e) {
    std::cerr << "Exception : " << e.what() << endl;
  }

  return 0;
}
