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
#define PROTOBUF_BUFF_SIZE 8

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

using boost::asio::ip::tcp;

// Saved network parameter
// We will keep received network parameter here for incremental offloading
NetParameter saved_net_param;
Net<float>* saved_net=NULL;
int transmitted_bytes = 0;
bool firstmodelmade = false;
boost::condition_variable_any cond;
boost::mutex mutex;

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

    int prev_transmitted_bytes = 0;

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
            cout << "Total size of feature data" << feature_size << " bytes" << endl;
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
      cout << "Received " << buffer_ptr-buffer << " bytes for forward" << endl;
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
                (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
                cout << "Time to receive: " << timechk << " s" << endl;

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
      if (prev_transmitted_bytes != transmitted_bytes) {
        prev_transmitted_bytes = transmitted_bytes;
//        net.reset(saved_net);
        saved_net->inUse = true;
        net = saved_net;
      }
      
      if(firstmodelmade == false){
         cout << "net not created, going into wait"<<endl;
        while(firstmodelmade == false){
          boost::unique_lock<boost::mutex> lock=boost::unique_lock<boost::mutex>(mutex);
          cond.wait(mutex);
        }
         cout <<"woke from wait"<<endl;
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
      cout << "Server-side forward time : " << timechk << " s" << endl;

      // Get output data
      Blob<float>* output_layer = net->output_blobs()[0];
      net->inUse = false;
      BlobProto output_proto;
      output_layer->ToProto(&output_proto);
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

void upload_server(int port)
{
  boost::asio::io_service io_service;
  tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
  cout << "Upload Server Started on Port " << port << endl;

  // Huge buffer. We will use this again and again.
  unsigned char* buffer = new unsigned char[BUFF_SIZE];

  Net<float>* net_test = NULL;

  

  for (;;)
  {
    tcp::socket sock(io_service);
    a.accept(sock);
    std::cout << "a client connected" << std::endl;

    Net<float>* net;
    NetParameter proto_param;
    NetParameter front_net_param;
    NetParameter rear_net_param;
    NetParameter total_proto_param;
    saved_net = NULL;
    std::list<Net<float>* > nets_inuse;


    for (;;) {
      memset(buffer, 0, BUFF_SIZE);
      unsigned char* buffer_ptr = buffer;

      int total_size = 0;	// proto + front_model + rear_model
      int proto_size = 0;
      int offloading_point = 0;
      int resume_point = 0;
      int prototxt_end = 0;
      int front_model_size = 0;
      int rear_model_size = 0;
      double timechk;
      double timechk2;
      struct timeval start;
      struct timeval finish;

      cout << "nets_inuse size before getting data from buffer: " << nets_inuse.size() <<endl;

      boost::system::error_code error;
/*      gettimeofday(&start, NULL);
      try {
        // Receive Data
        do {
          size_t length = sock.read_some(boost::asio::buffer(buffer_ptr, BUFF_SIZE), error);
          if (error == boost::asio::error::eof)
            break; // Connection closed cleanly by peer.
          else if (error)
            throw boost::system::system_error(error); // Some other error.

          if (buffer == buffer_ptr) {
            memcpy(&total_size, buffer, 4);
            cout << "Total size " << total_size << " bytes" << endl;
          }
          buffer_ptr += length;
        	//cout << "Uploading thread" << endl;
        //	cout << "Received data so far : " << buffer_ptr - buffer << endl;
        } while ((buffer_ptr - buffer) < total_size + 16 );
      }
      catch (std::exception& e) {
        std::cerr << "Exception in thread: " << e.what() << "\n";
        return;
      }
*/

      // testfornetcreat
/*
      boost::asio::read(sock, boost::asio::buffer(buffer_ptr, 4), boost::asio::transfer_all(), error);
      memcpy(&total_proto_size, buffer_ptr, 4);
      buffer_ptr = buffer_ptr + 4;
      boost::asio::read(sock, boost::asio::buffer(buffer_ptr, total_proto_size), boost::asio::transfer_all(), error);
      if (!(total_proto_param.ParseFromArray(buffer_ptr, total_proto_size))) {
        perror("protobuf total_prototxt decoding failed");
        exit(EXIT_FAILURE);
      }
      buffer_ptr += total_proto_size;
*/
      //testend...more later
//      size_t availBytes;
//      availBytes = sock.available();
//      while(availBytes < 4){
//        availBytes = sock.available();
//      }

      size_t n;
      do{
        n = boost::asio::read(sock, boost::asio::buffer(buffer_ptr, 4), boost::asio::transfer_all(), error);
      }
      while(n < 4);
      memcpy(&total_size, buffer_ptr, 4);
      cout << "Total size " << total_size << " bytes of model to process"<<endl;
      do {
        n  = boost::asio::read(sock, boost::asio::buffer(buffer_ptr+4, total_size+24), boost::asio::transfer_all(), error);
      }
      while(n < total_size + 24);

      if (error == boost::asio::error::eof)
        break; //for some reason a cout<<...; code in this block causes error in client
      gettimeofday(&finish, NULL);
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
          (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;


      cout << "Received " << n -24 << " bytes of model" << endl;
      cout << "Time to recieve : " << timechk << " s" << endl;

      CHECK_EQ(sizeof(int), 4);
      memcpy(&proto_size, buffer_ptr + 4, 4);
      memcpy(&front_model_size, buffer_ptr + 8, 4);
      memcpy(&rear_model_size, buffer_ptr + 12, 4);
      memcpy(&offloading_point, buffer_ptr + 16, 4);
      memcpy(&resume_point, buffer_ptr + 20, 4);
      memcpy(&prototxt_end, buffer_ptr + 24, 4);


      cout << "proto_size = " << proto_size << " bytes " << "from 1 to " << prototxt_end<< endl;
      gettimeofday(&start, NULL);
      int sizeofindexes = 28;
      // Decode received data
      if (!(proto_param.ParseFromArray(buffer_ptr + sizeofindexes, proto_size))) {
        perror("protobuf prototxt decoding failed");
        exit(EXIT_FAILURE);
      }
      if (front_model_size > 0) {
        if (!(front_net_param.ParseFromArray(buffer_ptr + sizeofindexes + proto_size, front_model_size))) {
          perror("Protobuf front network decoding failed");
          exit(EXIT_FAILURE);
        }
      }
      if (rear_model_size > 0) {
        if (!(rear_net_param.ParseFromArray(buffer_ptr + sizeofindexes + proto_size + front_model_size, rear_model_size))) {
          perror("Protobuf rear network decoding failed");
          exit(EXIT_FAILURE);
        }
      }
      gettimeofday(&finish, NULL);
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
                (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
      cout << "Server-side decode time of model data: " << timechk << " s" << endl;

      gettimeofday(&start, NULL);

      CHECK(!(front_model_size == 0 && rear_model_size == 0));

      // Initialize received network
      // If front/rear model is arrived, we create a new network
      // If not, we will just use the previous network
//      if (!(front_model_size == 0 && rear_model_size == 0)) {
        struct timeval start2;
        struct timeval finish2;

        // testcreatednet2
        /*
        if(net_test == NULL){
        gettimeofday(&start2, NULL);
          net_test = new Net<float>(total_proto_param);
        gettimeofday(&finish2, NULL);
        timechk2 = (double)(finish2.tv_sec) + (double)(finish2.tv_usec) / 1000000.0 -
        (double)(start2.tv_sec) - (double)(start2.tv_usec) / 1000000.0;
        cout << "Remaking of first total protoparam time cost : " << timechk2 << " s" << endl;
        }
        */
        //testend

//        cout << "breakpoint model1"<<endl;


        gettimeofday(&start2, NULL);
//        cin.get();


        Net<float>* new_net = new Net<float>(proto_param);
//        cout << "breakpoint model2" <<endl;
        gettimeofday(&finish2, NULL);
//        new_net->Reshape();
//        net.reset(new_net);
        timechk2 = (double)(finish2.tv_sec) + (double)(finish2.tv_usec) / 1000000.0 -
        (double)(start2.tv_sec) - (double)(start2.tv_usec) / 1000000.0;
        cout << "Remaking of protoparam time cost : " << timechk2 << " s" << endl;
        net = new_net;
        new_net ->inUse = false;




        net->setLayerIDLeft(offloading_point);
        net->setLayerIDRight(resume_point);
        if (front_model_size > 0) {
          gettimeofday(&start2, NULL);
          net->CopyTrainedLayersFrom(saved_net_param);
          gettimeofday(&finish2, NULL);
          net->CopyTrainedLayersFrom(front_net_param);

//          net_test->CopyTrainedLayersFrom(front_net_param);

          if (rear_model_size > 0)  // both front model and rear model arrived
            net->CopyTrainedLayersFrom(rear_net_param);
//            net_test->CopyTrainedLayersFrom(rear_net_param);
        }
        else {  // only rear model arrived
          gettimeofday(&start2, NULL);
          net->CopyTrainedLayersFrom(saved_net_param);
          gettimeofday(&finish2, NULL);
          net->CopyTrainedLayersFrom(rear_net_param);

//          net_test->CopyTrainedLayersFrom(rear_net_param);
        }
//      }
      gettimeofday(&finish, NULL);

      cout << "Layers in newmodel = "<<net->getLayerIDLeft() << ", " << net->getLayerIDRight() <<" num of layers "<<net->layer_names().size() <<endl;
      timechk = (double)(finish.tv_sec) + (double)(finish.tv_usec) / 1000000.0 -
                (double)(start.tv_sec) - (double)(start.tv_usec) / 1000000.0;
      cout << "Server-side Net creation time : " << timechk << " s" << endl;


      timechk2 = (double)(finish2.tv_sec) + (double)(finish2.tv_usec) / 1000000.0 -
      (double)(start2.tv_sec) - (double)(start2.tv_usec) / 1000000.0;

      cout << "Copying parameters of previous layer time : " << timechk2 << " s" << endl;

      // Send ACK to the client
//      buffer[0] = 'A';
//      buffer[1] = 'C';
//      buffer[2] = 'K';
//      int sent_bytes = boost::asio::write(sock, boost::asio::buffer(buffer, 3));
//      cout << "Sent " << sent_bytes << " bytes" << endl;
      Net<float>* temp_net = saved_net;
      saved_net = new_net;
//      if( temp_net != NULL and temp_net->inUse == false) {
//        cout << "before deconstructor of prev net";
//        temp_net->~Net();
//        cout << "after deconstructor of prev net";
//      }
//      else if(temp_net != NULL){
//        nets_inuse.push_back(temp_net);
//        cout << "nets_inuse size = " << nets_inuse.size() << endl;
//      }
//
      if (temp_net != NULL) {
        nets_inuse.push_back(temp_net);
      }
      if (nets_inuse.size() > 5 ){
        std::list<Net<float>* >::iterator i = nets_inuse.begin();
        while(i!= --(--nets_inuse.end()) ){
          if ((*i)->inUse == false){
            cout << "before deconstructor in list is needed";
            (*i)->~Net();
            i = nets_inuse.erase(i);
            cout << "after deconstructor is performed";
          }
          else {
            i++;
          }
        }
      }
//      for(std::list<Net<float>* >::iterator i = nets_inuse.begin(); i!=nets_inuse.end(); i++){
//        if ((*i)->inUse == false){
//          cout << "before deconstructor in list is needed";
//          (*i)->~Net();
//          nets_inuse.erase(i);
//          cout << "after deconstructor is performed";
//        }
//      }



      if(firstmodelmade == false){
        boost::unique_lock<boost::mutex> lock=boost::unique_lock<boost::mutex>(mutex);
        firstmodelmade = true;
        cond.notify_one();
        lock.unlock();
      }
      transmitted_bytes = total_size;
    
      // Save current net parameter for later use (incremental offloading)
      net->ToProto(&saved_net_param, false);
    }
  }
}

int main(int argc, char** argv) {

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
    boost::thread_group threads;
    threads.create_thread(boost::bind(upload_server, 7675));
    threads.create_thread(boost::bind(execution_server, 7676));
    threads.join_all();
  }
  catch (std::exception& e) {
    std::cerr << "Exception : " << e.what() << endl;
  }

  return 0;
}
