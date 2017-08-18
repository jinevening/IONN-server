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

#define BUFF_SIZE 256*1024*1024
#define PORT 7675

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

using boost::asio::ip::tcp;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
  shared_ptr<ExecutionGraph> graph_;
};

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";

  /* Create execution graph */
  graph_.reset(new ExecutionGraph(net_.get()));
  graph_->printLayers();
  graph_->createTimeExecutionGraph();
  graph_->createEnergyExecutionGraph();

  /* Get partitioning points of best path */
  list<pair<int, int> > server_part_time;
  list<pair<int, int> > server_part_energy;
  graph_->getBestPathForTime(&server_part_time);
  graph_->getBestPathForEnergy(&server_part_energy);
  list< pair<int, int> >::iterator i;
  cout << "Partitioning points for time optimization (offloading point, resume point)" << endl;
  for (i = server_part_time.begin(); i != server_part_time.end(); ++i) {
    cout << "(" << (*i).first << ", " << (*i).second << ")" << endl;
  }
  cout << "Partitioning points for energy optimization (offloading point, resume point)" << endl;
  for (i = server_part_energy.begin(); i != server_part_energy.end(); ++i) {
    cout << "(" << (*i).first << ", " << (*i).second << ")" << endl;
  }
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

void server(boost::asio::io_service& io_service, unsigned short port){
  cout << "Partitioning Server Started on Port " << port << endl;

  // Huge buffer. We will use this again and again.
  unsigned char* buffer = new unsigned char[BUFF_SIZE];

  for (;;) {
    tcp::acceptor a(io_service, tcp::endpoint(tcp::v4(), port));
    tcp::socket sock(io_service);
    a.accept(sock);

    for (;;) {
      memset(buffer, 0, BUFF_SIZE);
      unsigned char* buffer_ptr = buffer;

      int total_size = 0;	// model + feature
      int model_size = 0;

      boost::system::error_code error;
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
  //	cout << "Received data so far : " << buffer_ptr - buffer << endl;
	} while ((buffer_ptr - buffer) < total_size + 8 );
      }
      catch (std::exception& e) {
	std::cerr << "Exception in thread: " << e.what() << "\n";
	return;
      }

      if (error == boost::asio::error::eof)
	break;

      cout << "Received " << buffer_ptr - buffer << " bytes" << endl;

      CHECK_EQ(sizeof(int), 4);
      memcpy(&model_size, buffer + 4, 4);

      // Decode received data
      NetParameter net_param;
      BlobProto feature;
      if (!(net_param.ParseFromArray(buffer + 8, model_size))) {
	perror("Protobuf network decoding failed");
	exit(EXIT_FAILURE);
      }
      if (!(feature.ParseFromArray(buffer + 8 + model_size, total_size - model_size))) {
	perror("Protobuf feature decoding failed");
	exit(EXIT_FAILURE);
      }

      // Initialize received network
      Net<float> net(net_param);;
      Blob<float>* input_layer = net.input_blobs()[0];
      input_layer->FromProto(feature, true);

      // Run forward
      net.Forward();

      // Get output data
      Blob<float>* output_layer = net.output_blobs()[0];
      BlobProto output_proto;
      output_layer->ToProto(&output_proto);
      int output_size = output_proto.ByteSize();
      memcpy(buffer, &output_size, 4);
      output_proto.SerializeWithCachedSizesToArray(buffer + 4);
      // send(new_socket , buffer , output_size , 0 );
      cout << "Output blob size : " << output_size << " bytes" << endl;

      // Send output data to the client
      int sent_bytes = boost::asio::write(sock, boost::asio::buffer(buffer, output_size + 4));
      cout << "Sent " << sent_bytes << " bytes" << endl;
    }

    // Connection closed by client
    cout << "Client closed the connection" << endl;
  }
}

int main(int argc, char** argv) {

  google::InitGoogleLogging(argv[0]);
  try {
    boost::asio::io_service io_service;
    server(io_service, PORT);
  }
  catch (std::exception& e) {
    std::cerr << "Exception : " << e.what() << endl;
  }

  return 0;
}
