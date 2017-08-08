#ifndef CAFFE_EXECUTION_GRAPH_HPP_
#define CAFFE_EXECUTION_GRAPH_HPP_

#include <map>
#include <list>
#include <set>
#include <string>
//#include <utility>
#include <vector>

//#include "caffe/blob.hpp"
//#include "caffe/common.hpp"
//#include "caffe/layer.hpp"
//#include "caffe/proto/caffe.pb.h"
#include "caffe/net.hpp"

using namespace std;

namespace caffe {

/**
 * @brief Layer of ExecutionGraph. inception module is one layer
 *
 */
class ExecutionGraphLayer {
 public:
  ExecutionGraphLayer(string name_str)
   : name(name_str),
     input_feature_size(0),
     output_feature_size(0),
     model_size(0),
     exec_time_c(0.0),
     exec_time_s(0.0),
     input_s(0.0),
     output_s(0.0)
    {}

  void printExecutionGraphLayer();

  // layer name
  string name;

  // feature size
  int input_feature_size;
  int output_feature_size;

  // model size of this layer (data to be transmitted)
  int model_size;

  // execution time for client and server
  float exec_time_c;
  float exec_time_s;

  // input/output of this layer
  float input_s;
  float output_s;
};

/**
 * @brief Creates an execution graph to find out the shortest path for offloading
 *
 */
class ExecutionGraph {
 public:
  explicit ExecutionGraph(Net<float>* net) {
    net_ = net;
    setUpExecutionGraphLayers();
  }
  virtual ~ExecutionGraph() {}

  void printLayers();
  void shortestPath(int src);
  void createTimeExecutionGraph();
  void createEnergyExecutionGraph();

  // average watt for transfer/compute
  float transfer_watt;
  float compute_watt;

 private:
  void setUpExecutionGraphLayers();

  void addEdge(int src, int dst, float weight);
  
  // adjacency list graph implementation
  list<pair<int, float> > * graph_;
  int num_vertices_;
  vector<ExecutionGraphLayer*> graph_layers_;
  Net<float>* net_;
  map<string, int> layer_names_index_;

};

class TimeExecutionGraph : public ExecutionGraph {
 public:
  explicit TimeExecutionGraph(Net<float>* net)
    : ExecutionGraph(net) {}
  ~TimeExecutionGraph() {}
};

class EnergyExecutionGraph : public ExecutionGraph {
 public:
  explicit EnergyExecutionGraph(Net<float>* net)
    : ExecutionGraph(net) {}
  ~EnergyExecutionGraph() {}

};

}  // namespace caffe

#endif  // CAFFE_EXECUTION_GRAPH_HPP_
