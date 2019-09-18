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
  enum OptTarget {
    TIME = 0,
    ENERGY = 1
  };

  explicit ExecutionGraph(Net<float>* net)
    : transfer_watt(1.0),
      compute_watt(1.0),
      time_graph_(NULL),
      energy_graph_(NULL),
      net_(net) {
//    net_ = net;
    // create simplified NN layers for execution graph
    // from real caffe NN layers
    setUpExecutionGraphLayers();
  }

  virtual ~ExecutionGraph() {}

  // print simplified layers
  void printLayers();

  // print best path for time/energy
  void getBestPathForTime(list<pair<int, int> >* result);
  void getBestPathForEnergy(list<pair<int, int> >* result);

  // create NN execution graph for time/energy optimization
  void createTimeExecutionGraph();
  void createEnergyExecutionGraph();

  // average watt for transfer/compute
  float transfer_watt;
  float compute_watt;

 private:
  void setUpExecutionGraphLayers();

  void addEdge(list<pair<int, float> > * graph, int src, int dst, float weight);
  void shortestPath(OptTarget opt_target, list<pair<int, int> >* result);
  
  // adjacency list graph implementation
  list<pair<int, float> > * time_graph_;
  list<pair<int, float> > * energy_graph_;
  Net<float>* net_;
  vector<ExecutionGraphLayer*> graph_layers_;
  map<string, int> layer_names_index_;
};

}  // namespace caffe

#endif  // CAFFE_EXECUTION_GRAPH_HPP_
