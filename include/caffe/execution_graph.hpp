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
  ExecutionGraphLayer(const string& name_str)
   : name(name_str),
     input_feature_size(0),
     output_feature_size(0),
     model_size(0),
     exec_time_c(0.0),
     exec_time_s(0.0),
     input_s(0.0),
     output_s(0.0),
     loading_time_s(0.0)
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

  // ID mapping with real caffe layers
  int start_layer_id;
  int end_layer_id;

  // server-side loading time
  float loading_time_s;
};

struct offloadInfo {//Type used in execution graph
  int left;
  int right;
  float gain;
};

struct layerInfo {
  int left;
  int right;
  int start_layer_id;
  int end_layer_id;
  int input_feature_size;
  int output_feature_size;
  int layer_size;
};

struct layerGroupsInfo {
  list<layerInfo> layer_groups;
  float first_client_layers_execution_time;
  float total_client_execution_time;
  float total_server_execution_time;
  float total_input_feature_upload_time;
  float total_output_feature_download_time;
  float total_offloaded_model_size;
};

class SimulationClass {
  public:
//    SimulationClass(){
//      ;
//    }
    SimulationClass(const string& name_, float network_speed_, int model_size_, float client_time_, float server_time_);
    SimulationClass(const string& name_, float network_speed_);
    void setConfigInfo(bool include_loading_time_, bool group_first_layers_, bool finer_groups_, bool use_slope_, bool use_area_);
    void initializeNextLayerGroups();
    void deleteLastLayerGroups();
    void addLayerInfo(int left, int right, int start_layer_id, int end_layer_id, int input_feature_size, int output_feature_size, int layer_size);
    bool checkLayerExistInRecentLayerGroups(int left, int right);
    void doSimulation();
    
    vector<layerGroupsInfo > execution_schemes_;
    string name_;
  private:
    int model_size_;
    float network_speed_;
    float client_time_;
    float server_time_;
    bool include_loading_time_;
    bool group_first_layers_;
    bool finer_groups_;
    bool use_slope_;
    bool use_area_;

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

  explicit ExecutionGraph(Net<float>* net, float network_speed)
    : idle_watt_(3.977),
      transfer_watt_(6.611),
      compute_watt_(7.239),
      time_graph_(NULL),
      energy_graph_(NULL),
      network_speed_(network_speed * 1000.0 * 1000.0 / 8000.0 ),  // network_speed Mbps
      net_(net), 
      include_loading_time_(false), 
      group_first_layers_(false), 
      finer_groups_(false), 
      use_slope_(true), 
      use_area_(false),
      do_simulation_(false){

    // compute dominators of each layer
    computeDominatorLayers();

    // create simplified NN layers for execution graph
    // from real caffe NN layers
    setUpExecutionGraphLayers();
  }

  bool include_loading_time_;
  bool group_first_layers_;
  bool finer_groups_;
  bool use_slope_;
  bool use_area_;
  bool do_simulation_;

  vector<SimulationClass> simulationHolder;



  virtual ~ExecutionGraph() {}

  // print simplified layers
  void printLayers();

  // get best partitioning plan
  void getBestPartitioningPlan(list<pair<int, int> >* result, OptTarget opt_target);

  void getBestPartitioningPlanOld(list<pair<int, int> >* result, OptTarget opt_target);
  //void set

  // update edge weights of NN execution graph
  // model transfer cost will decrease by k
  void updateNNExecutionGraphWeight(float k, OptTarget opt_target);

  // create NN execution graph for time/energy optimization
  void createNNExecutionGraph(OptTarget opt_target);

 private:
  void getBestPathForTime(list<pair<int, int> >* result);
  void getBestPathForEnergy(list<pair<int, int> >* result);
  void getBestPathForTimeOld(list<pair<int, int> >* result);
  void getBestPathForEnergyOld(list<pair<int, int> >* result);

  void createTimeExecutionGraph();
  void createEnergyExecutionGraph();
  void updateTimeExecutionGraphWeight(float k);
  void updateEnergyExecutionGraphWeight(float k);

  // power for idle/transfer/compute
  float idle_watt_;
  float transfer_watt_;
  float compute_watt_;
  // remaining upload size, used for calculating area gain per remaining upload
  int upload_size_remaining_;

  // compute dominators to handle multiple path problem
  void computeDominatorLayers();
  void setUpExecutionGraphLayers();
  vector<int> dominators_;

  void addEdge(list<pair<int, float> > * graph, int src, int dst, float weight);
  void shortestPath(OptTarget opt_target, list<pair<int, int> >* result);
  void shortestPathOld(OptTarget opt_target, list<pair<int, int> >* result);

  float distClient(list<pair<int,float> >* graph, int left, int right);
  float distServer(list<pair<int,float> >* graph, int left, int right);
  float gain(list<pair<int,float> >* graph, int left, int right);
  float gainPerCostSCDiff(list<pair<int,float> >* graph, int left, int right, list<offloadInfo> &offloaded);
  float gainDiffPerCost(list<pair<int,float> >* graph, int left, int right, list<offloadInfo> &offloaded);
  float areaPerCost(list<pair<int,float> >* graph, int left, int right, list<offloadInfo> &offloaded);
  int modelSize(int left, int right);
  int getLeftRight(list<offloadInfo> &offloaded, int left, int right, bool getLeft);

  void createCandidate( list<pair<int,float> >* graph, 
                        list<offloadInfo > &toProcess,
                        list<offloadInfo > &candidates 
                        );
  void createCandidateArea( list<pair<int,float> >* graph, 
  list<offloadInfo > &toProcess,
  list<offloadInfo > &candidates 
  );
  offloadInfo getMaxCandidate(list<offloadInfo > &candidates, int left = -1, int right = 99);
  offloadInfo getMaxCandidateFirst(list<offloadInfo > &candidates, list<pair<int,float> >* graph);

  void insertToOffload(list<offloadInfo> &offloaded, offloadInfo toOffload); 
  void removeCandidates( list<offloadInfo > &candidates, int left, int right);
  void updateToProcess(list<offloadInfo> &toProcess, int left, int right);
  void updateCandidates(list<offloadInfo> &candidates, int left, int right, 
                        list<pair<int,float> >* graph, list<offloadInfo> &offloaded);
  void updateCandidates(list<offloadInfo> &candidates, list<pair<int,float> >* graph, list<offloadInfo> &offloaded);

  
  // adjacency list graph implementation
  list<pair<int, float> > * time_graph_;					// execution graph for migrating
  list<pair<int, float> > * energy_graph_;					// execution graph for migrating
  float network_speed_;
  Net<float>* net_;
  vector<ExecutionGraphLayer*> graph_layers_;
};

}  // namespace caffe

#endif  // CAFFE_EXECUTION_GRAPH_HPP_
