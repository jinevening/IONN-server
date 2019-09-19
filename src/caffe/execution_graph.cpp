

#include <queue>
#include <algorithm>
#include "caffe/execution_graph.hpp"

#define INF 9999999.0

using namespace std;

namespace caffe {

static vector<int> intersection(const vector<vector<int>* > *vecs) {
	vector<int> last_intersection = *(vecs->at(0));
	vector<int> curr_intersection;

	for (size_t i = 1; i < vecs->size(); ++i) {
		set_intersection(last_intersection.begin(), last_intersection.end(),
			vecs->at(i)->begin(), vecs->at(i)->end(),
			back_inserter(curr_intersection));
		swap(last_intersection, curr_intersection);
		curr_intersection.clear();
	}
	return last_intersection;
}

void ExecutionGraph::computeDominatorLayers() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  int num_layers = layers.size();
  int output_layer = num_layers - 1;

  // Set initial value
  for (int i = 0; i < num_layers; i++) {
    vector<int>* dominator = layers[i]->dominator();
    if (i == 0)
      dominator->push_back(0);
    else {
      for(int j = 0; j < num_layers; j++) {
        dominator->push_back(j);
      }
    }
  }

  // Get dominators
  bool changed;
  do {
    changed = false;
    for (int i = 1; i < num_layers; i++) {
      // Get intersection of parents' dom + current layer
      int num_parents = layers[i]->layer_param().bottom_size();
			vector<int> parents_layers;
      for (int j = 0; j < num_parents; j++) {
        string bottom_blob = layers[i]->layer_param().bottom(j);
        if (bottom_blob.find("split") != string::npos) {
          // This is for special characteristic of 'split' layer
          // hard coding.. but simple
          bottom_blob = bottom_blob.substr(0, bottom_blob.length() - 2);
        }
        int layer_id = net_->layer_id_by_name(bottom_blob);
        if (layer_id == -1 && bottom_blob.find("data") != string::npos) {
          // This is for special characteristic of 'data' layer
					layer_id = 0;
				}
				CHECK_GE(layer_id, 0);
        parents_layers.push_back(layer_id);
      }
      vector<vector<int>* > parents_doms;
      for (int j = 0; j < parents_layers.size(); j++) {
				parents_doms.push_back(layers[parents_layers[j]]->dominator());
			}
			vector<int> result = intersection(&parents_doms);
    	result.push_back(i);

			// Compare the result with the original dom list
			// If changed, update dom list
			vector<int> v;
			vector<int>* curr_dom = layers[i]->dominator();
			set_difference(curr_dom->begin(), curr_dom->end(), result.begin(), result.end(), inserter(v, v.begin()));
			if (v.size() > 0) {
				changed = true;
				curr_dom->assign(result.begin(), result.end());
			}
		}
  } while (changed);

	// Print dominators of output layer
	cout << "Dominators of output layer:";
	vector<int>* output_dom = layers[output_layer]->dominator();
  // remove output layer
  output_dom->pop_back();

  // i = 1 -> remove input layer
  for (int i = 1; i < output_dom->size(); i++) {
    if (strcmp(layers[output_dom->at(i)]->type(), "Split") != 0) {
      dominators_.push_back(output_dom->at(i));
      cout << " " << output_dom->at(i);
	  }
  }
	cout << endl;

	cout << "# of Dominators: " << dominators_.size() << endl;

  // Clear dominators
  // (This should be chagned.
  // we can allocate dominators in this function as local variables
  // so that they can be cleared naturally)
  for (int i = 0; i < num_layers; i++) {
    layers[i]->dominator()->clear();
  }
}

void ExecutionGraph::setUpExecutionGraphLayers() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();
  int num_layers = layers.size();

  ExecutionGraphLayer* current_layer = NULL;

  // ID of next dominator
  int next_dom_idx = dominators_.size() - 1;
  int next_dom = dominators_[next_dom_idx];

  // Create layers from output to input layer
  for (int i = num_layers - 2; i > 0 ; i--) {
    // Skip split layers
    // Because split layers do not have useful information but give wrong output param numbers
    if (strcmp(layers[i]->type(), "Split") == 0) {
      continue;
    }

    if (i == next_dom) {
      const string& layer_name = layer_names[i];

      // Create a new execution graph layer
      current_layer = new ExecutionGraphLayer(layer_name);
      current_layer->end_layer_id = i;

      // set output feature size of current layer
      for (int j = 0; j < net_->top_vecs()[i].size(); j++) {
        current_layer->output_feature_size += net_->top_vecs()[i][j]->count(1) * sizeof(float);
      }

			// push front
		  vector<ExecutionGraphLayer*>::iterator it = graph_layers_.begin();
			it = graph_layers_.insert (it , current_layer);

      // update next_dom (take care of out-of-range case)
      if (--next_dom_idx >= 0)
        next_dom = dominators_[next_dom_idx];
    }

    current_layer->start_layer_id = i;

    // update model size
    for (int j = 0; j < layers[i]->blobs().size(); j++) {
      current_layer->model_size += layers[i]->blobs()[j]->count() * sizeof(float);
    }

    // update input feature size
    int input_f_size = 0;
    for (int j = 0; j < net_->bottom_vecs()[i].size(); j++) {
      input_f_size += net_->bottom_vecs()[i][j]->count(1) * sizeof(float);
    }
    current_layer->input_feature_size = input_f_size;

    // update execution time
    current_layer->exec_time_c += layers[i]->get_exec_time_c();
    current_layer->exec_time_s += layers[i]->get_exec_time_s();

    // update loading time
    if (strcmp(layers[i]->type(), "Convolution") == 0) {
      float filters = static_cast<float>(layers[i]->layer_param().convolution_param().num_output());
      current_layer->loading_time_s += 3.0e-3 * filters + 0.7;
    }
    else if (strcmp(layers[i]->type(), "InnerProduct") == 0) {
      float neurons = static_cast<float>(layers[i]->layer_param().inner_product_param().num_output());
      current_layer->loading_time_s += 3.0e-7 * (neurons * static_cast<float>(input_f_size)) + 1.3;
    }
    else {
      // very small loading time for other types of layers
      current_layer->loading_time_s += 0.00005;
    }
  }
}

void ExecutionGraph::printLayers() {
  cout << "There are " << graph_layers_.size()  << " layers" << endl;
  for (int i = 0; i < graph_layers_.size(); i++) {
//    cout << graph_layers_[i]->name << endl;
    graph_layers_[i]->printExecutionGraphLayer();
  }
}

//void ExecutionGraph::printEdges(ExecutionGraph::OptTarget opt_target, int i) {
//  list<pair<int, float> >* graph = opt_target == TIME ? time_graph_ : energy_graph_;
//  cout << "Layer " << i << " " << graph_layers_[i]->name << " edges" << endl;
//
//  int start_idx = opt_target == TIME ? ((i * 4) + 1) : ((i * 3) + 1);
//  graph
//}

void ExecutionGraphLayer::printExecutionGraphLayer() {
  cout << "name : " << name
  << " input_feature: " << input_feature_size
  << " output_feature: " << output_feature_size
  << " model_size: " << model_size
  << " exec_time_c: " << exec_time_c
  << " exec_time_s: " << exec_time_s
  << " loading_time_s: " << loading_time_s
  << endl;
}

void ExecutionGraph::addEdge(list<pair<int, float> >* graph, int src, int dst, float weight) {
  graph[src].push_back(make_pair(dst,weight));
  cout << "(" << src << ", " << dst << ", " << weight << ")" << endl;
}

void ExecutionGraph::createTimeExecutionGraph() {
  time_graph_ = new list<pair<int, float> >[(3 * graph_layers_.size()) + 2];   // +2 for input,output

  // edge from input
  addEdge(time_graph_, 0, 1, 0.0);

  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (3 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    float output_feature_size = graph_layers_[i]->output_feature_size;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_c = graph_layers_[i]->exec_time_c;
    float exec_time_s = graph_layers_[i]->exec_time_s;
    float loading_time_s = graph_layers_[i]->loading_time_s;

    // set edges inside a layer
    addEdge(time_graph_, idx, idx + 1, input_feature_size/network_speed_);
    if(include_loading_time_){
        addEdge(time_graph_, idx + 1, idx + 2, exec_time_s + (static_cast<float>(model_size)/network_speed_) + loading_time_s);
    }
    else{
	addEdge(time_graph_, idx + 1, idx + 2, exec_time_s + (static_cast<float>(model_size)/network_speed_));
    }
    addEdge(time_graph_, idx + 2, idx + 3, output_feature_size/network_speed_);
    addEdge(time_graph_, idx, idx + 3, exec_time_c);

    if (i > 0)
      addEdge(time_graph_, idx - 1, idx + 1, 0.0);
  }
}
void ExecutionGraph::updateNNExecutionGraphWeight(float k, OptTarget opt_target) {
	switch(opt_target) {
		case TIME:
      CHECK(time_graph_ != NULL);
      updateTimeExecutionGraphWeight(k);
      break;
    case ENERGY:
      CHECK(energy_graph_ != NULL);
      updateEnergyExecutionGraphWeight(k);
      break;
    default:
      cout << "Wrong optimization target" << endl;
      CHECK(false);
  }
}

void ExecutionGraph::createNNExecutionGraph(OptTarget opt_target) {
	switch(opt_target) {
		case TIME:
      CHECK(time_graph_ == NULL);
      createTimeExecutionGraph();
      break;
    case ENERGY:
      CHECK(energy_graph_ == NULL);
      createEnergyExecutionGraph();
      break;
    default:
      cout << "Wrong optimization target" << endl;
      CHECK(false);
  }
}

void ExecutionGraph::getBestPartitioningPlan(list<pair<int, int> >* result, OptTarget opt_target) {
	switch(opt_target) {
		case TIME:
      CHECK(time_graph_ != NULL);
      getBestPathForTime(result);
      break;
    case ENERGY:
      CHECK(energy_graph_ != NULL);
      getBestPathForEnergy(result);
      break;
    default:
      cout << "Wrong optimization target" << endl;
      CHECK(false);
  }
}

void ExecutionGraph::getBestPartitioningPlanOld(list<pair<int, int> >* result, OptTarget opt_target) {
switch(opt_target) {
  case TIME:
    CHECK(time_graph_ != NULL);
    getBestPathForTimeOld(result);
    break;
  case ENERGY:
    CHECK(energy_graph_ != NULL);
    getBestPathForEnergyOld(result);
    break;
  default:
    cout << "Wrong optimization target" << endl;
    CHECK(false);
  }
}


// model transfer cost (including model loading time at the server) is updated to (1 - k) * model_transfer_cost
// ex1 : k = 0, model transfer cost is the same with first offloading
// ex2 : k = 1, model transfer cost will be 0
// also, model transfer cost is 
void ExecutionGraph::updateTimeExecutionGraphWeight(float k) {
  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (3 * i) + 1;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_s = graph_layers_[i]->exec_time_s;
    float loading_time_s = graph_layers_[i]->loading_time_s;

    // update edege weights for transmitting a DNN model
		list< pair<int, float> >::iterator j;
		for (j = time_graph_[idx + 1].begin(); j != time_graph_[idx + 1].end(); ++j) {
			int v = (*j).first;
      if (v == idx + 2){
	if(include_loading_time_){
          (*j).second = ((static_cast<float>(model_size) * (1-k))/network_speed_) + exec_time_s + (loading_time_s * (1-k));
	}
	else{
          (*j).second = ((static_cast<float>(model_size) * (1-k))/network_speed_) + exec_time_s;
	}
      }
    }
  }
}

void ExecutionGraph::updateEnergyExecutionGraphWeight(float k) {
  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (3 * i) + 1;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_s = graph_layers_[i]->exec_time_s;
    float loading_time_s = graph_layers_[i]->loading_time_s;

    // update edege weights for transmitting a DNN model
		list< pair<int, float> >::iterator j;
		for (j = energy_graph_[idx + 1].begin(); j != energy_graph_[idx + 1].end(); ++j) {
			int v = (*j).first;
      if (v == idx + 2){
	if(include_loading_time_){
          (*j).second = transfer_watt_ * ((static_cast<float>(model_size)*(1-k))/network_speed_) + idle_watt_ * (exec_time_s + (loading_time_s * (1-k)));
	}
	else{
          (*j).second = transfer_watt_ * ((static_cast<float>(model_size)*(1-k))/network_speed_) + idle_watt_;
	}
      }
    }
  }
}

void ExecutionGraph::createEnergyExecutionGraph() {
  energy_graph_ = new list<pair<int, float> >[(3 * graph_layers_.size()) + 2];   // +2 for input,output

  // edge from input
  addEdge(energy_graph_, 0, 1, 0.0);

  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (3 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    float output_feature_size = graph_layers_[i]->output_feature_size;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_c = graph_layers_[i]->exec_time_c;
    float exec_time_s = graph_layers_[i]->exec_time_s;
    float loading_time_s = graph_layers_[i]->loading_time_s;

    // set edges inside a layer
    addEdge(energy_graph_, idx, idx + 1, transfer_watt_*(input_feature_size/network_speed_));
    if(include_loading_time_){
      addEdge(energy_graph_, idx + 1, idx + 2, transfer_watt_*(static_cast<float>(model_size)/network_speed_) + idle_watt_*(exec_time_s + loading_time_s));
    }
    else{
      addEdge(energy_graph_, idx + 1, idx + 2, transfer_watt_*(static_cast<float>(model_size)/network_speed_) + idle_watt_*(exec_time_s));
    }
    addEdge(energy_graph_, idx + 2, idx + 3, transfer_watt_ * (output_feature_size/network_speed_));
    addEdge(energy_graph_, idx, idx + 3, compute_watt_ * exec_time_c);

    // server route
    if (i > 0)
      addEdge(energy_graph_, idx - 1, idx + 1, 0.0);
  }
}

void ExecutionGraph::getBestPathForTime(list<pair<int, int> >* result) {
  shortestPath(TIME, result);
}

void ExecutionGraph::getBestPathForEnergy(list<pair<int, int> >* result) {
  shortestPath(ENERGY, result);
}

void ExecutionGraph::getBestPathForTimeOld(list<pair<int, int> >* result) {
  shortestPathOld(TIME, result);
}

void ExecutionGraph::getBestPathForEnergyOld(list<pair<int, int> >* result) {
  shortestPathOld(ENERGY, result);
}

typedef pair<float, int> fiPair;

static inline bool isServerNode(int id) {
	return (id % 3) != 1;
}

void ExecutionGraph::shortestPathOld(OptTarget opt_target, list<pair<int, int> >* result) {
  static bool holderInitialized = false;
  if(!holderInitialized){
    holderInitialized = true;
    simulationHolder.push_back(SimulationClass("Original, K method", network_speed_));
  }
  if(simulationHolder.back().name_.compare("Original, K method")!= 0){
    cout <<"Likely Error, Do original method fiiirst"<<endl;
  }

  list<pair<int, float> >* graph = NULL;
  int V = (3 * graph_layers_.size()) + 2;

	switch(opt_target) {
		case TIME:
			graph = time_graph_;
			break;
		case ENERGY:
			graph = energy_graph_;
			break;
		default:
			// unreachable
			CHECK(false);
	}

  int src = 0;

	// Save the previous node of each node to restore the shortest path
	vector<int> path(V, -1);

	// Create a priority queue to store vertices that
	// are being preprocessed
	priority_queue< fiPair, vector <fiPair> , greater<fiPair> > pq;

	// Create a vector for distances and initialize all
	// distances as infinite (INF)
	vector<float> dist(V, INF);

	// Insert source itself in priority queue and initialize
	// its distance as 0.
	pq.push(make_pair(0.0, src));
	dist[src] = 0.0;

	/* Looping till priority queue becomes empty (or all
		distances are not finalized) */
	while (!pq.empty())	{
		// The first vertex in pair is the minimum distance
		// vertex, extract it from priority queue.
		int u = pq.top().second;
		pq.pop();

		// 'i' is used to get all adjacent vertices of a vertex
		list< pair<int, float> >::iterator i;
		for (i = graph[u].begin(); i != graph[u].end(); ++i) {
			// Get vertex label and weight of current adjacent of u
			int v = (*i).first;
			float weight = (*i).second;

			//  If there is shorted path to v through u.
			if (dist[v] > dist[u] + weight) {
				// Updating distance of v
				dist[v] = dist[u] + weight;
				pq.push(make_pair(dist[v], v));
				path[v] = u;
			}
		}
	}

	// Print shortest distances stored in dist[]
//	cout << "Vertex   Distance from Source" << endl;
//	for (int i = 0; i < V; ++i)
//			cout << i << "\t\t" << dist[i] << endl;

//	cout << "Shortest path from src to dst" << endl;
	int node = V-1;
  int resume_node = 0;

  list<pair<int, int> > result_temp;
  while (node != src) {
    if (!isServerNode(node) && isServerNode(path[node])) {
      resume_node = path[node];
    }
    else if (isServerNode(node) && !isServerNode(path[node])) {

      int nodes_per_layer = 3;

      // get index of real caffe layers
      int offloading_point = graph_layers_[(path[node] - 1)/nodes_per_layer]->start_layer_id;
      int resume_point = graph_layers_[(resume_node - 1)/nodes_per_layer]->end_layer_id;

      result->push_front(make_pair(-1, -1));
      result->push_front(make_pair(offloading_point, resume_point));
    }

//  		cout << node << " ";
    node = path[node];
  }
//	  cout << node << endl;
}







/*
      result_temp.push_front(make_pair((path[node] - 1)/nodes_per_layer, (resume_node - 1)/nodes_per_layer));
    }

//  		cout << node << " ";
    node = path[node];
  }
//	  cout << node << endl;
  bool checkRepeat = true;
  for(list<pair<int, int> >::iterator i = result_temp.begin(); i != result_temp.end(); i++){
    if(!simulationHolder.back().checkLayerExistInRecentLayerGroups(i->left, i->right)){
      checkRepeat = false;
    }
  }
  if(!checkRepeat){
    for(list<pair<int, int> >::iterator i = result_temp.begin(); i != result_temp.end(); i++){
      simulationHolder.back().addLayerInfo(i->left, i->right, graph_layers_[i->left]->start_layer_id, graph_layers_[i->right]->end_layer_id, graph_layers_[i->left]->input_feature_size, graph_layers_[i->right]->output_feature_size, modelSize(i->left, i->right));
    }
    simulationHolder.back().
  }
}
*/
float ExecutionGraph::distClient(list<pair<int,float> >* graph, int left, int right) {
  int u = 1+ 3*left;
  int v = 1+ 3*right;
  float dist = 0.0;
  list< pair<int, float> >::iterator i;
  for (; u <= v; u += 3) {
    i = graph[u].begin();
    if ((*i).first != u+3){
      ++i;  //second pair contains dist of client to client, calculation
    }
    int dest = (*i).first;
    float weight = (*i).second;
    if(dest != u+3) {
      cout<<"distClient destination Error " << "from " << u << " to "<< dest << endl;
    }
    dist += weight;
  }
//  cout <<"distClient: " << dist << endl ;
  return dist;
}

float ExecutionGraph::distServer(list<pair<int,float> >* graph, int left, int right) {
  int u = 1+ 3*left;
  int v = 1+ 3*right;
  int dest;
  float weight;
  float dist = 0.0;
  list< pair<int, float> >::iterator i;

  i = graph[u].begin();
  if((*i).first != u+1){
    i++;
  }
  if((*i).first != u+1){
    cout<<"distServer destination Error";
  }
  dist += (*i).second; //dist of client to server, data
  for (int j = u; j <= v; j= j+3) {
    i = graph[j+1].begin();
/*    if((*i).first != u+2){
      i++;
    }*/
    dest = (*i).first;
    weight = (*i).second;
    if(dest != j+2) {
      cout<<"distServer destination Error";
    }
    dist += weight;  //dist of server to server, calculation
  }

//  u--;
  i = graph[v+2].begin();
  if((*i).first != v+3){
    i++;
  }
  dest = (*i).first;
  weight = (*i).second;
  if( dest != v+3) {
    cout <<"distServer destination Error";
  }
  dist += weight; //dist of server to client, data

//  cout <<"distServer: " << dist << endl ;
  return dist;
}

float ExecutionGraph::gain(list<pair<int,float> >* graph, int left, int right) {
  float speedup;
  speedup = distClient(graph, left, right) - distServer(graph, left, right);
  return speedup;
}

float ExecutionGraph::gainPerCostSCDiff(list<pair<int,float> >* graph, int left, int right, list<offloadInfo> &offloaded) {
  int l = getLeftRight(offloaded, left, right, true);
  int r = getLeftRight(offloaded, left, right, false);

//  float cost = (float)modelSize(left, right);
  float scgain = gain(graph, l, r);
//  float g = scgain/(cost/(network_speed_));
  float g = scgain/(modelSize(l,r)/(network_speed_));

  return g;
}

float ExecutionGraph::gainDiffPerCost(list<pair<int,float> >* graph, int left, int right, list<offloadInfo> &offloaded) {
  int l = getLeftRight(offloaded, left, right, true);
  int r = getLeftRight(offloaded, left, right, false);

  float prevdist = 0.0;
  int ltest=left;
  int rtest=right;
  for(list<offloadInfo>::iterator i = offloaded.begin(); i != offloaded.end(); i++) {
    if( i->left == right + 1 or i->right == left - 1){
      prevdist += distServer(graph, i->left, i->right);
      if(i->left < ltest){
        ltest=i->left;
      }
      if(i->right > rtest){
        rtest=i->right;
      }
    }
  }

  prevdist += distClient(graph, left, right);

  if( ltest != l or rtest != r){
    cout << "gainDiffPerCost wrong" << endl;
  }

//  cout << left << "," << right << " to "<<l << "," << r<< " ";
//  int cost = modelSize(l, r);
  float cost = (float)modelSize(left, right);
//  cout <<"modelsize: " << modelSize(l, r) << " netoffloadratio: "<<  gain(graph, l, r)/modelSize(l,r) << " speedup: "<<gain(graph, l, r) << " networkspeed: "<< network_speed_*1000*1000*1000/(1024*1024)<< " effperuploadspeed: "<< gain(graph,l,r)*network_speed_*1000*1000.0/(1024.0*1024*modelSize(l,r))<<endl;
  float sgain = distServer(graph, l, r);
//  cout << "serverclientdiff: "<<scgain << " prevtime: " << gainsofar<<endl;
  float g = (prevdist-sgain)/(cost/(network_speed_));
//  float g = gain(graph, l, r)*(upload_size_remaining_-cost)/cost;
  return g;
}

float ExecutionGraph::areaPerCost(list<pair<int,float> >* graph, int left, int right, list<offloadInfo> &offloaded) {
  int l = getLeftRight(offloaded, left, right, true);
  int r = getLeftRight(offloaded, left, right, false);

  float prevdist = 0.0;
  int ltest=left;
  int rtest=right;
  for(list<offloadInfo>::iterator i = offloaded.begin(); i != offloaded.end(); i++) {
    if( i->left == right + 1 or i->right == left - 1){
      prevdist += distServer(graph, i->left, i->right);
      if(i->left < ltest){
        ltest=i->left;
      }
      if(i->right > rtest){
        rtest=i->right;
      }
    }
  }

  prevdist += distClient(graph, left, right);

  if( ltest != l or rtest != r){
    cout << "gainDiffPerCost wrong" << endl;
  }

  float cost = (float)modelSize(left, right);
  float sgain = distServer(graph, l, r);
//  cout << "serverclientdiff: "<<scgain << " prevtime: " << gainsofar<<endl;
  float g = (prevdist-sgain)*(upload_size_remaining_-cost)/cost/1000.0;
//  float g = gain(graph, l, r)*(upload_size_remaining_-cost)/cost;
  return g;
}


int ExecutionGraph::modelSize(int left, int right){
  int cost = 0;
  for(int i = left; i <= right; ++i){
    cost += graph_layers_[i]->model_size;
  }
  return cost;
}

void ExecutionGraph::shortestPath(OptTarget opt_target, list<pair<int, int> >* result) {
  list<pair<int, float> >* graph = NULL;
  int V = (3 * graph_layers_.size()) + 2;
  switch(opt_target) {
    case TIME:
      graph = time_graph_;
      break;
    case ENERGY:
      graph = energy_graph_;
      break;
    default:
			// unreachable
      CHECK(false);
  }

  updateNNExecutionGraphWeight(1.0, opt_target);
//  createNNExecutionGraph(opt_target);

  int src = 0;

	// Save the previous node of each node to restore the shortest path
	vector<int> path(V, -1);

	// Create a priority queue to store vertices that
	// are being preprocessed
	priority_queue< fiPair, vector <fiPair> , greater<fiPair> > pq;

	// Create a vector for distances and initialize all
	// distances as infinite (INF)
	vector<float> dist(V, INF);

	// Insert source itself in priority queue and initialize
	// its distance as 0.
	pq.push(make_pair(0.0, src));
	dist[src] = 0.0;

	/* Looping till priority queue becomes empty (or all
		distances are not finalized) */
	while (!pq.empty())	{
		// The first vertex in pair is the minimum distance
		// vertex, extract it from priority queue.
		int u = pq.top().second;
		pq.pop();

		// 'i' is used to get all adjacent vertices of a vertex
		list< pair<int, float> >::iterator i;
		for (i = graph[u].begin(); i != graph[u].end(); ++i) {
			// Get vertex label and weight of current adjacent of u
			int v = (*i).first;
			float weight = (*i).second;

			//  If there is shorted path to v through u.
			if (dist[v] > dist[u] + weight) { 
				// Updating distance of v
				dist[v] = dist[u] + weight;
				pq.push(make_pair(dist[v], v));
				path[v] = u;
			}
		}
	}

  list<offloadInfo> toProcess;
  list<offloadInfo> candidates;
  list<offloadInfo> offloaded;
  upload_size_remaining_ = 0;
  int node1 = V-1;
  int resume_node1 = 0;
  while (node1 != src) {
    if (!isServerNode(node1) && isServerNode(path[node1])) {
      resume_node1 = path[node1];
    }
    else if (isServerNode(node1) && !isServerNode(path[node1])) {
      int nodes_per_layer = 3;
      int off_index = (path[node1]-1)/nodes_per_layer;
      int res_index = (resume_node1-1)/nodes_per_layer;

      offloadInfo t = {off_index, res_index, gainPerCostSCDiff(graph, off_index, res_index, offloaded)};
      upload_size_remaining_ += modelSize(off_index, res_index);
      toProcess.push_back(t);

    }

    node1 = path[node1];

  }
  if(use_area_ and use_slope_){
    cout << "error cant do both slope and area";
  }

  if ((use_area_ and group_first_layers_) or use_slope_) {
    createCandidate(graph, toProcess, candidates);
  }
  if ((use_area_ and group_first_layers_) or (use_slope_ and group_first_layers_)){
    offloadInfo max_candidate;
    max_candidate = getMaxCandidateFirst(candidates, graph);
//  }
//  if ((use_area_ and group_first_layers_) or (use_slope_ and group_first_layers_)) {
    if(max_candidate.left != -999){ // if valid first candidate
      insertToOffload(offloaded, max_candidate);
      int max_left = max_candidate.left;
      int max_right = max_candidate.right;
      for(list<offloadInfo>::iterator i = offloaded.begin(); i != offloaded.end(); i++) {
        int offloading_point = graph_layers_[i->left]->start_layer_id;
        int resume_point = graph_layers_[i->right]->end_layer_id;
        result->push_back(make_pair(offloading_point, resume_point));
//	if(do_simulation_){
//	  list<simulationInfo> temp_simul_data = {i->left, i->right, ,offloading_point, resume_point, graph_layers_[i->left]->input_feature_size, graph_layers_[i->right]->output_feature_size}
//	  simulation_results_.push_back(temp_simul_data);

//	}
      }
      result->push_back(make_pair(-1, -1));
      if (use_area_ and group_first_layers_){
        createCandidateArea(graph, toProcess, candidates);
      }
      removeCandidates(candidates, max_left, max_right);
      if (use_slope_){
        updateCandidates(candidates, max_left, max_right, graph, offloaded);
      }
      else if(use_area_){
	updateCandidates(candidates, graph, offloaded);
        upload_size_remaining_ -= modelSize(max_left, max_right);
      }
      updateToProcess(toProcess, max_left, max_right);
    }
  }

  if (use_area_ and not group_first_layers_){
    createCandidateArea(graph, toProcess, candidates);
  }
  if(use_slope_){
  while(!toProcess.empty()){
    offloadInfo max_candidate;
    max_candidate = getMaxCandidate(candidates);

    if(finer_groups_){
      offloadInfo smaller;
      smaller =  getMaxCandidate(candidates, max_candidate.left, max_candidate.right);
      while(smaller.left != -999){
        max_candidate = smaller;
        smaller = getMaxCandidate(candidates, max_candidate.left, max_candidate.right);
      }
    }


    insertToOffload(offloaded, max_candidate);

    for(list<offloadInfo>::iterator i = offloaded.begin(); i != offloaded.end(); i++) {
      int offloading_point = graph_layers_[i->left]->start_layer_id;
      int resume_point = graph_layers_[i->right]->end_layer_id;
      result->push_back(make_pair(offloading_point, resume_point));
    }
    result->push_back(make_pair(-1, -1));

    removeCandidates(candidates, max_candidate.left, max_candidate.right);
    updateCandidates(candidates, max_candidate.left, max_candidate.right, graph, offloaded);
    updateToProcess(toProcess, max_candidate.left, max_candidate.right);
  }
  }
  else if(use_area_){
  while(!toProcess.empty()){
    offloadInfo max_candidate;
    max_candidate = getMaxCandidate(candidates);
    cout << "from: "<< max_candidate.left << " " << max_candidate.right << ". ";
    if(finer_groups_){
    offloadInfo smaller;
    smaller =  getMaxCandidate(candidates, max_candidate.left, max_candidate.right);
    while(smaller.left != -999){
      max_candidate = smaller;
      smaller = getMaxCandidate(candidates, max_candidate.left, max_candidate.right);
    }
    }
    insertToOffload(offloaded, max_candidate);

    for(list<offloadInfo>::iterator i = offloaded.begin(); i != offloaded.end(); i++) {
      int offloading_point = graph_layers_[i->left]->start_layer_id;
      int resume_point = graph_layers_[i->right]->end_layer_id;
      result->push_back(make_pair(offloading_point, resume_point));
    }
    result->push_back(make_pair(-1, -1));

    cout << "max candidate: " << max_candidate.left << " " << max_candidate.right << 
         " " << max_candidate.gain <<  endl;
    removeCandidates(candidates, max_candidate.left, max_candidate.right);
    upload_size_remaining_ -= modelSize(max_candidate.left, max_candidate.right);
    updateCandidates(candidates, graph, offloaded);
    updateToProcess(toProcess, max_candidate.left, max_candidate.right);
  }
  }



}

void ExecutionGraph::createCandidate( list<pair<int,float> >* graph, 
                                      list<offloadInfo > &toProcess,
                                      list<offloadInfo > &candidates 
                                      )  {

  candidates.erase(candidates.begin(), candidates.end());
  list<offloadInfo >::iterator blocks;
  list<offloadInfo> empty;
  for(blocks = toProcess.begin(); blocks != toProcess.end(); ++blocks) {
    int range = blocks->right - blocks->left;
    cout << blocks->right - blocks->left << endl;
    for (int range_it = 0; range_it <= range; ++range_it){
      for(int i = blocks->left; i <= blocks->right-range_it; ++i) {
          offloadInfo temp;
          temp.left = i;
          temp.right = i+range_it;
          temp.gain = gainPerCostSCDiff(graph, temp.left, temp.right, empty);
          candidates.push_back(temp);
      }
    }
  }
}


void ExecutionGraph::createCandidateArea( list<pair<int,float> >* graph, 
                                      list<offloadInfo > &toProcess,
                                      list<offloadInfo > &candidates 
                                      )  {

  candidates.erase(candidates.begin(), candidates.end());
  list<offloadInfo >::iterator blocks;
  list<offloadInfo> empty;
  for(blocks = toProcess.begin(); blocks != toProcess.end(); ++blocks) {
    int range = blocks->right - blocks->left;
    for (int range_it = 0; range_it <= range; ++range_it){
      for(int i = blocks->left; i <= blocks->right-range_it; ++i) {
          offloadInfo temp;
          temp.left = i;
          temp.right = i+range_it;
          temp.gain = areaPerCost(graph, temp.left, temp.right, empty);
          candidates.push_back(temp);
      }
    }
  }
}


offloadInfo ExecutionGraph::getMaxCandidate(list<offloadInfo > &candidates, int left, int right){
  if(left == -1){
    float temp = -1;
    offloadInfo toReturn;
    toReturn.left = -999;
    list<offloadInfo >::iterator blocks;
    for(blocks = candidates.begin(); blocks != candidates.end(); ++blocks){
      if(blocks->gain > temp){
        temp = blocks->gain;
        toReturn = *blocks;
      }
    }
    return toReturn;
  }
  else{
    float temp = 0;
    offloadInfo toReturn;
    toReturn.left = -999;
    list<offloadInfo >::iterator blocks;
    for(blocks = candidates.begin(); blocks != candidates.end(); ++blocks){
      if(blocks->gain > temp and blocks->left >= left and blocks->right <= right and (blocks->left != left or blocks->right != right) and (modelSize(blocks->left, blocks->right) != modelSize(left, right))){
        temp = blocks->gain;
        toReturn = *blocks;
      }
    }
    return toReturn;
  }
}

offloadInfo ExecutionGraph::getMaxCandidateFirst(list<offloadInfo > &candidates, list<pair<int,float> >* graph){
  float temp = -1;
  offloadInfo toReturn;
  toReturn.left = -999;
  list<offloadInfo >::iterator blocks;
  for(blocks = candidates.begin(); blocks != candidates.end(); ++blocks){
  float speedupwithupload = blocks->gain * modelSize(blocks->left, blocks->right)/(network_speed_) - modelSize(blocks->left, blocks->right)/(network_speed_);
    if(speedupwithupload > temp and speedupwithupload > 0) {
//      cout << blocks->left<<","<<blocks->right<<":"<<blocks->gain * modelSize(blocks->left, blocks->right)<<","<<blocks->gain<<endl;
      temp = speedupwithupload;
      cout << blocks->left << ","<<blocks->right<<":"<<temp << endl;
      toReturn = *blocks;
    }
  }
  return toReturn;
}


void ExecutionGraph::insertToOffload(list<offloadInfo> &offloaded, offloadInfo toOffload){

  list<offloadInfo>::iterator it;
  for(it = offloaded.begin(); it != offloaded.end() and it->left < toOffload.left; it++);
  if(it != offloaded.begin()){
    list<offloadInfo>::iterator temp;
    temp = it;
    temp--;
    if(temp->right == toOffload.left - 1){
      if(it != offloaded.end() and it->left == toOffload.right + 1){
        temp->right = it->right;
        it = offloaded.erase(it);
      }
      else {
        temp->right = toOffload.right;
      }
    }
    else if(it!=offloaded.end() and it->left == toOffload.right + 1){
      it->left = toOffload.left;
    }
    else{
      offloaded.insert(it, toOffload);
    }
  }
  else if(it!= offloaded.end()){
    if(it->left == toOffload.right + 1){
      it->left = toOffload.left;
    }
    if(it->right == toOffload.left -1){
      it->right = toOffload.right;
    }
  }
  else{
    offloaded.push_back(toOffload);
  }
}

int ExecutionGraph::getLeftRight(list<offloadInfo> &offloaded, int left, int right, bool getLeft){
  list<offloadInfo>::iterator it;
  it  = offloaded.begin();
  int toReturn;
  if(getLeft == true){
    toReturn = left;
    for(;it != offloaded.end();++it){
      if(it->right == left - 1){
        toReturn = it->left;
      }
    }
  }
  else{//getLeft == false
    toReturn = right;
    for(;it != offloaded.end();++it){
      if(it->left == right + 1){
        toReturn = it->right;
      }
    }
  }
  return toReturn;
}

void ExecutionGraph::removeCandidates( list<offloadInfo > &candidates, int left, int right){
  int i = left;
  int j = right;
  for(list<offloadInfo>::iterator it = candidates.begin(); it != candidates.end(); ++it){
    while(it != candidates.end() and (((*it).left >= i and it->left <= j) or (it->right >= i and (*it).right <= j) or (it->left < i and it->right > j))){
      it = candidates.erase(it);
    }
  }
}

void ExecutionGraph::updateToProcess(list<offloadInfo> &toProcess, int left, int right){
  int deleted = 0;
  list<offloadInfo>::iterator it = toProcess.begin();
  while(it != toProcess.end()){
//    int right_temp = it->right;
    int left_temp = it->left;
    if(it->left < left){
      if(it->right >= left){
        if(it->right <= right){
          it->right = left - 1;
        }
        else{
          it->left = right + 1;
          offloadInfo temp = {left_temp, left - 1, 0.0};
          toProcess.insert(it, temp);
        }
      }
    }
    else if(it->left >= left and it->left <= right){
      if(it->right <= right){
        it = toProcess.erase(it);
        deleted = 1;
      }
      else if(it->right > right){
        it->left = right + 1;
      }
    }
      
    if(deleted == 0){
      it++;
    }
    else{
      deleted = 0;
    }

  }
}

void ExecutionGraph::updateCandidates(list<offloadInfo> &candidates, int left, int right, 
                                      list<pair<int,float> >* graph, list<offloadInfo> &offloaded){
  for(list<offloadInfo>::iterator it = candidates.begin(); it != candidates.end(); it++){
    if(it->left == right+1 or it->right == left-1){
      it->gain = gainDiffPerCost(graph, it->left, it->right, offloaded);
    }
  }
}

void ExecutionGraph::updateCandidates(list<offloadInfo> &candidates, list<pair<int,float> >* graph, list<offloadInfo> &offloaded){
  for(list<offloadInfo>::iterator it = candidates.begin(); it != candidates.end(); it++){
      it->gain = areaPerCost(graph, it->left, it->right, offloaded);
  }
//  cout << "updated all." << endl;
}

void SimulationClass::initializeNextLayerGroups(){
  execution_schemes_.push_back(layerGroupsInfo());
  execution_schemes_.back().first_client_layers_execution_time = -1;
  execution_schemes_.back().total_client_execution_time = -1;
  execution_schemes_.back().total_server_execution_time = -1;
  execution_schemes_.back().total_input_feature_upload_time=0;
  execution_schemes_.back().total_output_feature_download_time=0;
  execution_schemes_.back().total_offloaded_model_size=0;
}

SimulationClass::SimulationClass(const string& name_, float network_speed_, int model_size_, float client_time_, float server_time_){
  this->name_=name_;
  model_size_=model_size_;
  network_speed_=network_speed_;
  client_time_=client_time_;
  server_time_=server_time_;
}

SimulationClass::SimulationClass(const string& name_, float network_speed_){
  this->name_=name_;
  network_speed_=network_speed_;
}

void setConfigInfo(bool include_loading_time_, bool group_first_layers_, bool finer_groups_, bool use_slope_, bool use_area_){
  include_loading_time_ =  include_loading_time_;
  group_first_layers_ =    group_first_layers_;
  finer_groups_ =          finer_groups_;
  use_slope_ =             use_slope_;
  use_area_ =              use_area_;
}

void SimulationClass::deleteLastLayerGroups(){
  if(execution_schemes_.size() == 0) {
    cout << "Error of Simulation Class or wrong call" << endl;
  }
  execution_schemes_.erase(execution_schemes_.end()-1);
}

void SimulationClass::addLayerInfo(int left, int right, int start_layer_id, int end_layer_id, int input_feature_size, int output_feature_size, int layer_size) {
  layerInfo temp = {left, right, start_layer_id, end_layer_id, input_feature_size, output_feature_size, layer_size }; 
  execution_schemes_.back().layer_groups.push_back(temp);
//  execution_schemes_.back().first_client_layers_execution_time;
//  execution_schemes_.back().total_client_execution_time;
//  execution_schemes_.back().total_server_execution_time;
  execution_schemes_.back().total_input_feature_upload_time += input_feature_size / network_speed_;
  execution_schemes_.back().total_output_feature_download_time += output_feature_size / network_speed_;
  execution_schemes_.back().total_offloaded_model_size += layer_size;

}

bool SimulationClass::checkLayerExistInRecentLayerGroups(int left, int right){
  if(execution_schemes_.size() <= 0){
    return false;
  }
  for (list<layerInfo>::iterator i = execution_schemes_[execution_schemes_.size()-1].layer_groups.begin(); i != execution_schemes_[execution_schemes_.size()-1].layer_groups.end(); i++){
    if( i->left >= left and i->right <= right) {
      if(i->left > left or i-> right < right){
	cout <<"errrrrrrr";
      }
      return true;
    }
  }
  return false;
}

void SimulationClass::doSimulation(){
  cout << "Simulation with partitioning method " << name_ << " total size "<<model_size_<< " network speed(Mbps) "<< network_speed_<< " client execution time " << client_time_ << " server execution time "<<server_time_<< endl;

  float forward_finish_time = 0.0;
  float upload_finish_time = 0.0;

  for(vector<layerGroupsInfo>::iterator i = execution_schemes_.begin(); i != execution_schemes_.end(); i++){
    cout << "Offload Scheme number  "<<distance(execution_schemes_.begin(), i) << ":" << endl;

    for(list<layerInfo>::iterator j = i->layer_groups.begin(); j != i->layer_groups.end(); j++){
//      total_layer_size += j->layer_size;
      cout <<"("<< j->left<<":"<<j->start_layer_id << ","<<j->right<<":"<<j->end_layer_id<<") "<< j->input_feature_size<<j->layer_size <<" ";
    }
//    cout <<"has "<< i->total_client_execution_time <<" "<<i->total_server_execution_time<<" "<< i->total_input_feature_size<<"c,s execution time, total_input_feature_size" << endl;
  }
  return;
}

}  // namespace caffe
