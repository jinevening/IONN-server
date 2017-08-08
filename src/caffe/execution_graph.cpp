

#include <queue>
#include "caffe/execution_graph.hpp"

#define INF 9999999.0

using namespace std;

namespace caffe {

void ExecutionGraph::setUpExecutionGraphLayers() {
  const vector<shared_ptr<Layer<float> > >& layers = net_->layers();
  const vector<string>& layer_names = net_->layer_names();
  int num_layers = layers.size();

  int layer_id = 0;
  ExecutionGraphLayer* current_layer = NULL;
  for (int i = 0; i < num_layers; i++) {
    const string& layer_full_name = layer_names[i];
    string layer_name = layer_full_name.substr(0, layer_full_name.find("/"));

    // Skip split layers
    // Because split layers do not have useful information but give wrong output param numbers
    if (layer_full_name.find("split") != string::npos) {
      continue;
    }

    if (layer_names_index_.find(layer_name) == layer_names_index_.end()) {
      // create a new layer and set it as the current layer
      current_layer = new ExecutionGraphLayer(layer_name);
      graph_layers_.push_back(current_layer);
      layer_names_index_[layer_name] = layer_id++;

      // set input feature size of current layer
      for (int j = 0; j < net_->bottom_vecs()[i].size(); j++) {
        current_layer->input_feature_size += net_->bottom_vecs()[i][j]->count() * sizeof(float);
      }
    }

    // update model size
    for (int j = 0; j < layers[i]->blobs().size(); j++) {
      current_layer->model_size += layers[i]->blobs()[j]->count() * sizeof(float);
    }

    // update output feature size
    int output_f_size = 0;
    for (int j = 0; j < net_->top_vecs()[i].size(); j++) {
      output_f_size += net_->top_vecs()[i][j]->count() * sizeof(float);
    }
    current_layer->output_feature_size = output_f_size;

    // update execution time
    current_layer->exec_time_c += 1.0;
    current_layer->exec_time_s += 1.0;
  }
}
void ExecutionGraph::printLayers() {
  cout << "There are " << graph_layers_.size()  << " layers" << endl;
  for (int i = 0; i < graph_layers_.size(); i++) {
//    cout << graph_layers_[i]->name << endl;
    graph_layers_[i]->printExecutionGraphLayer();
  }
}

void ExecutionGraphLayer::printExecutionGraphLayer() {
  cout << "name : " << name
  << " input_feature: " << input_feature_size
  << " output_feature: " << output_feature_size
  << " model_size: " << model_size
  << " exec_time_c: " << exec_time_c
  << " exec_time_s: " << exec_time_s
  << endl;
}

void ExecutionGraph::addEdge(int src, int dst, float weight) {
  graph_[src].push_back(make_pair(dst,weight));
}

void ExecutionGraph::createTimeExecutionGraph() {
	num_vertices_ = (4 * graph_layers_.size()) + 2;
  graph_ = new list<pair<int, float> >[num_vertices_];   // +2 for input,output
  float network_speed = 1.0;

  // edge from input
  addEdge(0, 1, 0.0);

  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (4 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    float output_feature_size = graph_layers_[i]->output_feature_size;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_c = graph_layers_[i]->exec_time_c;
    float exec_time_s = graph_layers_[i]->exec_time_s;

    // set edges inside a layer
    addEdge(idx, idx + 1, (static_cast<float>(model_size) + input_feature_size)/network_speed);
    addEdge(idx + 1, idx + 2, exec_time_s);
    addEdge(idx + 2, idx + 3, output_feature_size/network_speed);
    addEdge(idx, idx + 3, exec_time_c);

    // set edges in-between layers
    // client route
    addEdge(idx + 3, idx + 4, 0.0);

    // server route
    if (idx > 2)
      addEdge(idx - 2, idx + 1, static_cast<float>(model_size)/network_speed);
  }
}

void ExecutionGraph::createEnergyExecutionGraph() {
	num_vertices_ = (3 * graph_layers_.size()) + 2;
  graph_ = new list<pair<int, float> >[num_vertices_];   // +2 for input,output
  float network_speed = 1.0;

  // edge from input
  addEdge(0, 1, 0.0);

  for (int i = 0; i < graph_layers_.size(); i++) {
    int idx = (3 * i) + 1;
    float input_feature_size = graph_layers_[i]->input_feature_size;
    float output_feature_size = graph_layers_[i]->output_feature_size;
    int model_size = graph_layers_[i]->model_size;
    float exec_time_c = graph_layers_[i]->exec_time_c;

    // set edges inside a layer
    addEdge(idx, idx + 1, transfer_watt * ((static_cast<float>(model_size) + input_feature_size)/network_speed));
    addEdge(idx + 1, idx + 2, transfer_watt * (output_feature_size/network_speed));
    addEdge(idx, idx + 2, compute_watt * exec_time_c);

    // set edges in-between layers
    // client route
    addEdge(idx + 2, idx + 3, 0.0);

    // server route
    if (idx > 2)
      addEdge(idx - 2, idx + 1, transfer_watt * (static_cast<float>(model_size)/network_speed));
  }
}

typedef pair<float, int> fiPair;

void ExecutionGraph::shortestPath(int src) {
	int V = num_vertices_;

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
		for (i = graph_[u].begin(); i != graph_[u].end(); ++i) {
			// Get vertex label and weight of current adjacent of u
			int v = (*i).first;
			int weight = (*i).second;

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
	cout << "Vertex   Distance from Source" << endl;
	for (int i = 0; i < V; ++i)
			cout << i << "\t\t" << dist[i] << endl;

	cout << "Shortest path from src to dst" << endl;
	int node = V-1;
	while (node != src) {
		cout << node << " ";
		node = path[node];
	}
	cout << node << endl;
}

}  // namespace caffe
