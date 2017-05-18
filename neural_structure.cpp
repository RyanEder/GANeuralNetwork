#include <stdio.h>
#include "neural_structure.h"

namespace nn {

class neural_layer;

double
neural_connection::compute_link_value() {
    return _back_link->value() * _weight;
}

void 
neural_node::connect_back_nodes(neural_layer *l) {
    int index = 0;
    for (auto &node : l->get_nodes()) {
        neural_connection *connection = new neural_connection();
        connection->update_back_link(node);
        connection->update_weight_and_sigmoid(_config._connection_weights[index++]);
        _connections.push_back(connection);
    }
}

void
neural_node::compute() {
    _value = 0;
    for (auto &connection : _connections) {
        _value += connection->compute_link_value();
    }
    flatten_and_converge();
}

void 
neural_structure::fill_input_neurons(std::vector<double> &inputs) {
    assert((size_t)inputs.size() == (size_t)_layers[0]->node_count());
    std::vector<neural_node *> &input_nodes = _layers[0]->get_nodes();
    int input_size = inputs.size();
    for (int i = 0; i < input_size; i++) {
        input_nodes[i]->update_value(inputs[i]);
    }
}


void
neural_structure::compute_network() {
    for (int layer_index = 1; layer_index < _layer_count; layer_index++) {
        std::vector<neural_node *> &nodes = _layers[layer_index]->get_nodes();
        for (auto &node : nodes) {
            node->compute();
        }
    }
}



// Enumerate
void
neural_layer::enumerate(int show_nodes = false) {
    for (auto &n : _nodes) {
        if (show_nodes)
            printf("(%x)%.2f ", (unsigned int)n, n->value());
        else
            printf("%.2f ", n->value());
    }
    printf("\n");
}

void
neural_structure::enumerate() {
    int i = 0;
    for (auto &l : _layers) {
        printf("Layer %d (%x): ", i++, (unsigned int)l);
        l->enumerate();
    }
    printf("\n");
}

}
