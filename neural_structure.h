#include <stdio.h>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <random>
#include "neural_map.h"

namespace nn {

class neural_node;
class neural_layer;
class neural_connection {
public:
    neural_connection() {}

    void update_weight(double weight) { _weight = weight; }
    void update_back_link(neural_node *node) { _back_link = node; }

    void update_weight_and_sigmoid(double weight) {
        _weight = (weight/(1+abs(weight)));
    }

    double compute_link_value();

private:
    double _weight = 0;
    neural_node *_back_link= nullptr;
};

class neural_node {

public:
    neural_node(std::mt19937 &gen,
                node_config &config)
        : _gen(gen),
          _config(config) {
        _activation_threshold = _config._activation_threshold;
    }

    void connect_back_nodes(neural_layer *l);

    void update_value(double value) { _value = value; }

    double value() { return _value; }

    void flatten_and_converge() {
        _value = (_value/(1+abs(_value))); // Apply Sigmoid.
        if (_value > _activation_threshold) {
            _value = 1;
        }
        else {
            _value = 0;
        }
    }

    void compute();

    ~neural_node() {
        for (auto &c : _connections) {
            delete c;
        }
    }

private:
    double                              _value = 0;
    double                              _activation_threshold = 0;
    std::mt19937                       &_gen;
    node_config                        &_config;
    std::vector<neural_connection *>    _connections;
};

class neural_layer {

public:
    neural_layer(std::mt19937 &gen,
                 layer_config &config)
        : _gen(gen),
          _config(config) {}

    void init() {
        _node_count = _config._node_count;
        for (int i = 0; i < _node_count; i++) {
            neural_node *node = new neural_node(_gen, _config._node_configs[i]);
            _nodes.push_back(node);
        }
    }

    void delete_nodes() {
        for (auto &n : _nodes) {
            delete n;
        }
    }

    void connect_back_layer(neural_layer *l) {
        for (auto &n : _nodes) {
            n->connect_back_nodes(l);
        }
    }

    std::vector<neural_node *> &get_nodes() { return _nodes; }

    int node_count() { return _config._node_count; }

    void enumerate(int show_nodes);

    ~neural_layer() { delete_nodes(); }

private:
    int                         _node_count = 0;
    std::mt19937               &_gen;
    std::vector<neural_node *>  _nodes;
    layer_config               &_config;
};


class neural_structure {

public:
    neural_structure(std::mt19937 &gen,
                     structure_config config)
        : _gen(gen),
          _config(config) {}

    void init() {
        _layer_count = _config._layer_count;
        for (int i = 0; i < _layer_count; i++) {
            neural_layer *layer = new neural_layer(_gen, _config._layer_configs[i]);
            layer->init();
            _layers.push_back(layer);
        }
        connect_layers();
    }

    void connect_layers() {
        neural_layer *last = nullptr;
        for (auto &l : _layers) {
            if (!last) {
                last = l;
                continue;
            }
            l->connect_back_layer(last);
            last = l;
        }
    }
        
    void delete_layers() {
        for (auto &l : _layers) {
            delete l;
        }
    }

    void fill_input_neurons(std::vector<double> &inputs);

    void compute_network();

    neural_layer *get_input_layer() { return _layers[0]; }

    void enumerate();

    ~neural_structure() { delete_layers(); }

    void describe() { _config.describe(); }

private:
    int                              _layer_count = 0;
    std::mt19937                   &_gen;
    std::vector<neural_layer *>     _layers;
    structure_config                _config;
};

}
