#pragma once
#include <stdio.h>
#include <random>
#include <vector>
#include <cstdint>

namespace nn {

struct mutation_chart {
    uint32_t del_layer      = 100;
    uint32_t zero_conn      = 98;
    uint32_t add_layer      = 95;
    uint32_t del_node       = 90;
    uint32_t invert_conn    = 85;
    uint32_t add_node       = 80;
    uint32_t mut_strength   = 70;
    uint32_t mod_weight     = 60;
    uint32_t mod_thresh     = 35;
    uint32_t nothing        = 15;
};

struct node_config {
    double _activation_threshold = 0;
    std::vector<double> _connection_weights;
};

struct layer_config {
    uint32_t _node_count = 0;

    // index: node, value: node config
    std::vector<node_config> _node_configs;
};

class structure_config {

public:
    structure_config(std::mt19937 &gen,
                     uint32_t max_layer_count = 7,
                     uint32_t max_node_count = 10,
                     double min_threshold = .33)
    : _gen(gen), _mutate_attribute_distribution(0, 100),
      _layer_count_distribution(2, max_layer_count),
      _node_count_distribution(5, max_node_count),
      _threshold_distribution(min_threshold, 1),
      _weight_distribution(0, 1),
      _negative_selector(-1, 1) {}

    void random() {
        _layer_count = _layer_count_distribution(_gen);
        uint32_t prev_layer_count = 0;
        uint32_t output_layer_index = _layer_count - 1;
        uint32_t count = 0;
        for (uint32_t i = 0; i < _layer_count; i++) {
            _layer_configs.push_back(layer_config());
            if (i == 0)                         count = _input_neuron_count;
            else if (i == output_layer_index)   count = _output_neuron_count;
            else                                count = _node_count_distribution(_gen);
            _layer_configs[i]._node_count = count;
            for (uint32_t j = 0; j < count; j++) {
                _layer_configs[i]._node_configs.push_back(node_config());
                _layer_configs[i]._node_configs[j]._activation_threshold = _threshold_distribution(_gen);
                for (uint32_t l = 0; l < prev_layer_count; l++) {
                    _layer_configs[i]._node_configs[j]._connection_weights.push_back(_weight_distribution(_gen));
                }
            }
            prev_layer_count = count;
        }
    }

    void describe() {
        printf("Neural Structure: %d layers.\n", _layer_count);
        uint32_t layer = 0;
        for (auto &l : _layer_configs) {
            uint32_t node = 0;
            printf("\tLayer %d has %lu nodes\n", layer++, l._node_configs.size());
            for (auto &n : l._node_configs) {
                printf("\t\tNode %d: Threshold: %.2f\n", node++, n._activation_threshold);
            }
        }
    }

    bool mutate(bool allow_reentry = true) {
        uint32_t mutate_attribute = _mutate_attribute_distribution(_gen);
        if (mutate_attribute <= _mutation_chart.nothing) {
            //printf("Mutate nothing.\n");
            return false;
        }
        else if (mutate_attribute <= _mutation_chart.mod_thresh) {
            //printf("Mutate a threshold.\n");
            mutate_threshold();
        }
        if (mutate_attribute <= _mutation_chart.mod_weight) { 
            //printf("Mutate a weight.\n");
            mutate_weight();
        }
        else if (mutate_attribute <= _mutation_chart.mut_strength) {
            //printf("Mutate mutation strength.\n");
            mutate_mutation_strength();
            if (allow_reentry) {
                return mutate(false);
            }
            return false;
        }
        else if (mutate_attribute <= _mutation_chart.add_node) {
            if (_layer_count == 2) return false;
            mutate_add_node();
            //printf("Mutate add node.\n");
        }
        else if (mutate_attribute <= _mutation_chart.invert_conn) {
            mutate_invert_connection();
            //printf("Mutate zero connection.\n");
        }
        else if (mutate_attribute <= _mutation_chart.del_node) {
            if (_layer_count == 2) return false;
            mutate_delete_node();
            //printf("Mutate delete node.\n");
        }
        else if (mutate_attribute <= _mutation_chart.add_layer) {
            mutate_add_layer();
            //printf("Mutate add layer.\n");
        }
        else if (mutate_attribute <= _mutation_chart.zero_conn) {
            mutate_zero_connection();
            //printf("Mutate forward connection.\n");
        }
        else if (mutate_attribute <= _mutation_chart.del_layer) {
            if (_layer_count == 2) return false;
            mutate_delete_layer();
            //printf("Mutate delete layer.\n");
        }
        else {
            printf("Unknown mutation\n");
            return false;
        }
        return true;
    }

    uint32_t get_layer_count() { return _layer_count; }

    std::vector<layer_config> &get_layer_configs() { return _layer_configs; }

    void set_input_neuron_count(uint32_t in) { _input_neuron_count = in; }
    void set_output_neuron_count(uint32_t out) { _output_neuron_count = out; }

private:

    uint32_t pick_layer(bool center_only = false) { 
        uint32_t count = _layer_count;
        if (center_only) { count--; }

        std::uniform_real_distribution<> layer_selector(1, count);
        return layer_selector(_gen);
    }

    uint32_t pick_node(uint32_t layer) {
        std::uniform_real_distribution<> node_selector(0, _layer_configs[layer]._node_count);
        return node_selector(_gen);
    }

    uint32_t pick_connection(uint32_t layer, uint32_t node) {
        uint32_t connections_len = _layer_configs[layer]._node_configs[node]._connection_weights.size();
        std::uniform_real_distribution<> connection_selector(0, connections_len);
        return connection_selector(_gen);

    }

    void mutate_weight() {
        uint32_t layer = pick_layer();
        uint32_t node  = pick_node(layer);
        uint32_t connection = pick_connection(layer, node);

        _layer_configs[layer]._node_configs[node]._connection_weights[connection] = _weight_distribution(_gen);
    }

    void mutate_threshold() {
        uint32_t layer = pick_layer();
        uint32_t node  = pick_node(layer);

        _layer_configs[layer]._node_configs[node]._activation_threshold = _threshold_distribution(_gen);
    }

    void mutate_mutation_strength() {
        int32_t percent = _mutate_attribute_distribution(_gen);
        if (_negative_selector(_gen) < 0) { percent *= -1; }

        mutation_chart old_chart = _mutation_chart;

        double movement = (double)_mutation_chart.nothing * (double)percent / 100.0;

        _mutation_chart.nothing += movement;
        double sum = _mutation_chart.nothing;
        double temp = old_chart.mod_thresh - old_chart.nothing;
        
        _mutation_chart.mod_thresh = temp - movement * temp / 100.0 + sum;
        sum += _mutation_chart.mod_thresh;
        temp = old_chart.mod_weight - old_chart.mod_thresh;

        _mutation_chart.mod_weight = temp - movement * temp / 100.0 + sum;
        sum += _mutation_chart.mod_weight;
        temp = old_chart.mut_strength - old_chart.mod_weight;

        _mutation_chart.mut_strength = temp - movement * temp / 100.0 + sum;
        sum += _mutation_chart.mut_strength;
        temp = old_chart.add_node - old_chart.mut_strength;

        _mutation_chart.add_node = temp - movement * temp / 100.0 + sum;
        sum += _mutation_chart.add_node;
        temp = old_chart.invert_conn - old_chart.add_node;

        _mutation_chart.invert_conn = temp - movement * temp / 100.0 + sum;
        sum += _mutation_chart.invert_conn;
        temp = old_chart.del_node - old_chart.invert_conn;

        _mutation_chart.del_node = temp - movement * temp / 100.0 + sum;
        sum += _mutation_chart.del_node;
        temp = old_chart.add_layer - old_chart.del_node;

        _mutation_chart.add_layer = temp - movement * temp / 100.0 + sum;
        sum += _mutation_chart.add_layer;
        temp = old_chart.zero_conn - old_chart.add_layer;

        _mutation_chart.zero_conn = temp - movement * temp / 100.0 + sum;

        // Leave del_layer where it was since it's at the top.
    }

    void mutate_add_node() {
        bool center_only = true;
        uint32_t layer = pick_layer(center_only);
        _layer_configs[layer]._node_count++;
        _layer_configs[layer]._node_configs.push_back(node_config());
        _layer_configs[layer]._node_configs.back()._activation_threshold = _threshold_distribution(_gen);
        for (uint32_t l = 0; l < _layer_configs[layer - 1]._node_count; l++) {
            _layer_configs[layer]._node_configs.back()._connection_weights.push_back(_weight_distribution(_gen));
        }
    }

    void mutate_invert_connection() {
        uint32_t layer = pick_layer();
        uint32_t node = pick_node(layer);
        uint32_t connection = pick_connection(layer, node);
        _layer_configs[layer]._node_configs[node]._connection_weights[connection] = 1 - _layer_configs[layer]._node_configs[node]._connection_weights[connection];
    }

    void mutate_delete_node() {
        bool center_only = true;
        uint32_t layer = pick_layer(center_only);
        uint32_t node = pick_node(layer);
        _layer_configs[layer]._node_count--;
        _layer_configs[layer]._node_configs.erase(_layer_configs[layer]._node_configs.begin() + node);
    }

    void mutate_add_layer() {
        bool center_only = true;
        uint32_t layer = pick_layer(center_only);
        _layer_count++;
        _layer_configs.insert(_layer_configs.begin() + layer, layer_config());
        uint32_t count = _node_count_distribution(_gen);
        _layer_configs[layer]._node_count = count;
        for (uint32_t j = 0; j < count; j++) {
            _layer_configs[layer]._node_configs.push_back(node_config());
            _layer_configs[layer]._node_configs[j]._activation_threshold = _threshold_distribution(_gen);
            for (uint32_t l = 0; l < _layer_configs[layer - 1]._node_count; l++) {
                _layer_configs[layer]._node_configs[j]._connection_weights.push_back(_weight_distribution(_gen));
            }
        }
    }

    void mutate_zero_connection() {
        uint32_t layer = pick_layer();
        uint32_t node = pick_node(layer);
        uint32_t connection = pick_connection(layer, node);
        _layer_configs[layer]._node_configs[node]._connection_weights[connection] = 0;
    }

    void mutate_delete_layer() {
        bool center_only = true;
        uint32_t layer = pick_layer(center_only);
        _layer_count--;
        _layer_configs.erase(_layer_configs.begin() + layer);

    }

    uint32_t                 _layer_count = 0; 
    uint32_t                 _input_neuron_count = 0;
    uint32_t                 _output_neuron_count = 0;
    std::mt19937            &_gen;
    
    std::uniform_real_distribution<> _mutate_attribute_distribution;
    std::uniform_real_distribution<> _layer_count_distribution;
    std::uniform_real_distribution<> _node_count_distribution;
    std::uniform_real_distribution<> _threshold_distribution;
    std::uniform_real_distribution<> _weight_distribution;
    std::uniform_real_distribution<> _negative_selector;

    mutation_chart           _mutation_chart;
    std::vector<layer_config> _layer_configs;
};

}
