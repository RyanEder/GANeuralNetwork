#include <stdio.h>
#include <random>
#include <vector>
#include <cstdint>

namespace nn {

struct mutate_probabilities {
    uint32_t del_layer      = 2;
    uint32_t forward_conn   = 3;
    uint32_t add_layer      = 5;
    uint32_t del_node       = 5;
    uint32_t zero_conn      = 5;
    uint32_t add_node       = 10;
    uint32_t mut_strength   = 10;
    uint32_t nothing        = 15;
    uint32_t mod_thresh     = 20;
    uint32_t mod_weight     = 25;
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
        structure_config(std::mt19937 &gen) : _gen(gen) {}

        void random() {
            std::uniform_real_distribution<> layer_count(2, 7);
            std::uniform_real_distribution<> node_count(5, 10);
            std::uniform_real_distribution<> threshold(.5, 1);
            std::uniform_real_distribution<> weight(0, 1);
            _layer_count = layer_count(_gen);
            uint32_t prev_layer_count = 0;
            uint32_t output_layer_index = _layer_count - 1;
            uint32_t count = 0;
            for (uint32_t i = 0; i < _layer_count; i++) {
                _layer_configs.push_back(layer_config());
                if (i == 0)                         count = _input_neuron_count;
                else if (i == output_layer_index)   count = _output_neuron_count;
                else                                count = node_count(_gen);
                _layer_configs[i]._node_count = count;
                for (uint32_t j = 0; j < count; j++) {
                    _layer_configs[i]._node_configs.push_back(node_config());
                    _layer_configs[i]._node_configs[j]._activation_threshold = threshold(_gen);
                    for (uint32_t l = 0; l < prev_layer_count; l++) {
                        _layer_configs[i]._node_configs[j]._connection_weights.push_back(weight(_gen));
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

        uint32_t                 _layer_count = 0; 
        uint32_t                 _input_neuron_count = 0;
        uint32_t                 _output_neuron_count = 0;
        std::mt19937       &_gen;

        // index: layer, value: layer config
        std::vector<layer_config> _layer_configs;
};

}
