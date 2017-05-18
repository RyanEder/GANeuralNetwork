#include <stdio.h>
#include <random>
#include <vector>

namespace nn {

struct mutate_probabilities {
    unsigned del_layer      = 2;
    unsigned forward_conn   = 3;
    unsigned add_layer      = 5;
    unsigned del_node       = 5;
    unsigned zero_conn      = 5;
    unsigned add_node       = 10;
    unsigned mut_strength   = 10;
    unsigned nothing        = 15;
    unsigned mod_thresh     = 20;
    unsigned mod_weight     = 25;
};

struct node_config {
    double _activation_threshold = 0;
    std::vector<double> _connection_weights;
};

struct layer_config {
    int _node_count = 0;

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
            int prev_layer_count = 0;
            int output_layer_index = _layer_count - 1;
            int count = 0;
            for (int i = 0; i < _layer_count; i++) {
                _layer_configs.push_back(layer_config());
                if (i == 0)                         count = _input_neuron_count;
                else if (i == output_layer_index)   count = _output_neuron_count;
                else                                count = node_count(_gen);
                _layer_configs[i]._node_count = count;
                for (int j = 0; j < count; j++) {
                    _layer_configs[i]._node_configs.push_back(node_config());
                    _layer_configs[i]._node_configs[j]._activation_threshold = threshold(_gen);
                    for (int l = 0; l < prev_layer_count; l++) {
                        _layer_configs[i]._node_configs[j]._connection_weights.push_back(weight(_gen));
                    }
                }
                prev_layer_count = count;
            }
        }

        void describe() {
            printf("Neural Structure: %d layers.\n", _layer_count);
            int layer = 0;
            for (auto &l : _layer_configs) {
                int node = 0;
                printf("\tLayer %d has %d nodes\n", layer++, l._node_configs.size());
                for (auto &n : l._node_configs) {
                    printf("\t\tNode %d: Threshold: %.2f\n", node++, n._activation_threshold);
                }
            }
        }

        int                 _layer_count = 0; 
        int                 _input_neuron_count = 0;
        int                 _output_neuron_count = 0;
        std::mt19937       &_gen;

        // index: layer, value: layer config
        std::vector<layer_config> _layer_configs;
};

}
