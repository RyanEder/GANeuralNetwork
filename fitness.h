#include <assert.h>
#include "neural_pool.h"

namespace nn {

class fitness_measure {
public:
    fitness_measure(neural_pool &pool) : _pool(pool) {}

    double evaluate_fitness() { 
        std::vector<double> inputs;
        inputs.push_back(0.04);
        inputs.push_back(0.24);
        inputs.push_back(0.84);
        inputs.push_back(0.91);
        inputs.push_back(0.25);

        printf("Fitness called\n");
        std::vector<neural_structure *> &structures = _pool.get_structures();
        while(1) {
            for (const auto &s : structures) {
                neural_layer *output = s->get_output_layer();
                if (output->get_nodes()[2]->value() == 1) {
                //for (const auto &n : output->get_nodes()) {
                //    if (n->value() == 1) {
                        printf("Decision made!\n");
                        return 1;
                    }
                //}
            }
            for (auto &s : structures) {
                s->mutate();
                _pool.feed_inputs(inputs);

            }
            _pool.compute_pool();
            //printf("Recomputation\n");
            //_pool.enumerate_pool();
        }
        return 0;
    }    

private:
    neural_pool &_pool;
};

};
