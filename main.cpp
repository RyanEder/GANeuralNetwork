#include <stdio.h>
#include <vector>
#include "neural_pool.h"

int main() {
    printf("Starting Neural Net.\n");
    nn::neural_pool pool(1);

    std::vector<double> inputs;
    inputs.push_back(0.04);
    inputs.push_back(0.24);
    inputs.push_back(0.84);
    inputs.push_back(0.91);
    inputs.push_back(0.25);

    pool.init();
    pool.feed_inputs(inputs);
    //pool.enumerate_pool();
    pool.compute_pool();
    //pool.enumerate_pool();
    inputs[2] = 1;
    inputs[3] = .77;
    pool.feed_inputs(inputs);
    pool.compute_pool();

    for (int i = 0; i < 1000; i++) {
        pool.compute_pool();
    }
    //pool.enumerate_pool();
    
    printf("Complete\n");
    return 0;
}
