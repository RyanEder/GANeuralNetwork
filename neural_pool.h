#include <stdio.h>
#include <unordered_map>
#include <random>
#include <ctime>
#include <thread>
#include <chrono>
#include "mingw.thread.h"
#include "neural_structure.h"

namespace nn {
class neural_pool {

public:
    neural_pool(int candidate_pool_size) :
        _size(candidate_pool_size),
        _worker_count(std::thread::hardware_concurrency())
    {}

    void init() {
        std::srand(std::time(0));
        _gen.seed(std::rand());
        for (int i = 0; i < _size; i++) {
            structure_config config(_gen);
            config._input_neuron_count = 5;
            config._output_neuron_count = 3;
            config.random();
            //config.describe();
            neural_structure *s = new neural_structure(_gen, config);
            s->init();
            _structures.push_back(s);
        }

        int worker_data_index = 0;
        int thread_set_size = _size / _worker_count;
        int overflow = _size % _worker_count;

        _workers_complete_indicator = (1 << _worker_count) - 1;

        for (int i = 0; i < _worker_count; i++) {
            std::vector<neural_structure *> temp;
            for (int j = 0; j < thread_set_size; j++) {
                temp.push_back(_structures[worker_data_index++]);
            }
            _worker_data.push_back(temp);
        }
        for (int j = 0; j < overflow; j++) {
            _worker_data[0].push_back(_structures[worker_data_index++]);
        }
        for (int i = 0; i < _worker_count; i++) {
            _workers.push_back(std::thread(&neural_pool::worker_thread, this, i));
        }
        wait_for_workers();
    }

    void feed_inputs(std::vector<double> &inputs) {
        for (auto &s : _structures) {
            s->fill_input_neurons(inputs);
        }
    }

    void compute_pool() { 
        //for (auto &s : _structures) {
        //    s->compute_network();
        //}
        _workers_finished = 0;
        wait_for_workers();
    }

    void wait_for_workers() {
        while (true) {
            if (_workers_finished == _workers_complete_indicator) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void enumerate_pool() {
        for (auto &s : _structures) {
            s->enumerate();
        }
    }

    void worker_thread(int i) {
        int mask = 1 << i;
        _workers_finished |= mask;
        while (!_stop_threads) {
            while (_workers_finished & mask) std::this_thread::sleep_for(std::chrono::microseconds(100));
            if (_stop_threads) return;
            for (auto &s : _worker_data[i]) {
                s->compute_network();
            }
            _workers_finished |= mask;
        }
    }

    ~neural_pool() {
        _stop_threads = 1;
        _workers_finished = 0;
        for (auto &s : _structures) delete s;
        for (auto &t : _workers)    t.join();
    }

private:
    int                _size = 0;
    int                _worker_count = 0;
    volatile bool      _stop_threads = 0;
    volatile int       _workers_finished = 0;
    int                _workers_complete_indicator = 0;
    std::mt19937       _gen;

    std::vector<neural_structure *>                             _structures;
    std::vector<std::vector<neural_structure *> >               _worker_data;
    std::vector<std::thread>                                    _workers;
};

}
