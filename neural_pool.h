#include <unordered_map>
#include <ctime>
#include <thread>
#include <chrono>
#include "neural_structure.h"

#ifndef __linux__
#include "mingw.thread.h"
#define SLEEP(x) std::this_thread::sleep_for(std::chrono::microseconds(x));
#else
#include <unistd.h>
#define SLEEP(x) usleep(x);
#endif


namespace nn {
class neural_pool {

public:
    neural_pool(uint32_t candidate_pool_size) :
        _size(candidate_pool_size),
        _worker_count(std::thread::hardware_concurrency())
    {}

    void init() {
        std::srand(std::time(0));
        _gen.seed(std::rand());
        for (uint32_t i = 0; i < _size; i++) {
            structure_config config(_gen);
            config._input_neuron_count = 5;
            config._output_neuron_count = 3;
            config.random();
            //config.describe();
            neural_structure *s = new neural_structure(_gen, config);
            s->init();
            _structures.push_back(s);
        }

        uint32_t worker_data_index = 0;
        uint32_t thread_set_size = _size / _worker_count;
        uint32_t overflow = _size % _worker_count;

        uint64_t mover = 1;
        uint64_t compute = mover << _worker_count;
        _workers_complete_indicator = compute - 1;

        for (uint32_t i = 0; i < _worker_count; i++) {
            std::vector<neural_structure *> temp;
            for (uint32_t j = 0; j < thread_set_size; j++) {
                temp.push_back(_structures[worker_data_index++]);
            }
            _worker_data.push_back(temp);
        }
        for (uint32_t j = 0; j < overflow; j++) {
            _worker_data[0].push_back(_structures[worker_data_index++]);
        }
        for (uint32_t i = 0; i < _worker_count; i++) {
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
            SLEEP(10);
        }
    }

    void enumerate_pool() {
        for (auto &s : _structures) {
            s->enumerate();
        }
    }

    void worker_thread(uint32_t i) {
        uint32_t mask = 1 << i;
        _workers_finished |= mask;
        while (!_stop_threads) {
            while (_workers_finished & mask) SLEEP(10);
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
    uint32_t           _size = 0;
    uint64_t           _worker_count = 0;
    volatile bool      _stop_threads = 0;
    volatile uint64_t  _workers_finished = 0;
    uint64_t           _workers_complete_indicator = 0;
    std::mt19937       _gen;

    std::vector<neural_structure *>                             _structures;
    std::vector<std::vector<neural_structure *> >               _worker_data;
    std::vector<std::thread>                                    _workers;
};

}
