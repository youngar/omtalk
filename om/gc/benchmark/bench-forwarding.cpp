#include <algorithm>
#include <benchmark/benchmark.h>
#include <cstdint>
#include <functional>
#include <iostream>
#include <iterator>
#include <om/GC/ForwardingMap.h>
#include <random>
#include <vector>

using namespace om::gc;

static ForwardingMap *map;
static std::vector<void *> *vec;

static void BM_ForwardingMapRandomLookup(benchmark::State &state) {

  if (state.thread_index == 0) {
    // std::random_device device;
    // std::mt19937 engine{device()};
    // std::uniform_int_distribution<uintptr_t> dist;
    // auto gen = [&dist, &engine]() { return dist(engine); };

    map = new ForwardingMap;
    // std::vector<int> vec(state.range(0));
    vec = new std::vector<void *>(state.range(0));
    for (std::size_t i = 0; i < vec->size(); i++) {
      (*vec)[i] = reinterpret_cast<void *>(0x1000 + 16 * i);
    }
    map->rebuild(vec->begin(), vec->size());
    benchmark::ClobberMemory();
  }

  for (auto _ : state) {
    for (auto v : *vec) {
      map->at(v);
    }
  }

  if (state.thread_index == 0) {
    delete map;
    delete vec;
  }
}

BENCHMARK(BM_ForwardingMapRandomLookup)->Arg(0x10000)->Threads(1);
BENCHMARK(BM_ForwardingMapRandomLookup)->Arg(0x10000)->Threads(2);
