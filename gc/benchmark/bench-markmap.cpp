#include <benchmark/benchmark.h>
#include <omtalk/Heap.h>

using namespace omtalk::gc;

RegionMap *map;
static void BM_Markmap(benchmark::State &state) {

  if (state.thread_index == 0) {
    map = new RegionMap();
  }

  for (auto _ : state) {
    map->mark(toHeapIndex(0x100));
  }

  if (state.thread_index == 0) {
    delete map;
  }
}
BENCHMARK(BM_Markmap)->Threads(2);
