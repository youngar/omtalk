#include <benchmark/benchmark.h>
#include <example/Object.h>
#include <omtalk/GlobalCollector.h>
#include <omtalk/Handle.h>
#include <omtalk/MemoryManager.h>

using namespace omtalk;
using namespace omtalk::gc;

static void BM_Allocate(benchmark::State &state) {

  static auto mm =
      MemoryManagerBuilder<TestCollectorScheme>()
          .withRootWalker(std::make_unique<RootWalker<TestCollectorScheme>>())
          .build();

  Context<TestCollectorScheme> context(mm);

  for (auto _ : state) {
    HandleScope scope = context.getAuxData().rootScope.createScope();
    auto ref = allocateTestStructObject(context, 10);
    Handle<TestStructObject> handle(scope, ref);
    context.yieldForGC();
  }
}
BENCHMARK(BM_Allocate);
BENCHMARK(BM_Allocate)->Threads(2);
BENCHMARK(BM_Allocate)->Threads(4);
BENCHMARK(BM_Allocate)->Threads(8);