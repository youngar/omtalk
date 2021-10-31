if(OMTALK_TESTS)
  add_library(catch2-main OBJECT
    CatchMain.cpp
  )
  add_subdirectory(Catch2)
  target_link_libraries(catch2-main PUBLIC Catch2::Catch2)
endif()

if(OMTALK_BENCHMARKS)
  # Disable testing in google benchmark
  set(BENCHMARK_ENABLE_TESTING false)
  add_subdirectory(benchmark)
endif()
