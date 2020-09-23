# Disable testing in google benchmark
set(BENCHMARK_ENABLE_TESTING false)

add_subdirectory(benchmark)
add_subdirectory(Catch2)

add_library(catch2_main OBJECT
  Catch2/examples/000-CatchMain.cpp
)

target_link_libraries(catch2_main PUBLIC Catch2::Catch2)
