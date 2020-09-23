# Disable testing in google benchmark
set(BENCHMARK_ENABLE_TESTING false)

add_library(catch2-main OBJECT
  CatchMain.cpp
)

target_link_libraries(catch2-main PUBLIC Catch2::Catch2)

add_subdirectory(benchmark)
add_subdirectory(Catch2)
