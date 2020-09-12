add_executable(bench-allocation
    bench-allocation.cpp
    bench-markmap.cpp
    main.cpp
)

target_link_libraries(bench-allocation
    benchmark::benchmark
    omtalk-gc
)