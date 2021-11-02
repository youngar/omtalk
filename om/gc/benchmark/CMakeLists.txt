add_executable(om-gc-bench
    bench-allocate.cpp
    bench-forwarding.cpp
    bench-markmap.cpp
    main.cpp
)

target_link_libraries(om-gc-bench
    benchmark::benchmark
    om-gc
    om-gc-example
)