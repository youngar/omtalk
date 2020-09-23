add_executable(omtalk-gc-bench
    bench-forwarding.cpp
    bench-markmap.cpp
    main.cpp
)

target_link_libraries(omtalk-gc-bench
    benchmark::benchmark
    omtalk-gc
    omtalk-gc-example
)