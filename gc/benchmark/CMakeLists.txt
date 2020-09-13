add_executable(omtalk-gc-bench
    bench-markmap.cpp
    main.cpp
)

target_link_libraries(omtalk-gc-bench
    benchmark::benchmark
    omtalk-gc
)