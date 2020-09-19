add_omtalk_executable(omtalk-gc-test
    main.cpp
    test-dispatcher.cpp
    test-exclusive.cpp
    test-forwarding.cpp
    test-gc.cpp
    test-handle.cpp
    test-regions.cpp
    test-startup.cpp
)

target_link_libraries(omtalk-gc-test
    PRIVATE
        omtalk-gc
        Catch2::Catch2
)

add_test(omtalk-gc-test omtalk-gc-test)