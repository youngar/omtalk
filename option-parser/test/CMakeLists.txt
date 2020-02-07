add_executable(op-test
    test_everything.cpp
)

target_link_libraries(op-test
    op-runtime-lib
    gtest_main
)

add_test(op-test op-test)
