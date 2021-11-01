
add_executable(omtalk-test
    test_allocation.cpp
    test_allocator.cpp
    test_integer.cpp
    test_object.cpp
    test_parser.cpp
    test_stack.cpp
    test_startup.cpp
    test_symbol_table.cpp
)

target_link_libraries(omtalk-test
    gtest_main
    libomtalk
)

add_test(omtalk-test omtalk-test)
