
add_omtalk_library(omtalk-util
    src/Empty.cpp
)

target_include_directories(omtalk-util
    PUBLIC
        include
)

add_omtalk_executable(omtalk-util-test
    test/test-atomic.cpp
    test/test-bitarray.cpp
    test/test-bits.cpp
    test/test-eytzinger.cpp
    test/test-intrusivelist.cpp
    test/test-math.cpp
)

target_link_libraries(omtalk-util-test
    PRIVATE
        omtalk-util
        Catch2::Catch2
        catch2-main
)

add_test(omtalk-util-test omtalk-util-test)
