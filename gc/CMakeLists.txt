
add_omtalk_library(omtalk-gc
    src/MemoryManager.cpp
)

target_include_directories(omtalk-gc
    PUBLIC
        include
)

target_link_libraries(omtalk-gc
    PUBLIC
        omtalk-util
        ${CMAKE_THREAD_LIBS_INIT}
)


add_omtalk_executable(omtalk-gc-test
    test/main.cpp
    test/test-gc.cpp
    test/test-handle.cpp
    test/test-regions.cpp
)

target_link_libraries(omtalk-gc-test
    PRIVATE
        omtalk-gc
        Catch2::Catch2
)

add_test(omtalk-gc-test omtalk-gc-test)
