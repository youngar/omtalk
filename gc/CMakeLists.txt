
add_omtalk_library(omtalk-gc
    src/MemoryManager.cpp
    src/MutatorMutex.cpp
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

if(OMTALK_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

add_subdirectory(example)

if(OMTALK_TESTS)
  add_subdirectory(test)
endif()
