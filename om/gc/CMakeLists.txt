add_omtalk_library(om-gc
    src/MemoryManager.cpp
    src/MutatorMutex.cpp
)

target_include_directories(om-gc
    PUBLIC
        include
)

target_link_libraries(om-gc
    PUBLIC
        ab-util
        ${CMAKE_THREAD_LIBS_INIT}
)

if(OM_BENCHMARKS)
  add_subdirectory(benchmark)
endif()

add_subdirectory(example)

if(OM_TESTS)
  add_subdirectory(test)
endif()
