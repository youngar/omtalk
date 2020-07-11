
add_library(omtalk-gc
    src/Allocate.cpp
    src/MemoryManager.cpp
)

target_include_directories(omtalk-gc
    PUBLIC
        include
)

target_link_libraries(omtalk-gc
    PUBLIC
        omtalk-util
)

if(OMTALK_WARNINGS)
    target_compile_options(omtalk-gc
        PRIVATE
            -Werror -Wall -Wextra -Wno-unused-parameter
    )
endif()

add_executable(omtalk-gc-test
    test/main.cpp
    test/test-gc.cpp
    test/test-handle.cpp
)

target_link_libraries(omtalk-gc-test
    PRIVATE
        omtalk-gc
        Catch2::Catch2
)

if(OMTALK_WARNINGS)
    target_compile_options(omtalk-gc-test
        PRIVATE
            -Werror -Wall -Wextra -Wno-unused-parameter
    )
endif()

add_test(omtalk-gc-test omtalk-gc-test)
