
add_library(omtalk-util
    src/Empty.cpp
)

target_include_directories(omtalk-util
    PUBLIC
        include
)

if(OMTALK_WARNINGS)
target_compile_options(omtalk-util
    PRIVATE
        -Werror
        -Wall
        # -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
)
endif()

add_executable(omtalk-util-test
    test/main.cpp
    test/test-atomic.cpp
    test/test-hashmap.cpp
)

target_link_libraries(omtalk-util-test
    PRIVATE
        omtalk-util
        Catch2::Catch2
)

if(OMTALK_WARNINGS)
target_compile_options(omtalk-util-test
    PRIVATE
        -Werror
        -Wall
        # -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
)
endif()

add_test(omtalk-util-test omtalk-util-test)
