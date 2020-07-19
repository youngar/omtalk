
add_library(omtalk-objectmodel
    src/ObjectModel.cpp
)

target_include_directories(omtalk-objectmodel
    PUBLIC
        include
)

target_link_libraries(omtalk-objectmodel
PUBLIC
    omtalk-gc
    omtalk-util
)

if(OMTALK_WARNINGS)
target_compile_options(omtalk-objectmodel
    PRIVATE
        -Werror
        -Wall
        # -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
)
endif()

add_executable(omtalk-objectmodel-test
    test/main.cpp
)

target_link_libraries(omtalk-objectmodel-test
PRIVATE
    omtalk-gc
    Catch2::Catch2
)

if(OMTALK_WARNINGS)
target_compile_options(omtalk-objectmodel-test
    PRIVATE
        -Werror
        -Wall
        # -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
)
endif()

add_test(omtalk-objectmodel-test omtalk-objectmodel-test)