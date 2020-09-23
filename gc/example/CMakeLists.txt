
add_library(omtalk-gc-example INTERFACE)

target_include_directories(omtalk-gc-example
    INTERFACE
        include
)

target_link_libraries(omtalk-gc-example
    INTERFACE
        omtalk-gc
)