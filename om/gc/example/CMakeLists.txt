
add_library(om-gc-example INTERFACE)

target_include_directories(om-gc-example
    INTERFACE
        include
)

target_link_libraries(om-gc-example
    INTERFACE
        om-gc
)