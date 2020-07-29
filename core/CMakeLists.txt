add_library(omtalk-core
    include/omtalk/IRGen/IRGen.h
    src/IRGen/IRGen.cpp
    src/Runtime.cpp
)

target_include_directories(omtalk-core
    PUBLIC
        include
)

target_link_libraries(omtalk-core
    PUBLIC
        MLIROmtalk
        omtalk-gc
        omtalk-parser
)

add_dependencies(omtalk-core OmtalkObjectsIncGen)

add_subdirectory(include)