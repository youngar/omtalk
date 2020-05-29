add_library(omtalk-core
    include/omtalk/IRGen/IRGen.h
    src/IRGen/IRGen.cpp
)

target_include_directories(omtalk-core
    PUBLIC
        include
)

target_link_libraries(omtalk-core
    PUBLIC
        MLIROmtalk
        omtalk-parser
)