add_library(omtalk
    include/omtalk/IRGen/IRGen.h
    src/IRGen/IRGen.cpp
)

target_include_directories(omtalk
    PUBLIC
        include
)

target_link_libraries(omtalk
    PUBLIC
        MLIROmtalk
        omtalk-parser
)