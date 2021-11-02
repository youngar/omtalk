enable_language(ASM_NASM)

add_om_library(omtalk-core
    include/omtalk/IRGen/IRGen.h
    src/IRGen/IRGen.cpp
    src/Runtime.cpp
)

target_include_directories(omtalk-core
    PUBLIC
        include
        ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(omtalk-core
    PUBLIC
        om-gc
        omtalk-ir
        omtalk-parser
)
