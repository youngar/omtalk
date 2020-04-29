add_library(omtalk-parser
    src/Dumper.cpp
    src/Parser.cpp
)

target_include_directories(omtalk-parser
    PUBLIC
        include
)
