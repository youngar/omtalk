add_library(omtalk-parser
    src/Parser.cpp
)

target_include_directories(omtalk-parser
    PUBLIC
        include
)

if(OMTALK_WARNINGS)
    target_compile_options(omtalk-parser
        PRIVATE
            -Werror
            -Wall
            # -Wextra
            -Wno-unused-parameter
            -Wno-unused-function
    )
endif()
