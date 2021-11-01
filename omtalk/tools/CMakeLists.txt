if(OMTALK_COMP_CORE)
    add_subdirectory(omtalk)
endif()

if(OMTALK_COMP_COMPILER)
    add_subdirectory(omtalk-opt)
endif()

if(OMTALK_COMP_PARSER)
    add_subdirectory(omtalk-parser)
endif()
