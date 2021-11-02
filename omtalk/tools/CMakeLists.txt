if(OMTALK_COMPONENT_CORE)
    add_subdirectory(omtalk)
endif()

if(OMTALK_COMPONENT_COMPILER)
    add_subdirectory(omtalk-opt)
endif()

if(OMTALK_COMPONENT_PARSER)
    add_subdirectory(omtalk-parser)
endif()
