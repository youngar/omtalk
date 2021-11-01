if(OM_COMP_OMTALK_CORE)
    add_subdirectory(omtalk)
endif()

if(OM_COMP_OMTALK_COMPILER)
    add_subdirectory(omtalk-opt)
endif()

if(OM_COMP_OMTALK_PARSER)
    add_subdirectory(omtalk-parser)
endif()
