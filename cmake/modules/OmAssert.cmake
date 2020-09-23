if(OM_ASSERT)
    return()
endif()
set(OM_ASSERT 1)

include(OmUtil)

function(om_assert)
    cmake_parse_arguments(ARG "" "MSG" "" ${ARGN})
    set(condition ${ARG_UNPARSED_ARGUMENTS})
    set_default(ARG_MSG "${condition} adfasdfasdfasdfasdfasdfasdfasdfasdfasdf")
    if(${condition})
    else()
        message(FATAL_ERROR "assertion failed: " ${ARG_MSG})
    endif()
endfunction()