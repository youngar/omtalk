if(OM_UTIL_)
    return()
endif()

set(OM_UTIL_ TRUE)

function(coerce_flag output)
	if(${ARGN})
		set(${output} ON PARENT_SCOPE)
	else()
		set(${output} OFF PARENT_SCOPE)
	endif()
endfunction()

macro(set_default output)
    if(NOT DEFINED ${output})
        set(${output} ${ARGN})
    endif()
endmacro()
