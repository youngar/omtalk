if(OM_UTIL_)
	return()
endif()
set(OM_UTIL_ TRUE)

macro(set_default output)
	if(NOT DEFINED ${output})
		set(${output} ${ARGN})
	endif()
endmacro()
