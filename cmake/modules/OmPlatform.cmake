if(OM_PLATFORM_)
	return()
endif()
set(OM_PLATFORM_ TRUE)

include (CheckCCompilerFlag)

# Note we are assuming that if the c compiler accepts the flag, the cxx compiler will as well

set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=thread)
check_c_compiler_flag(-fsanitize=thread OM_HAVE_TSAN)

set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=address)
check_c_compiler_flag(-fsanitize=address OM_HAVE_ASAN)

set(CMAKE_REQUIRED_LINK_OPTIONS -fsanitize=undefined)
check_c_compiler_flag(-fsanitize=undefined OM_HAVE_UBSAN)
