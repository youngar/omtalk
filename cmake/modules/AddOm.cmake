if(ADD_OMTALK_)
	return()
endif()
set(ADD_OMTALK_ TRUE)

# Add omtalk specific build options to a target.
function(add_om_target target)
	target_compile_definitions(${target}
		PUBLIC
			${OMTALK_COMPILE_DEFINITIONS}
	)

	target_compile_options(${target}
		PUBLIC
			${OMTALK_COMPILE_OPTIONS}
	)

	target_link_options(${target}
		PUBLIC
			${OMTALK_LINK_OPTIONS}
	)

	if(OMTALK_CCACHE)
		set_target_properties(${target}
			PROPERTIES
				C_COMPILER_LAUNCHER ${OMTALK_TOOL_CCACHE}
				CXX_COMPILER_LAUNCHER ${OMTALK_TOOL_CCACHE}
		)
	endif()

	if(OMTALK_STATIC_CLANG_TIDY)
		set_target_properties(${target}
			PROPERTIES
				C_CLANG_TIDY ${OMTALK_TOOL_CLANG_TIDY}
				CXX_CLANG_TIDY ${OMTALK_TOOL_CLANG_TIDY}
		)
	endif()

	if(OMTALK_STATIC_IWYU)
		set_target_properties(${target}
			PROPERTIES
				C_INCLUDE_WHAT_YOU_USE ${OMTALK_IWYU_EXECUTABLE}
				CXX_INCLUDE_WHAT_YOU_USE ${OMTALK_IWYU_EXECUTABLE}
		)
	endif()

	if(OMTALK_STATIC_LWYU)
		set_target_properties(${target}
			PROPERTIES
				LINK_WHAT_YOU_USE true
		)
	endif()
endfunction()

# Add an omtalk executable.  Will add build options depending on how the project
# was configured.
function(add_om_executable target)
	add_executable(${ARGV})
	add_om_target(${ARGV})
endfunction()

# Add an omtalk library.  Will add build options depending on how the project
# was configured.
function(add_om_library name)
	add_library(${ARGV})
	add_om_target(${ARGV})
endfunction()
