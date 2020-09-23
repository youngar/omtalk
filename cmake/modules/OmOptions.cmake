if(OM_OPTIONS)
	return()
endif()
set(OM_OPTIONS TRUE)

include(OmAssert)
include(OmUtil)

function(om_add_option_group)
	foreach(group IN LISTS ARGN)
		define_property(GLOBAL PROPERTY ${group}
			BRIEF_DOCS "An option group."
			FULL_DOCS "A list of options which may need to be validated."
		)
		set_property(GLOBAL PROPERTY ${group} "")
	endforeach(group)
endfunction(om_add_option_group)

function(om_group_options group)
	set_property(GLOBAL APPEND PROPERTY ${group} ${ARGN})
endfunction()

function(_om_get_option_group output group)
	get_property(${output} GLOBAL PROPERTY ${group})
	set(${output} ${${output}} PARENT_SCOPE)
endfunction()

function(om_declare_option opt)
	define_property(GLOBAL PROPERTY ${opt}_REQUIRES
		BRIEF_DOCS "flags required by ${opt}"
		FULL_DOCS "A list of variables that must not be false/off if ${opt} is true."
	)
	define_property(GLOBAL PROPERTY ${opt}_CONFLICTS
		BRIEF_DOCS "flags conflicting with ${opt}"
		FULL_DOCS "A list of variables that must be false/off if ${opt} is true."
	)
endfunction()

function(om_option_requires opt)
	set_property(GLOBAL APPEND PROPERTY ${opt}_REQUIRES ${ARGN})
endfunction()

function(om_option_conflicts opt)
	set_property(GLOBAL APPEND PROPERTY ${opt}_CONFLICTS ${ARGN})
endfunction()

function(om_add_option opt group)
	cmake_parse_arguments(ARG "" "DOC" "DEFAULT;REQUIRES;CONFLICTS" ${ARGN})

	if (DEFINED ARG_COMMENT)
		set(comment ${ARG_COMMENT})
	else()
		set(comment "")
	endif()

	if (NOT DEFINED ARG_DEFAULT)
		set(default OFF)
	elseif (${ARG_DEFAULT})
		set(default ON)
	else()
		set(default OFF)
	endif()

	om_declare_option(${opt} ${group})
	om_group_options(${group} ${opt})
	set(${opt} ${default} CACHE BOOL "${comment}")

	if(DEFINED ARG_REQUIRES)
		om_option_requires(${opt} ${ARG_REQUIRES})
	endif()

	if(DEFINED ARG_CONFLICTS)
		om_option_conflicts(${opt} ${ARG_CONFLICTS})
	endif()
endfunction()

function(_om_check_option_requirements opt)
	get_property(requirements GLOBAL PROPERTY ${option}_REQUIRES)
	foreach(requirement IN LISTS requirements)
		om_assert(${requirement} MSG "option ${option} requires ${requirement}")
	endforeach()
endfunction()

function(_om_check_option_conflicts option)
	get_property(conflicts GLOBAL PROPERTY ${option}_CONFLICTS)
	foreach(conflict IN LISTS conflicts)
		om_assert(NOT ${conflict} MSG "option ${option} conflicts with ${conflict}")
	endforeach()
endfunction()

function(om_validate_option option)
	if(${option})
		_om_check_option_requirements(${option})
		_om_check_option_conflicts(${option})
	endif()
endfunction()

function(om_validate_option_group)
	foreach(group IN LISTS ARGN)
		_om_get_option_group(options ${group})
		foreach(option IN LISTS options)
			om_validate_option(${option})
		endforeach()
	endforeach()
endfunction()
