
# Add omtalk specific build options to a target.
function(add_omtalk_target target)
  target_compile_options(${target} PUBLIC ${OMTALK_COMPILE_OPTIONS})
  target_link_options(${target} PUBLIC ${OMTALK_LINK_OPTIONS})

  if(OMTALK_STATIC_CLANG_TIDY)
    set_target_properties(
      ${target}
      PROPERTIES
        C_CLANG_TIDY ${OMTALK_TOOL_CLANG_TIDY}
        CXX_CLANG_TIDY ${OMTALK_TOOL_CLANG_TIDY}
    )
  endif()

  if(OMTALK_STATIC_IWYU)
    set_target_properties(
      ${target}
      PROPERTIES
        C_INCLUDE_WHAT_YOU_USE ${OMTALK_TOOL_IWYU}
        CXX_INCLUDE_WHAT_YOU_USE ${OMTALK_TOOL_IWYU} 
    )
  endif()

  if(OMTALK_STATIC_LWYU)
    set_target_properties(
      ${target}
      PROPERTIES
        LINK_WHAT_YOU_USE true
    )
  endif()
endfunction()

# Add an omtalk executable.  Will add build options depending on how the project
# was configured.
function(add_omtalk_executable target)
  add_executable(${ARGV})
  add_omtalk_target(${ARGV})
endfunction()

# Add an omtalk library.  Will add build options depending on how the project
# was configured.
function(add_omtalk_library name)
  add_library(${ARGV})
  add_omtalk_target(${ARGV})
endfunction()

# Use omtalk tablegen to generate the code for a type universe.
function(omtalk_tablegen ofn)
  tablegen(OMTALK ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

# Add an omtalk universe target and generate all code necessary to use instances
# of these types from C++.
function(add_omtalk_universe file universe)
  set(LLVM_TARGET_DEFINITIONS ${file}.td)
  omtalk_tablegen(${file}.h.inc -gen-type-defs -universe=${universe})
  add_public_tablegen_target(Omtalk${universe}IncGen)
endfunction()
