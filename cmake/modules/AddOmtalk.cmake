
function(omtalk_tablegen ofn)
  tablegen(OMTALK ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

function(add_omtalk_universe universe universe_namespace)
  set(LLVM_TARGET_DEFINITIONS ${universe}.td)
  omtalk_tablegen(${universe}.h.inc -gen-type-defs -universe=${universe})
  add_public_tablegen_target(Omtalk${universe}IncGen)
endfunction()
