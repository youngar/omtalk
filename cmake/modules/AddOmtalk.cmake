
function(omtalk_tablegen ofn)
  tablegen(OMTALK ${ARGV})
  set(TABLEGEN_OUTPUT ${TABLEGEN_OUTPUT} ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
      PARENT_SCOPE)
endfunction()

function(add_omtalk_universe file universe)
  set(LLVM_TARGET_DEFINITIONS ${file}.td)
  omtalk_tablegen(${file}.h.inc -gen-type-defs -universe=${universe})
  add_public_tablegen_target(Omtalk${universe}IncGen)
endfunction()
