if(OM_COMPONENT_GC)
  add_subdirectory(gc)
endif()

if(OM_COMPONENT_COMPILER)
  add_subdirectory(ir)
endif()

if(OM_LIT_TESTS)
  add_subdirectory(lit-test)
endif()

if(OM_COMPONENT_OM)
  add_subdirectory(om)
endif()
