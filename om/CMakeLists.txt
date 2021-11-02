if(OM_COMP_OM_GC)
  add_subdirectory(gc)
endif()

if(ON_COMP_COMPILER)
  add_subdirectory(ir)
endif()

if(OM_LIT_TESTS)
  add_subdirectory(lit-test)
endif()

if(OM_COMP_OM_OM)
  add_subdirectory(om)
endif()
