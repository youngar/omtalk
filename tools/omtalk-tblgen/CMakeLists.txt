set(LLVM_LINK_COMPONENTS
  Support
)

add_tablegen(omtalk-tblgen OMTALK
  omtalk-tblgen.cpp
)

set_target_properties(omtalk-tblgen PROPERTIES FOLDER "Omtalk tablegenning")
