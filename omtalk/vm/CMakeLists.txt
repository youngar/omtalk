enable_language(ASM_NASM)

add_omtalk_library(omtalk-vm
  interpreter.cpp
  interpreter.nasm
  omtalk.cpp
)

target_link_libraries(omtalk-vm
  PUBLIC
)

target_include_directories(omtalk-vm
  PUBLIC
  include
  ${CMAKE_CURRENT_BINARY_DIR}/include
)

# add_subdirectory(test)
