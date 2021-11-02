add_omtalk_library(omtalk-ir
  src/OmtalkDialect.cpp
  src/OmtalkOps.cpp
  src/OmtalkTypes.cpp
)

add_dependencies(omtalk-ir omtalk-ir-incgen)

target_include_directories(omtalk-ir
  PUBLIC
    include/
    ${CMAKE_CURRENT_BINARY_DIR}/include
)

target_link_libraries(omtalk-ir
  PUBLIC
    MLIRIR
)

add_subdirectory(include/omtalk/IR)