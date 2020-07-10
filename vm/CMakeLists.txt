add_library(libomtalk
    interpreter.cpp
    interpreter.nasm
    omtalk.cpp
)

target_link_libraries(libomtalk
  PUBLIC
)

target_include_directories(libomtalk
    PUBLIC
        include
        ${CMAKE_CURRENT_BINARY_DIR}/include
)

# add_subdirectory(test)
