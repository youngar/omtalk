# LLVM Project

function(build_llvm location)
    cmake_parse_arguments(ARG "" "" "OPTIONS" ${ARGN})

    set(OMTALK_LLVM_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${location}")
    set(OMTALK_LLVM_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/${location}")

    list(APPEND CMAKE_MODULE_PATH "${OMTALK_LLVM_BINARY_DIR}/lib/cmake/llvm")
    list(APPEND CMAKE_MODULE_PATH "${OMTALK_LLVM_BINARY_DIR}/lib/cmake/mlir")

    message("Building LLVM with ${ARG_OPTIONS}")
    message("Setting CMAKE_MODULE_PATH to ${CMAKE_MODULE_PATH}")

    set(OMTALK_LLVM_SOURCE_DIR "${OMTALK_LLVM_SOURCE_DIR}" PARENT_SCOPE)
    set(OMTALK_LLVM_BINARY_DIR "${OMTALK_LLVM_BINARY_DIR}" PARENT_SCOPE)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)

    file(MAKE_DIRECTORY "${OMTALK_LLVM_BINARY_DIR}")

    execute_process(
        COMMAND "${CMAKE_COMMAND}" "${OMTALK_LLVM_SOURCE_DIR}/llvm" -G "${CMAKE_GENERATOR}" ${ARG_OPTIONS}
        WORKING_DIRECTORY "${OMTALK_LLVM_BINARY_DIR}")

    add_custom_target(omtalk_llvm ALL
        COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${OMTALK_LLVM_BINARY_DIR}"
        COMMENT "Building LLVM"
        USES_TERMINAL)
endfunction()
