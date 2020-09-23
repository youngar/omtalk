include (CheckCCompilerFlag)

# Note we are assuming that if the c compiler accepts the flag, the cxx compiler will as well

check_c_compiler_flag(-fsanitize=thread OM_HAVE_TSAN)
check_c_compiler_flag(-fsanitize=address OM_HAVE_ASAN)
check_c_compiler_flag(-fsanitize=undefined OM_HAVE_UBSAN)
