
# # include(AddLLVM)
# # LLVM_PROJECT_SOURCE_DIRECTORY

# set(GTEST_INCULDE_DIRS
# 	${LLVM_PROJECT_SOURCE_DIRECTORY}/llvm/utils/unittest/googletest/include)

# if (TARGET gtest)
# 	set_target_properties(gtest
# 		PROPERTIES
# 			INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCULDE_DIRS}"
# 	)
# endif()


# # add_custom_target(omtalk-unittests)
# # add_unittest(omtalk-unittests omtalk-test
# # 	test_parser.cpp
# # )

# add_executable(omtalk-test
# 	main.cpp
# 	test.cpp
# )

# target_link_libraries(omtalk-test
# 	libomtalk-ir
# 	gtest
# )

# add_test(omtalk-test omtalk-test)
 