
add_library(omtalk-gc
	include/omtalk/GC.h
	include/omtalk/Heap.h
	include/omtalk/Ref.h
	src/GC.cpp
)

target_include_directories(omtalk-gc
	PUBLIC
		include
)

if(OMTALK_WARNINGS)
	target_compile_options(omtalk-gc
		PRIVATE
			-Werror -Wall -Wextra -Wno-unused-parameter
	)
endif()

add_executable(omtalk-gc-test
	test/main
)

target_link_libraries(omtalk-gc-test
	PRIVATE
		omtalk-gc
)

if(OMTALK_WARNINGS)
	target_compile_options(omtalk-gc-test
		PRIVATE
			-Werror -Wall -Wextra -Wno-unused-parameter
	)
endif()

# add_test()