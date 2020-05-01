
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

add_executable(omtalk-gc-test
	test/main
)

target_link_libraries(omtalk-gc-test
	PRIVATE
		omtalk-gc
)

# add_test()