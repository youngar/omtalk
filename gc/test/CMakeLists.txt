add_omtalk_executable(omtalk-gc-test
	test-compact.cpp
	test-dispatcher.cpp
	test-exclusive.cpp
	test-forwarding.cpp
	test-gc.cpp
	test-handle.cpp
	test-mark.cpp
	test-mutatormutex.cpp
	test-regions.cpp
	test-startup.cpp
)

target_link_libraries(omtalk-gc-test
	PRIVATE
	Catch2::Catch2
	catch2-main
	omtalk-gc
	omtalk-gc-example
)

add_test(omtalk-gc-test omtalk-gc-test)
