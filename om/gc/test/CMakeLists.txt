add_om_executable(om-gc-test
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

target_link_libraries(om-gc-test
	PRIVATE
	Catch2::Catch2
	catch2-main
	om-gc
	om-gc-example
)

add_test(om-gc-test om-gc-test)
