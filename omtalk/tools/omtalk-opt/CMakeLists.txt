add_executable(omtalk-opt omtalk-opt.cpp)
target_link_libraries(omtalk-opt
	PRIVATE
	MLIRMlirOptMain
	om-ir
	omtalk-ir
)
