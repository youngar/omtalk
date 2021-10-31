add_library(omtalk-objectmodel
  src/ObjectModel.cpp
)

target_include_directories(omtalk-objectmodel
  PUBLIC
    include
)

target_link_libraries(omtalk-objectmodel
  PUBLIC
    omtalk-gc
    omtalk-util
)

if(OMTALK_WARNINGS)
  target_compile_options(omtalk-objectmodel
    PRIVATE
      -Werror
      -Wall
      # -Wextra
      -Wno-unused-parameter
      -Wno-unused-function
  )
endif()

if (OMTALK_TEST)
  add_executable(omtalk-objectmodel-test
    test/test-objectmodel.cpp
    test/test-structbuilder.cpp
    test/test-type.cpp
  )

  target_link_libraries(omtalk-objectmodel-test
    PRIVATE
      omtalk-gc
      omtalk-objectmodel
      Catch2::Catch2
      catch2-main
  )

  if(OMTALK_WARNINGS)
    target_compile_options(omtalk-objectmodel-test
      PRIVATE
        -Werror
        -Wall
        # -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
    )
  endif()

  add_test(omtalk-objectmodel-test omtalk-objectmodel-test)
endif()
