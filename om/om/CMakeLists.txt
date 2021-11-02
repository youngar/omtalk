add_om_library(om-objectmodel
  src/ObjectModel.cpp
)

target_include_directories(om-objectmodel
  PUBLIC
    include
)

target_link_libraries(om-objectmodel
  PUBLIC
    om-gc
    ab-util
)

if(OMTALK_WARNINGS)
  target_compile_options(om-objectmodel
    PRIVATE
      -Werror
      -Wall
      # -Wextra
      -Wno-unused-parameter
      -Wno-unused-function
  )
endif()

if (OMTALK_TEST)
  add_executable(om-objectmodel-test
    test/test-objectmodel.cpp
    test/test-structbuilder.cpp
    test/test-type.cpp
  )

  target_link_libraries(om-objectmodel-test
    PRIVATE
      om-gc
      om-objectmodel
      Catch2::Catch2
      catch2-main
  )

  if(OMTALK_WARNINGS)
    target_compile_options(om-objectmodel-test
      PRIVATE
        -Werror
        -Wall
        # -Wextra
        -Wno-unused-parameter
        -Wno-unused-function
    )
  endif()

  add_test(om-objectmodel-test om-objectmodel-test)
endif()
