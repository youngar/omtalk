add_library(ab-util INTERFACE)

target_include_directories(ab-util
  INTERFACE
    include
)

if(OM_TESTS)
  add_om_executable(ab-util-test
    test/test-atomic.cpp
    test/test-bitarray.cpp
    test/test-bits.cpp
    test/test-eytzinger.cpp
    test/test-intrusivelist.cpp
    test/test-math.cpp
  )

  target_link_libraries(ab-util-test
    PRIVATE
      ab-util
      Catch2::Catch2
      catch2-main
  )

  add_test(ab-util-test ab-util-test)
endif()
