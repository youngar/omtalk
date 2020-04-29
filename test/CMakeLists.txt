configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(OMTALK_OPT_TEST_DEPENDS
    FileCheck count not
    omtalk-opt
)

add_lit_testsuite(check-omtalk-opt "Running the omtalk-opt regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${OMTALK_OPT_TEST_DEPENDS}
)

set_target_properties(check-omtalk-opt PROPERTIES FOLDER "Tests")

add_lit_testsuites(OMTALK_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${OMTALK_OPT_TEST_DEPENDS})
