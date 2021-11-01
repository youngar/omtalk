configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(OMTALK_TEST_DEPENDS
    FileCheck count not
    omtalk-opt
    omtalk-parser
)

add_lit_testsuite(check-omtalk
    "Running the omtalk regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${OMTALK_TEST_DEPENDS}
)

set_target_properties(check-omtalk
    PROPERTIES
        FOLDER "Tests"
)

add_lit_testsuites(OMTALK ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${OMTALK_OPT_TEST_DEPENDS}
)
