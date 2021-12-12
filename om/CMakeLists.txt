include(OmAssert)

om_assert(OMTALK_COMP_OM)

add_library(om)

target_link_libraries(om
    PUBLIC
        omtalk-gc
        omtalk-util
        ${CMAKE_THREAD_LIBS_INIT}
)

target_include_directories(om
    PUBLIC
        include/
)
