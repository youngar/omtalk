include(OmAssert)

om_assert(
    OMTALK_COMP_OM AND
    OMTALK_COMP_COMPILER
)

add_executable(om-gen
    src/om-gen-main.cpp
)

target_link_libraries(om-gen
    PRIVATE
)
