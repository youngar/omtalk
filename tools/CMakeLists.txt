add_subdirectory(omtalk)
add_subdirectory(omtalk-opt)
add_subdirectory(omtalk-parser)

# omtalk-tblgen must be added from the direcory
# above.  This is because of the way tablegen cmake
# helpers modify the parent scope.
# add_subdirectory(omtalk-tblgen)