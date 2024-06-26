include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${MS_CCSRC_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/stub/runtime/)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CUDA_INCLUDE_DIRS})

include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)

if(ENABLE_ACL)
    add_definitions(-D ENABLE_ACL)
endif()
file(GLOB_RECURSE CXX_ST_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cc)
add_executable(cxx_st_tests ${CXX_ST_SRC})
target_link_libraries(cxx_st_tests PRIVATE mindspore_shared_lib _c_dataengine mindspore::gtest)
target_link_options(cxx_st_tests PRIVATE -Wl,--unresolved-symbols=ignore-all)
set_target_properties(cxx_st_tests PROPERTIES SKIP_BUILD_RPATH TRUE RUNTIME_OUTPUT_DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR})
file(COPY ${gtest_LIBPATH}/libgtest${CMAKE_SHARED_LIBRARY_SUFFIX}
        DESTINATION ${CMAKE_CURRENT_SOURCE_DIR} FOLLOW_SYMLINK_CHAIN)