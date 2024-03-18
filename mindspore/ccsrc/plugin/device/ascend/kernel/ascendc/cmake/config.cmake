set(CMAKE_CXX_FLAGS_DEBUG "")
set(CMAKE_CXX_FLAGS_RELEASE "")
set(ENABLE_SOURCE_PACKAGE True)
set(ENABLE_BINARY_PACKAGE True)
set(ASCEND_COMPUTE_UNIT "ascend910b;")
set(vendor_name custom_ascendc_ops)
set(ASCEND_PYTHON_EXECUTABLE python3)
set(PKG_PATH lib/plugin/ascend)
set(ENABLE_CROSS_COMPILE False)

if(DEFINED ENV{ASCEND_CUSTOM_PATH})
  set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CUSTOM_PATH}/latest)
else()
  set(ASCEND_CANN_PACKAGE_PATH /usr/local/Ascend/latest)
endif()

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH
      /usr/local/Ascend/latest
      CACHE PATH "")
endif()
if(NOT DEFINED ASCEND_PYTHON_EXECUTABLE)
  set(ASCEND_PYTHON_EXECUTABLE
      python3
      CACHE STRING "")
endif()
if(NOT DEFINED ASCEND_COMPUTE_UNIT)
  message(FATAL_ERROR "ASCEND_COMPUTE_UNIT not set in CMakePreset.json !
")
endif()
set(ASCEND_TENSOR_COMPILER_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
set(ASCEND_CCEC_COMPILER_PATH ${ASCEND_TENSOR_COMPILER_PATH}/ccec_compiler/bin)
set(ASCEND_AUTOGEN_PATH ${CMAKE_BINARY_DIR}/autogen)
file(MAKE_DIRECTORY ${ASCEND_AUTOGEN_PATH})
set(CUSTOM_COMPILE_OPTIONS "custom_compile_options.ini")
execute_process(COMMAND rm -rf ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS}
                COMMAND touch ${ASCEND_AUTOGEN_PATH}/${CUSTOM_COMPILE_OPTIONS})
