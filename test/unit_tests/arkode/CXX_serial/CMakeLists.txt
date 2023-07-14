# ---------------------------------------------------------------
# Programmer(s): Daniel R. Reynolds @ SMU
# ---------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2023, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ---------------------------------------------------------------
# ARKode C++ serial unit_tests
# ---------------------------------------------------------------

# List of test tuples of the form "name\;args"
set(unit_tests
  "ark_test_accumerror_brusselator.cpp\;3 1"
  "ark_test_accumerror_brusselator.cpp\;-4 2"
  "ark_test_accumerror_brusselator.cpp\;5 3"
  "ark_test_accumerror_kpr.cpp\;0 3"
  "ark_test_accumerror_kpr.cpp\;1 3 -5.0"
  "ark_test_accumerror_kpr.cpp\;1 5 -5.0"
  "ark_test_analytic_sys_mri.cpp\;0"
  "ark_test_analytic_sys_mri.cpp\;1"
  "ark_test_dahlquist_mri.cpp\;"
  "ark_test_butcher.cpp\;"
  "ark_test_getjac.cpp\;"
  "ark_test_getjac_mri.cpp\;"
  "ark_test_kpr_mriadapt.cpp\;--hs 0.01 --mri_type 0 --scontrol 0 --fast_type 2 --fcontrol 1"
  "ark_test_kpr_mriadapt.cpp\;--hs 0.01 --mri_type 0 --scontrol 0 --fast_type 0"
  "ark_test_kpr_mriadapt.cpp\;--hs 0.002 --rtol 0.000004 --mri_type 5 --scontrol 0 --fast_type 0"
  "ark_test_kpr_mriadapt.cpp\;--hs 0.002 --mri_type 4 --scontrol 0 --fast_type 3"
  "ark_test_kpr_mriadapt.cpp\;--hs 0.001 --hf 1.e-5 --rtol 1.e-9 --G -100.0 --w 100.0 --e 0.5 --mri_type 10 --scontrol 0 --fast_type 2 --fcontrol 0"
)

# Add the build and install targets for each test
foreach(test_tuple ${unit_tests})

  # Parse the test tuple
  list(GET test_tuple 0 test)
  list(GET test_tuple 1 test_args)

  # Extract the file name without extension
  get_filename_component(test_target ${test} NAME_WE)

  # Check if this test has already been added, only need to add test source
  # files once for testing with different inputs
  if(NOT TARGET ${test_target})

    # Test source files
    add_executable(${test_target} ${test})

    # Folder to organize targets in an IDE
    set_target_properties(${test_target} PROPERTIES FOLDER "unit_tests")

    # Include location of public and private header files
    target_include_directories(${test_target} PRIVATE
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
      ${PROJECT_SOURCE_DIR}/test/unit_tests/utilities
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}/src)

    # Libraries to link against
    target_link_libraries(${test_target}
      sundials_arkode
      sundials_nvecserial
      ${EXE_EXTRA_LINK_LIBS})

  endif()

  # Check if test args are provided and set the test name
  if("${test_args}" STREQUAL "")
    set(test_name ${test_target})
  else()
    string(REPLACE " " "_" test_name "${test_target}_${test_args}")
  endif()

  if(SUNDIALS_PRECISION MATCHES "DOUBLE")
    set(diff_output "")
  else()
    set(diff_output "NODIFF")
  endif()

  # add test to regression tests
  sundials_add_test(${test_name} ${test_target}
    TEST_ARGS ${test_args}
    ANSWER_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    ANSWER_FILE ${test_name}.out
    EXAMPLE_TYPE "develop"
    ${diff_output}
  )

endforeach()

message(STATUS "Added ARKODE CXX serial unit tests")
