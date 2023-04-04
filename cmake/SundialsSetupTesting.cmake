# ---------------------------------------------------------------
# Programmer(s): David J. Gardner @ LLNL
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
# Enable SUNDIALS Testing
# ---------------------------------------------------------------

# Enable testing with 'make test'
include(CTest)

# Check if test runner is needed
if(SUNDIALS_TEST_DIFF OR SUNDIALS_TEST_PROFILE)
  set(SUNDIALS_TEST_USE_RUNNER TRUE)
else()
  set(SUNDIALS_TEST_USE_RUNNER FALSE)
endif()

if(SUNDIALS_TEST_USE_RUNNER)

  # Python is needed to use the test runner
  find_package(Python3 COMPONENTS Interpreter)
  if(NOT Python3_FOUND)
    print_error("Python3 is required to diff or profile tests. Set SUNDIALS_TEST_DIFF and SUNDIALS_TEST_PROFILE to OFF")
  endif()

  # Look for the testRunner script in the test directory
  find_program(TESTRUNNER testRunner PATHS test NO_DEFAULT_PATH)
  if(NOT TESTRUNNER)
    print_error("Could not locate testRunner. Set SUNDIALS_TEST_DIFF and SUNDIALS_TEST_PROFILE to OFF")
  endif()
  message(STATUS "Found testRunner: ${TESTRUNNER}")
  set(TESTRUNNER ${TESTRUNNER} CACHE INTERNAL "")

endif()

if(SUNDIALS_TEST_DIFF)

  # Create the default test output directory if necessary
  if(NOT EXISTS ${SUNDIALS_TEST_OUTPUT_DIR})
    file(MAKE_DIRECTORY ${SUNDIALS_TEST_OUTPUT_DIR})
  endif()
  message(STATUS "Test output directory: ${SUNDIALS_TEST_OUTPUT_DIR}")

  # If a non-default answer directory was provided make sure it exists
  if(SUNDIALS_TEST_ANSWER_DIR)
    message(STATUS "Test answer directory: ${SUNDIALS_TEST_ANSWER_DIR}")
    if(NOT EXISTS ${SUNDIALS_TEST_ANSWER_DIR})
      print_error("SUNDIALS_TEST_ANSWER_DIR does not exist!")
    endif()
  endif()

  message(STATUS "Test float comparison precision: ${SUNDIALS_TEST_FLOAT_PRECISION}")
  message(STATUS "Test integer comparison precision: ${SUNDIALS_TEST_INTEGER_PRECISION}")

endif()

# If examples are installed, create post install smoke test targets
if(EXAMPLES_INSTALL)

  # Directories for installation testing
  set(TEST_INSTALL_DIR ${PROJECT_BINARY_DIR}/Testing_Install)
  set(TEST_INSTALL_ALL_DIR ${PROJECT_BINARY_DIR}/Testing_Install_All)

  # Create installation testing directories
  if(NOT EXISTS ${TEST_INSTALL_DIR})
    file(MAKE_DIRECTORY ${TEST_INSTALL_DIR})
  endif()

  if(NOT EXISTS ${TEST_INSTALL_ALL_DIR})
    file(MAKE_DIRECTORY ${TEST_INSTALL_ALL_DIR})
  endif()

  # Create test_install and test_install_all targets
  add_custom_target(test_install
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan
    "All installation tests complete.")

  add_custom_target(test_install_all
    ${CMAKE_COMMAND} -E cmake_echo_color --cyan
    "All installation tests complete.")

endif()
