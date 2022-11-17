# ------------------------------------------------------------------------------
# Programmer(s): Steven Smith and David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2022, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ------------------------------------------------------------------------------
#
# SUNDIALS_ADD_TEST(<test name> <executable>)
#
# CMake macro to add a SUNDIALS regression test. Keyword input arguments can be
# added after <executable> to set regression test options (see oneValueArgs and
# multiValueArgs below).
#
# When SUNDIALS_TEST_DEVTESTS is OFF (default) the executable is run and success
# or failure is determined by the executable return value (zero or non-zero
# respectively).
#
# When SUNDIALS_TEST_DEVTESTS is ON the executable is run and its output is
# compared with the corresponding .out file. If the output differs significantly
# then the test fails. The default level of significance is 4 decimal places for
# floating point values and 10% for integer values.
#
# The level of precision can be adjusted for an individual test with the
# FLOAT_PRECISION AND INTEGER_PRECISION keyword inputs to the macro or globally
# for all tests with the cache variables SUNDIALS_TEST_FLOAT_PRECISION and
# SUNDIALS_TEST_INTEGER_PRECISION.
#
#  -D SUNDIALS_TEST_FLOAT_PRECISION=<number of digits>
#  -D SUNDIALS_TEST_INTEGER_PRECISION=<% difference>
#
# By default testing output is written to builddir/Testing/output and the .out
# answer file directory is set using the ANSWER_DIR keyword input to
# sourcedir/examples/package/testdir. These can be changed by setting the cache
# variables SUNDIALS_TEST_OUTPUT_DIR and SUNDIALS_TEST_ANSWER_DIR.
#
#  -D SUNDIALS_TEST_OUTPUT_DIR=<path to output directory>
#  -D SUNDIALS_TEST_ANSWER_DIR=<path to answer directory>
# ------------------------------------------------------------------------------

macro(SUNDIALS_ADD_TEST NAME EXECUTABLE)

  # macro options
  # NODIFF = do not diff the test output against an answer file
  set(options "NODIFF")

  # macro keyword inputs followed by a single value
  # MPI_NPROCS        = number of mpi tasks to use in parallel tests
  # FLOAT_PRECISION   = precision for floating point failure comparision (num digits),
  #                     to use the default, either don't provide the keyword, or
  #                     provide the value "default"
  # INTEGER_PRECISION = integer percentage difference for failure comparison
  # ANSWER_DIR        = path to the directory containing the test answer file
  # ANSWER_FILE       = name of test answer file
  # EXAMPLE_TYPE      = release or develop examples
  set(oneValueArgs "MPI_NPROCS" "FLOAT_PRECISION" "INTEGER_PRECISION"
    "ANSWER_DIR" "ANSWER_FILE" "EXAMPLE_TYPE")

  # macro keyword inputs followed by multiple values
  # TEST_ARGS = command line arguments to pass to the test executable
  # EXTRA_ARGS = additional command line arguments not added to the test name
  set(multiValueArgs "TEST_ARGS" "EXTRA_ARGS")

  # parse inputs and create variables SUNDIALS_ADD_TEST_<keyword>
  cmake_parse_arguments(SUNDIALS_ADD_TEST
    "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # ---------------------------------
  # check if the test should be added
  # ---------------------------------

  set(_add_test TRUE)

  # always excluded
  if("${SUNDIALS_ADD_TEST_EXAMPLE_TYPE}" STREQUAL "exclude")
    set(_add_test FALSE)
  endif()

  # precision-specific exclusions
  string(TOLOWER "exclude-${SUNDIALS_PRECISION}" _exclude_precision)
  if("${SUNDIALS_ADD_TEST_EXAMPLE_TYPE}" STREQUAL _exclude_precision)
    message(STATUS "Skipped test ${NAME} because it had type ${SUNDIALS_ADD_TEST_EXAMPLE_TYPE}")
    set(_add_test FALSE)
  endif()

  # development only tests
  if(("${SUNDIALS_ADD_TEST_EXAMPLE_TYPE}" STREQUAL "develop")
      AND SUNDIALS_TEST_DEVTESTS)
    set(_add_test FALSE)
  endif()

  # --------
  # add test
  # --------

  if(_add_test)

    # set run command if necessary and remove trailing white space from the
    # command (i.e., empty MPIEXEC_PREFLAGS) as it can cause erroneous test
    # failures with some MPI implementations
    if((SUNDIALS_ADD_TEST_MPI_NPROCS) AND (MPIEXEC_EXECUTABLE))
      set(_run_command "${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${SUNDIALS_ADD_TEST_MPI_NPROCS} ${MPIEXEC_PREFLAGS}")
      string(STRIP "${_run_command}" _run_command)
    endif()

    # set the test input args
    if(SUNDIALS_ADD_TEST_TEST_ARGS)
      string(REPLACE ";" " " _user_args "${SUNDIALS_ADD_TEST_TEST_ARGS}")
      set(_run_args "${_user_args}")
      unset(_user_args)
    endif()

    if(SUNDIALS_ADD_TEST_EXTRA_ARGS)
      string(REPLACE ";" " " _extra_args "${SUNDIALS_ADD_TEST_EXTRA_ARGS}")
      set(_run_args "${_run_args} ${_extra_args}")
      unset(_extra_args)
    endif()

    if(_run_args)
      string(STRIP "${_run_args}" _run_args)
    endif()

    if(SUNDIALS_TEST_USE_RUNNER)

      # command line arguments for the test runner script
      set(TEST_ARGS
        "--verbose"
        "--testname=${NAME}"
        "--executablename=$<TARGET_FILE:${EXECUTABLE}>"
        "--outputdir=${SUNDIALS_TEST_OUTPUT_DIR}"
        )

      # check if this test is run with MPI and set the MPI run command
      if(_run_command)
        list(APPEND TEST_ARGS "--runcommand=\"${_run_command}\"")
      endif()

      if(_run_args)
        list(APPEND TEST_ARGS "--runargs=\"${_run_args}\"")
      endif()

      if(SUNDIALS_TEST_PROFILE)
        list(APPEND TEST_ARGS "--profile")
      endif()

      # set answer directory
      if(SUNDIALS_TEST_ANSWER_DIR)
        list(APPEND TEST_ARGS "--answerdir=${SUNDIALS_TEST_ANSWER_DIR}")
      elseif(SUNDIALS_ADD_TEST_ANSWER_DIR)
        list(APPEND TEST_ARGS "--answerdir=${SUNDIALS_ADD_TEST_ANSWER_DIR}")
      endif()

      # set the test answer file name
      if(SUNDIALS_ADD_TEST_ANSWER_FILE)
        list(APPEND TEST_ARGS "--answerfile=${SUNDIALS_ADD_TEST_ANSWER_FILE}")
      endif()

      # set comparison precisions or do not diff the output and answer files
      if(SUNDIALS_TEST_DIFF AND NOT SUNDIALS_ADD_TEST_NODIFF)

        if(SUNDIALS_ADD_TEST_FLOAT_PRECISION AND
            (NOT SUNDIALS_ADD_TEST_FLOAT_PRECISION MATCHES "DEFAULT|default"))
          list(APPEND TEST_ARGS "--floatprecision=${SUNDIALS_ADD_TEST_FLOAT_PRECISION}")
        else()
          list(APPEND TEST_ARGS "--floatprecision=${SUNDIALS_TEST_FLOAT_PRECISION}")
        endif()

        if(SUNDIALS_ADD_TEST_INTEGER_PRECISION AND
            (NOT SUNDIALS_ADD_TEST_INTEGER_PRECISION MATCHES "DEFAULT|default"))
          list(APPEND TEST_ARGS "--integerpercentage=${SUNDIALS_ADD_TEST_INTEGER_PRECISION}")
        else()
          list(APPEND TEST_ARGS "--integerpercentage=${SUNDIALS_TEST_INTEGER_PRECISION}")
        endif()

      else()

        list(APPEND TEST_ARGS "--nodiff")

      endif()

      add_test(NAME ${NAME} COMMAND ${Python3_EXECUTABLE} ${TESTRUNNER} ${TEST_ARGS})

    else()

      if(_run_command)
        add_test(NAME ${NAME} COMMAND ${_run_command} $<TARGET_FILE:${EXECUTABLE}> ${_run_args})
      else()
        add_test(NAME ${NAME} COMMAND $<TARGET_FILE:${EXECUTABLE}> ${_run_args})
      endif()

    endif()

  endif()

  unset(_add_test)
  unset(_use_runner)
  unset(_run_command)
  unset(_run_args)

endmacro()
