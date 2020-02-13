#!/bin/bash
# ------------------------------------------------------------------------------
# Programmer(s): David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2020, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ------------------------------------------------------------------------------
# SUNDIALS tarball regression testing script
#
# Usage: ./suntest_tarscript.sh <package> <lib type> <real type> <index size>
#                               <TPL status> <test type> <memcheck>
#                               <build threads>
#
# Required Inputs:
#   <package>    = Which tarball to make and test:
#                    arkode   : create ARKode tarball only
#                    cvode    : create CVODE tarball only
#                    cvodes   : create CVODES tarball only
#                    ida      : create IDA tarball only
#                    idas     : create IDAS tarball only
#                    kinsol   : create KINSOL tarball only
#                    sundials : create sundials tarball containing all packages
#                    all      : all of the above options
#   <real type>  = SUNDIALS real type to build/test with:
#                    single   : single (32-bit) precision
#                    double   : double (64-bit) precision
#                    extended : extended (128-bit) precision
#                    all      : all of the above options
#   <index size> = SUNDIALS index size to build/test with:
#                    32       : 32-bit indices
#                    64       : 64-bit indices
#                    both     : both of the above options
#   <lib type>   = Which library type to test:
#                    static   : only build static libraries
#                    shared   : only build shared libraries
#                    each     : build static and shared separately
#                    both     : build static and shared simultaneously
#   <TPL status> = Enable/disable third party libraries:
#                    ON       : All possible TPLs enabled
#                    OFF      : No TPLs enabled
#   <test type>  = Test type to run:
#                    CONFIG   : configure only
#                    BUILD    : build only
#                    STD      : standard tests
#                    DEV      : development tests
#   <memcheck>   = Enable/disable memcheck test:
#                    ON       : run test_memcheck
#                    OFF      : do not run test_memcheck
#
# Optional Inputs:
#   <build threads> = number of threads to use in parallel build (default 1)
# ------------------------------------------------------------------------------

# check number of inputs
if [ "$#" -lt 6 ]; then
    echo "ERROR: SEVEN (7) inputs required"
    echo "package      : [arkode|cvode|cvodes|ida|idas|kinsol|sundials|all]"
    echo "real type    : [single|double|extended|all]"
    echo "index size   : [32|64|both]"
    echo "library type : [static|shared|each|both]"
    echo "TPLs         : [ON|OFF]"
    echo "test type    : [CONFIG|BUILD|STD|DEV]"
    echo "memcheck     : [ON|OFF]"
    exit 1
fi

package=$1      # sundials package to test
realtype=$2     # precision for realtypes
indexsize=$3    # integer size for indices
libtype=$4      # library type to build
tplstatus=$5    # enable/disable third party libraries
testtype=$6     # which test type to run
memcheck=$7     # memcheck test (make test_memcheck)
buildthreads=1  # default number threads for parallel builds

# check if the number of build threads was set
if [ "$#" -gt 7 ]; then
    buildthreads=$8
fi

# ------------------------------------------------------------------------------
# Check inputs
# ------------------------------------------------------------------------------

# check package option
case $package in
    arkode|cvode|cvodes|ida|idas|kinsol|sundials|all) ;;
    *)
        echo "ERROR: Unknown package option: $package"
        exit 1
        ;;
esac

# set real types to test
case "$realtype" in
    SINGLE|Single|single)       realtype=( "single" );;
    DOUBLE|Double|double)       realtype=( "double" );;
    EXTENDED|Extended|extended) realtype=( "extended" );;
    ALL|All|all)                realtype=( "single" "double" "extended" );;
    *)
        echo "ERROR: Unknown real type option: $realtype"
        exit 1
        ;;
esac

# set index sizes to test
case "$indexsize" in
    32)   indexsize=( "32" );;
    64)   indexsize=( "64" );;
    both) indexsize=( "32" "64" );;
    *)
        echo "ERROR: Unknown index size option: $indexsize"
        exit 1
        ;;
esac

# set library types
case "$libtype" in
    STATIC|Static|static) libtype=( "static" );;
    SHARED|Shared|shared) libtype=( "shared" );;
    EACH|Each|each)       libtype=( "static" "shared" ) ;;
    BOTH|Both|both)       libtype=( "both" ) ;;
    *)
        echo "ERROR: Unknown library type: $libtype"
        exit 1
        ;;
esac

# set TPL status
case "$tplstatus" in
    ON|On|on)
        TPLs=ON
        ;;
    OFF|Off|off)
        TPLs=OFF
        ;;
    *)
        echo "ERROR: Unknown third party library status: $tplstatus"
        exit 1
        ;;
esac

# which tests to run (if any)
case "$testtype" in
    CONFIG|Config|config)
        # configure only, do not compile or test
        testtype=CONFIG
        ;;
    BUILD|Build|build)
        # configure and compile only, do not test
        testtype=BUILD
        ;;
    STD|Std|std)
        # configure, compile, and run standard tests
        testtype=STD
        ;;
    DEV|Dev|dev)
        # configure, compile, and run development tests
        testtype=DEV
        ;;
    *)
        echo "ERROR: Unknown test option: $testtype"
        exit 1
        ;;
esac

# which tests to run (if any)
case "$memcheck" in
    ON|On|on)
        memcheck=ON
        ;;
    OFF|Off|off)
        memcheck=OFF
        ;;
    *)
        echo "ERROR: Unknown memcheck option: $memcheck"
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# Create tarballs
# ------------------------------------------------------------------------------

# location of testing directory
testdir=`pwd`

# remove old tarball directory and create new directory
\rm -rf tarballs || exit 1
mkdir tarballs   || exit 1

# run tarscript to create tarballs
cd ../scripts || exit 1

echo "START TARSCRIPT"
./tarscript $package | tee -a tar.log

# check tarscript return code
rc=${PIPESTATUS[0]}
echo -e "\ntarscript returned $rc\n" | tee -a tar.log
if [ $rc -ne 0 ]; then
    # remove temporary file created by tarscript and exit with error
    \rm -rf ../../tmp_dir.*
    exit 1;
fi

# relocate tarballs
mv tar.log $testdir/tarballs/.

# move tarballs to tarball directory
case $package in
    arkode)
        mv ../../arkode-*.tar.gz $testdir/tarballs/.   || exit 1
        ;;
    cvode)
        mv ../../cvode-*.tar.gz $testdir/tarballs/.    || exit 1
        ;;
    cvodes)
        mv ../../cvodes-*.tar.gz $testdir/tarballs/.   || exit 1
        ;;
    ida)
        mv ../../ida-*.tar.gz $testdir/tarballs/.      || exit 1
        ;;
    idas)
        mv ../../idas-*.tar.gz $testdir/tarballs/.     || exit 1
        ;;
    kinsol)
        mv ../../kinsol-*.tar.gz $testdir/tarballs/.   || exit 1
        ;;
    sundials)
        mv ../../sundials-*.tar.gz $testdir/tarballs/. || exit 1
        ;;
    all)
        mv ../../sundials-*.tar.gz $testdir/tarballs/. || exit 1
        mv ../../arkode-*.tar.gz $testdir/tarballs/.   || exit 1
        mv ../../cvode-*.tar.gz $testdir/tarballs/.    || exit 1
        mv ../../cvodes-*.tar.gz $testdir/tarballs/.   || exit 1
        mv ../../ida-*.tar.gz $testdir/tarballs/.      || exit 1
        mv ../../idas-*.tar.gz $testdir/tarballs/.     || exit 1
        mv ../../kinsol-*.tar.gz $testdir/tarballs/.   || exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# Test tarballs
# ------------------------------------------------------------------------------

# move to tarball directory
cd $testdir/tarballs || exit 1

# loop over tarballs and test each one
for tarball in *.tar.gz; do

    # get package name
    package=${tarball%.tar.gz}

    echo "START UNTAR"
    tar -xvzf $tarball 2>&1 | tee -a tar.log

    # check tar return code
    rc=${PIPESTATUS[0]}
    echo -e "\ntar -xzvf returned $rc\n" | tee -a tar.log
    if [ $rc -ne 0 ]; then exit 1; fi

    # move log to package directory
    mv tar.log $package/. || exit 1

    # move to the extracted package's test directory
    cd $package/test || exit 1

    # copy environment and testing scripts from original test directory
    cp $testdir/env*.sh .    || exit 1
    cp $testdir/suntest.sh . || exit 1

    # loop over build options
    for rt in "${realtype[@]}"; do
        for is in "${indexsize[@]}"; do
            for lt in "${libtype[@]}"; do

                # print test label for Jenkins section collapsing
                echo "TEST: $rt $is $lt $TPLs $testtype $memcheck $buildthreads"

                ./suntest.sh $rt $is $lt $TPLs $testtype $memcheck $buildthreads

                # check return flag
                if [ $? -ne 0 ]; then
                    echo "FAILED: $rt $is $lt $TPLs $testtype $memcheck $buildthreads"
                    cd $testdir
                    exit 1
                else
                    echo "PASSED"
                fi

            done
        done
    done

    # return to tarball directory
    cd $testdir/tarballs || exit 1

done

# ------------------------------------------------------------------------------
# Return
# ------------------------------------------------------------------------------

# if we make it here all tests have passed
cd $testdir
exit 0
