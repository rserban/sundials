#!/bin/bash
# ------------------------------------------------------------------------------
# Programmer(s): David J. Gardner @ LLNL
# ------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2019, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ------------------------------------------------------------------------------
# Script for creating Valgrind suppression files from log files generated using
# the --gen-suppressions option.
#
# This script will extract the unique suppression blocks in the input log file
# and write them to the given output file. If the output file is the same as
# the input file, the input file will be overwritten otherwise the suppression
# blocks will be appended to the output file.
#
# Usage: ./create_supp.sh <input file name> <output file name>
#
# Required Inputs:
#   <input file name>  = the full path to the log file to be processed
#   <output file name> = the full path to the output suppression file
# ------------------------------------------------------------------------------

# exit when any command fails
set -e

# check number of inputs
if [ "$#" -lt 2 ]; then
    echo "ERROR: TWO (2) inputs required"
    echo "input file name  : full path to the log file to be processed"
    echo "output file name : full path to the output suppression file"
    exit 1
fi
infile=$1
outfile=$2

# check that the file exists
if [ ! -f "$infile" ]; then
    echo "ERROR: $infile does not exist"
    exit 1
fi

# For easier debugging/updating each step is written to a temporary file
# 1. remove Valgrind output
# 2. remove all new lines
# 3. make each supression block a separate line
# 4. sort the suppression blocks
# 5. remove duplicate suppression blocks
# 6. insert new line after each opening brace
# 7. insert new line before each closing brace
# 8. break suppression paths into separate lines
# 9. remove empty lines
sed '/==.*/d' $infile > tmp.1
tr -d '\n' < tmp.1 > tmp.2
sed 's/{*}/&\n/g' tmp.2 > tmp.3
sort tmp.3 > tmp.4
uniq tmp.4 > tmp.5
sed 's/{/&\n/g' tmp.5 > tmp.6
sed 's/}/\n&/g' tmp.6 > tmp.7
sed 's/   /\n   /g' tmp.7 > tmp.8
sed '/^$/d' tmp.8 > tmp.9

# overwrite/append the output file
if [ "$infile" == "$outfile" ]; then
    cat tmp.9 > $outfile
else
    cat tmp.9 >> $outfile
fi

# remove temporary files
\rm -f tmp.[0-9]
