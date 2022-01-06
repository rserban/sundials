// ---------------------------------------------------------------
// Programmer: Cody J. Balos @ LLNL
// ---------------------------------------------------------------
// SUNDIALS Copyright Start
// Copyright (c) 2002-2022, Lawrence Livermore National Security
// and Southern Methodist University.
// All rights reserved.
//
// See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
// SUNDIALS Copyright End
// ---------------------------------------------------------------
// Swig interface file
// ---------------------------------------------------------------

// Macro for creating an interface to a SUNMatrix
%define %sunmatrix_impl(TYPE)
  %ignore _SUNMatrixContent_## TYPE ##;
%enddef

// %sunmatrix_impl(dense)

%apply (int* DIM1, double** ARGOUTVIEW_ARRAY1) {(sunindextype* length, realtype** array)}

// include the header file in the swig wrapper
%{
#include "sundials/sundials_matrix.h"
#include "sunmatrix/sunmatrix_dense.h"
%}

// Process and wrap functions in the following files
%include "sundials/sundials_matrix.h"
%include "sunmatrix/sunmatrix_dense.h"
