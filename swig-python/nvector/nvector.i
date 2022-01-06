// ---------------------------------------------------------------
// Programmer: Cody J. Balos @ LLNL
// ---------------------------------------------------------------
// SUNDIALS Copyright Start
// Copyright (c) 2002-2019, Lawrence Livermore National Security
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

// Macro for creating an interface to an N_Vector
%define %nvector_impl(TYPE)
  %ignore _N_VectorContent_## TYPE ##;
%enddef

%nvector_impl(Serial)

// Typemap for input arrays (e.g. this effects N_VMake functions)
%apply (int DIM1, double* IN_ARRAY1) {(sunindextype vec_length, realtype *v_data)}

// Typemape for output arrays (e.g. N_VArrayView)
%apply (int* DIM1, double** ARGOUTVIEW_ARRAY1) {(sunindextype* length, realtype** array)}

// include the header file in the swig wrapper
%{
#include "sundials/sundials_nvector.h"
#include "nvector/nvector_serial.h"
%}

// Process and wrap functions in the following files
%include "sundials/sundials_nvector.h"
%include "nvector/nvector_serial.h"
