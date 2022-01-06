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

// Macro for creating an interface to an SUNLinearSolver
%define %sunlinsol_impl(TYPE)
  %ignore _SUNLinearSolverContent_## TYPE ##;
%enddef

%sunlinsol_impl(dense)
%sunlinsol_impl(pcg)
%sunlinsol_impl(spbcgs)
%sunlinsol_impl(spfgmr)
%sunlinsol_impl(spgmr)
%sunlinsol_impl(sptfqmr)

// include the header file in the swig wrapper
%{
#include "sundials/sundials_iterative.h"
#include "sundials/sundials_linearsolver.h"
#include "sunlinsol/sunlinsol_dense.h"
#include "sunlinsol/sunlinsol_pcg.h"
#include "sunlinsol/sunlinsol_spbcgs.h"
#include "sunlinsol/sunlinsol_spfgmr.h"
#include "sunlinsol/sunlinsol_spgmr.h"
#include "sunlinsol/sunlinsol_sptfqmr.h"
%}

// Process and wrap functions in the following files
%include "sundials/sundials_iterative.h"
%include "sundials/sundials_linearsolver.h"
%include "sunlinsol/sunlinsol_dense.h"
%include "sunlinsol/sunlinsol_pcg.h"
%include "sunlinsol/sunlinsol_spbcgs.h"
%include "sunlinsol/sunlinsol_spfgmr.h"
%include "sunlinsol/sunlinsol_spgmr.h"
%include "sunlinsol/sunlinsol_sptfqmr.h"
