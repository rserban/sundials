/*---------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU
 *---------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2023, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 *---------------------------------------------------------------
 * File that provides a globally-defined, but NULL-valued,
 * SUNLinearSolver object, to ensure that F2C_CVODE_linsol is
 * defined for cases when no linear solver object is linked in
 * with the main executable.
 *--------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>
#include "fcvode.h"
#include "cvode_impl.h"

/*=============================================================*/

/* Define global linear solver variable */

SUNLinearSolver F2C_CVODE_linsol;

/*=============================================================*/

/* C routine that is called when using fixed-point nonlinear solvers */
void FCVNullLinsol()
{
  F2C_CVODE_linsol = NULL;
}

/*===============================================================
   EOF
===============================================================*/
