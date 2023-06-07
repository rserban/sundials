/* -----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ SMU,
 *      Alan C. Hindmarsh, Radu Serban and Aaron Collier @ LLNL
 * -----------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2023, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------
 * This is the header file for the IDABBDPRE module, for a
 * band-block-diagonal preconditioner, i.e. a block-diagonal
 * matrix with banded blocks.
 * -----------------------------------------------------------------*/

#ifndef _IDABBDPRE_H
#define _IDABBDPRE_H

#include <sundials/sundials_nvector.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

/* User-supplied function Types */

typedef int (*IDABBDLocalFn)(sunindextype Nlocal, realtype tt, N_Vector yy,
                             N_Vector yp, N_Vector gval, void* user_data);

typedef int (*IDABBDCommFn)(sunindextype Nlocal, realtype tt, N_Vector yy,
                            N_Vector yp, void* user_data);

/* Exported Functions */

SUNDIALS_EXPORT int IDABBDPrecInit(void* ida_mem, sunindextype Nlocal,
                                   sunindextype mudq, sunindextype mldq,
                                   sunindextype mukeep, sunindextype mlkeep,
                                   realtype dq_rel_yy, IDABBDLocalFn Gres,
                                   IDABBDCommFn Gcomm);

SUNDIALS_EXPORT int IDABBDPrecReInit(void* ida_mem, sunindextype mudq,
                                     sunindextype mldq, realtype dq_rel_yy);

/* Optional output functions */

SUNDIALS_EXPORT int IDABBDPrecGetWorkSpace(void* ida_mem, long int* lenrwBBDP,
                                           long int* leniwBBDP);

SUNDIALS_EXPORT int IDABBDPrecGetNumGfnEvals(void* ida_mem,
                                             long int* ngevalsBBDP);

#ifdef __cplusplus
}
#endif

#endif
