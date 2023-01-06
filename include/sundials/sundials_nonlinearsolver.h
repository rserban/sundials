/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner, and Cody J. Balos @ LLNL
 * -----------------------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2023, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -----------------------------------------------------------------------------
 * This is the header file for a generic nonlinear solver package. It defines
 * the SUNNonlinearSolver structure (_generic_SUNNonlinearSolver) which contains
 * the following fields:
 *   - an implementation-dependent 'content' field which contains any internal
 *     data required by the solver
 *   - an 'ops' filed which contains a structure listing operations acting on/by
 *     such solvers
 *
 * We consider iterative nonlinear solvers for systems in both root finding
 * (F(y) = 0) or fixed-point (G(y) = y) form. As a result, some of the routines
 * are applicable only to one type of nonlinear solver.
 * -----------------------------------------------------------------------------
 * This header file contains:
 *   - function types supplied to a SUNNonlinearSolver,
 *   - enumeration constants for SUNDIALS-defined nonlinear solver types,
 *   - type declarations for the _generic_SUNNonlinearSolver and
 *     _generic_SUNNonlinearSolver_Ops structures, as well as references to
 *     pointers to such structures (SUNNonlinearSolver),
 *   - prototypes for the nonlinear solver functions which operate
 *     on/by SUNNonlinearSolver objects, and
 *   - return codes for SUNLinearSolver objects.
 * -----------------------------------------------------------------------------
 * At a minimum, a particular implementation of a SUNNonlinearSolver must do the
 * following:
 *   - specify the 'content' field of a SUNNonlinearSolver,
 *   - implement the operations on/by the SUNNonlinearSovler objects,
 *   - provide a constructor routine for new SUNNonlinearSolver objects
 *
 * Additionally, a SUNNonlinearSolver implementation may provide the following:
 *   - "Set" routines to control solver-specific parameters/options
 *   - "Get" routines to access solver-specific performance metrics
 * ---------------------------------------------------------------------------*/

#ifndef _SUNNONLINEARSOLVER_H
#define _SUNNONLINEARSOLVER_H

#include <sundials/sundials_config.h>
#include <sundials/sundials_context.h>
#include <sundials/sundials_errors.h>
#include <sundials/sundials_linearsolver.h>
#include <sundials/sundials_nvector.h>
#include <sundials/sundials_types.h>

#ifdef __cplusplus /* wrapper to enable C++ usage */
extern "C" {
#endif

/* -----------------------------------------------------------------------------
 *  Forward references for SUNNonlinearSolver types defined below
 * ---------------------------------------------------------------------------*/

/* Forward reference for pointer to SUNNonlinearSolver_Ops object */
typedef _SUNDIALS_STRUCT_ _generic_SUNNonlinearSolver_Ops* SUNNonlinearSolver_Ops;

/* Forward reference for pointer to SUNNonlinearSolver object */
typedef _SUNDIALS_STRUCT_ _generic_SUNNonlinearSolver* SUNNonlinearSolver;

/* -----------------------------------------------------------------------------
 * Integrator supplied function types
 * ---------------------------------------------------------------------------*/

typedef SUNNlsStatus (*SUNNonlinSolSysFn)(N_Vector y, N_Vector F, void* mem);

typedef SUNNlsStatus (*SUNNonlinSolLSetupFn)(booleantype jbad, booleantype* jcur,
                                    void* mem);

typedef SUNNlsStatus (*SUNNonlinSolLSolveFn)(N_Vector b, void* mem);

typedef SUNNlsStatus (*SUNNonlinSolConvTestFn)(SUNNonlinearSolver NLS, N_Vector y,
                                      N_Vector del, realtype tol, N_Vector ewt,
                                      void* mem);

/* -----------------------------------------------------------------------------
 * SUNNonlinearSolver types
 * ---------------------------------------------------------------------------*/

typedef enum
{
  SUNNONLINEARSOLVER_ROOTFIND,
  SUNNONLINEARSOLVER_FIXEDPOINT
} SUNNonlinearSolver_Type;

/* -----------------------------------------------------------------------------
 * Generic definition of SUNNonlinearSolver
 * ---------------------------------------------------------------------------*/

/* Structure containing function pointers to nonlinear solver operations */
struct _generic_SUNNonlinearSolver_Ops
{
  SUNNonlinearSolver_Type (*gettype)(SUNNonlinearSolver);
  SUNErrCode (*initialize)(SUNNonlinearSolver);
  SUNNlsStatus (*setup)(SUNNonlinearSolver, N_Vector, void*);
  SUNNlsStatus (*solve)(SUNNonlinearSolver, N_Vector, N_Vector, N_Vector,
                       realtype, booleantype, void*);
  SUNErrCode (*free)(SUNNonlinearSolver);
  SUNErrCode (*setsysfn)(SUNNonlinearSolver, SUNNonlinSolSysFn);
  SUNErrCode (*setlsetupfn)(SUNNonlinearSolver, SUNNonlinSolLSetupFn);
  SUNErrCode (*setlsolvefn)(SUNNonlinearSolver, SUNNonlinSolLSolveFn);
  SUNErrCode (*setctestfn)(SUNNonlinearSolver, SUNNonlinSolConvTestFn, void*);
  SUNErrCode (*setmaxiters)(SUNNonlinearSolver, int);
  SUNErrCode (*getnumiters)(SUNNonlinearSolver, long int*);
  SUNErrCode (*getcuriter)(SUNNonlinearSolver, int*);
  SUNErrCode (*getnumconvfails)(SUNNonlinearSolver, long int*);
#ifdef __cplusplus
  _generic_SUNNonlinearSolver_Ops() = default;
#endif
};

/* A nonlinear solver is a structure with an implementation-dependent 'content'
   field, and a pointer to a structure of solver nonlinear solver operations
   corresponding to that implementation. */
struct _generic_SUNNonlinearSolver
{
  void* content;
  SUNNonlinearSolver_Ops ops;
  SUNContext sunctx;
#ifdef __cplusplus
  _generic_SUNNonlinearSolver() = default;
#endif
};

/* -----------------------------------------------------------------------------
 * Functions exported by SUNNonlinearSolver module
 * ---------------------------------------------------------------------------*/

/* empty constructor/destructor */
SUNDIALS_EXPORT
SUNNonlinearSolver SUNNonlinSolNewEmpty(SUNContext sunctx);

SUNDIALS_EXPORT
void SUNNonlinSolFreeEmpty(SUNNonlinearSolver NLS) SUNDIALS_NOEXCEPT;

/* core functions */
SUNDIALS_EXPORT SUNDIALS_PURE_VIRTUAL SUNNonlinearSolver_Type
SUNNonlinSolGetType(SUNNonlinearSolver NLS);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolInitialize(SUNNonlinearSolver NLS);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolSetup(SUNNonlinearSolver NLS, N_Vector y, void* mem);

SUNDIALS_EXPORT SUNDIALS_PURE_VIRTUAL
SUNNlsStatus SUNNonlinSolSolve(SUNNonlinearSolver NLS, N_Vector y0, N_Vector y, N_Vector w,
                  realtype tol, booleantype callLSetup, void* mem);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolFree(SUNNonlinearSolver NLS);

/* set functions */
SUNDIALS_EXPORT SUNDIALS_PURE_VIRTUAL SUNErrCode
SUNNonlinSolSetSysFn(SUNNonlinearSolver NLS, SUNNonlinSolSysFn SysFn);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolSetLSetupFn(SUNNonlinearSolver NLS,
                                   SUNNonlinSolLSetupFn SetupFn);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolSetLSolveFn(SUNNonlinearSolver NLS,
                                   SUNNonlinSolLSolveFn SolveFn);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolSetConvTestFn(SUNNonlinearSolver NLS,
                                     SUNNonlinSolConvTestFn CTestFn,
                                     void* ctest_data);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolSetMaxIters(SUNNonlinearSolver NLS, int maxiters);

/* get functions */
SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolGetNumIters(SUNNonlinearSolver NLS, long int* niters);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolGetCurIter(SUNNonlinearSolver NLS, int* iter);

SUNDIALS_EXPORT
SUNErrCode SUNNonlinSolGetNumConvFails(SUNNonlinearSolver NLS,
                                       long int* nconvfails);

/* -----------------------------------------------------------------------------
 * SUNNonlinearSolver return values
 * ---------------------------------------------------------------------------*/

#define SUN_NLS_SUCCESS 0 /* successful / converged */

/* Recoverable */
#define SUN_NLS_CONTINUE   +901 /* not converged, keep iterating      */
#define SUN_NLS_CONV_RECVR +902 /* convergece failure, try to recover */

/* Unrecoverable */

/* DEPRECATED: use SUNErrCode instead */
#define SUN_NLS_MEM_NULL -901 /* memory argument is NULL            */
/* DEPRECATED: use SUNErrCode instead */
#define SUN_NLS_MEM_FAIL -902 /* failed memory access / allocation  */
/* DEPRECATED: use SUNErrCode instead */
#define SUN_NLS_ILL_INPUT -903 /* illegal function input             */
/* DEPRECATED: use SUNErrCode instead */
#define SUN_NLS_VECTOROP_ERR -904 /* failed NVector operation           */
/* DEPRECATED: use SUNErrCode instead */
#define SUN_NLS_EXT_FAIL -905 /* failed in external library call    */

/* -----------------------------------------------------------------------------
 * SUNNonlinearSolver messages
 * ---------------------------------------------------------------------------*/

#if defined(SUNDIALS_EXTENDED_PRECISION)
#define SUN_NLS_MSG_RESIDUAL "\tnonlin. iteration %ld, nonlin. residual: %Lg\n"
#elif defined(SUNDIALS_DOUBLE_PRECISION)
#define SUN_NLS_MSG_RESIDUAL "\tnonlin. iteration %ld, nonlin. residual: %g\n"
#else
#define SUN_NLS_MSG_RESIDUAL "\tnonlin. iteration %ld, nonlin. residual: %g\n"
#endif

#ifdef __cplusplus
}
#endif

#endif
