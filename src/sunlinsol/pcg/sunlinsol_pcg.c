/* -----------------------------------------------------------------
 * Programmer(s): Daniel Reynolds, Ashley Crawford @ SMU
 * Based on sundials_pcg.c code, written by Daniel Reynolds @ SMU
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
 * This is the implementation file for the PCG implementation of
 * the SUNLINSOL package.
 * -----------------------------------------------------------------*/

#include <stdio.h>
#include <stdlib.h>

#include <sunlinsol/sunlinsol_pcg.h>
#include <sundials/sundials_math.h>

#include "sundials/sundials_errors.h"
#include "sundials/sundials_types.h"
#include "sundials_context_impl.h"
#include "sundials_logger_impl.h"

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)

/*
 * -----------------------------------------------------------------
 * PCG solver structure accessibility macros:
 * -----------------------------------------------------------------
 */

#define PCG_CONTENT(S)  ( (SUNLinearSolverContent_PCG)(S->content) )
#define PRETYPE(S)      ( PCG_CONTENT(S)->pretype )
#define LASTFLAG(S)     ( PCG_CONTENT(S)->last_flag )

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------------------
 * Function to create a new PCG linear solver
 */

SUNLinearSolver SUNLinSol_PCG(N_Vector y, int pretype, int maxl, SUNContext sunctx)
{
  SUNLinearSolver S;
  SUNLinearSolverContent_PCG content;

  SUNAssertContext(sunctx);

  /* check for legal pretype and maxl values; if illegal use defaults */
  if ((pretype != SUN_PREC_NONE)  && (pretype != SUN_PREC_LEFT) &&
      (pretype != SUN_PREC_RIGHT) && (pretype != SUN_PREC_BOTH))
    pretype = SUN_PREC_NONE;
  if (maxl <= 0)
    maxl = SUNPCG_MAXL_DEFAULT;

  /* Create linear solver */
  S = NULL;
  S = SUNCheckCallLastErrReturnNull(SUNLinSolNewEmpty(sunctx), sunctx);

  /* Attach operations */
  S->ops->gettype           = SUNLinSolGetType_PCG;
  S->ops->getid             = SUNLinSolGetID_PCG;
  S->ops->setatimes         = SUNLinSolSetATimes_PCG;
  S->ops->setpreconditioner = SUNLinSolSetPreconditioner_PCG;
  S->ops->setscalingvectors = SUNLinSolSetScalingVectors_PCG;
  S->ops->setzeroguess      = SUNLinSolSetZeroGuess_PCG;
  S->ops->initialize        = SUNLinSolInitialize_PCG;
  S->ops->setup             = SUNLinSolSetup_PCG;
  S->ops->solve             = SUNLinSolSolve_PCG;
  S->ops->numiters          = SUNLinSolNumIters_PCG;
  S->ops->resnorm           = SUNLinSolResNorm_PCG;
  S->ops->resid             = SUNLinSolResid_PCG;
  S->ops->lastflag          = SUNLinSolLastFlag_PCG;
  S->ops->space             = SUNLinSolSpace_PCG;
  S->ops->free              = SUNLinSolFree_PCG;

  /* Create content */
  content = NULL;
  content = (SUNLinearSolverContent_PCG) malloc(sizeof *content);
  SUNAssert(content, SUN_ERR_MALLOC_FAIL, sunctx);

  /* Attach content  */
  S->content = content;

  /* Fill content */
  content->last_flag   = 0;
  content->maxl        = maxl;
  content->pretype     = pretype;
  content->zeroguess   = SUNFALSE;
  content->numiters    = 0;
  content->resnorm     = ZERO;
  content->r           = NULL;
  content->p           = NULL;
  content->z           = NULL;
  content->Ap          = NULL;
  content->s           = NULL;
  content->ATimes      = NULL;
  content->ATData      = NULL;
  content->Psetup      = NULL;
  content->Psolve      = NULL;
  content->PData       = NULL;
  content->print_level = 0;
  content->info_file   = stdout;
#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
  content->info_file   = (sunctx->logger->info_fp) ? sunctx->logger->info_fp : stdout;
#endif

  /* Allocate content */
  content->r = SUNCheckCallLastErrReturnNull(N_VClone(y), sunctx);

  content->p = SUNCheckCallLastErrReturnNull(N_VClone(y), sunctx);

  content->z = SUNCheckCallLastErrReturnNull(N_VClone(y), sunctx);

  content->Ap = SUNCheckCallLastErrReturnNull(N_VClone(y), sunctx);

  return(S);
}


/* ----------------------------------------------------------------------------
 * Function to set the type of preconditioning for PCG to use
 */

SUNErrCode SUNLinSol_PCGSetPrecType(SUNLinearSolver S, int pretype)
{
  /* Check for legal pretype */
  SUNAssert((pretype == SUN_PREC_NONE) || (pretype == SUN_PREC_LEFT) ||
              (pretype == SUN_PREC_RIGHT) || (pretype == SUN_PREC_BOTH),
            SUN_ERR_ARG_OUTOFRANGE, S->sunctx);

  /* Set pretype */
  PRETYPE(S) = pretype;
  return SUN_SUCCESS;
}


/* ----------------------------------------------------------------------------
 * Function to set the maximum number of iterations for PCG to use
 */

SUNErrCode SUNLinSol_PCGSetMaxl(SUNLinearSolver S, int maxl)
{
  /* Check for legal number of iters */
  if (maxl <= 0)
    maxl = SUNPCG_MAXL_DEFAULT;

  /* Set max iters */
  PCG_CONTENT(S)->maxl = maxl;
  return SUN_SUCCESS;
}


/*
 * -----------------------------------------------------------------
 * implementation of linear solver operations
 * -----------------------------------------------------------------
 */

SUNLinearSolver_Type SUNLinSolGetType_PCG(SUNLinearSolver S) SUNDIALS_NOEXCEPT
{
  return(SUNLINEARSOLVER_ITERATIVE);
}


SUNLinearSolver_ID SUNLinSolGetID_PCG(SUNLinearSolver S) SUNDIALS_NOEXCEPT
{
  return(SUNLINEARSOLVER_PCG);
}


SUNErrCode SUNLinSolInitialize_PCG(SUNLinearSolver S)
{
  if (PCG_CONTENT(S)->maxl <= 0)
    PCG_CONTENT(S)->maxl = SUNPCG_MAXL_DEFAULT;

  SUNAssert(PCG_CONTENT(S)->ATimes, SUN_ERR_ARG_CORRUPT, S->sunctx);

  if ( (PRETYPE(S) != SUN_PREC_LEFT) &&
       (PRETYPE(S) != SUN_PREC_RIGHT) &&
       (PRETYPE(S) != SUN_PREC_BOTH) )
    PRETYPE(S) = SUN_PREC_NONE;

  SUNAssert((PCG_CONTENT(S)->pretype == SUN_PREC_NONE) ||
              (PCG_CONTENT(S)->Psolve != NULL),
            SUN_ERR_ARG_CORRUPT, S->sunctx);

  /* no additional memory to allocate */

  return SUN_SUCCESS;
}


SUNErrCode SUNLinSolSetATimes_PCG(SUNLinearSolver S, void* ATData,
                                  SUNATimesFn ATimes)
{
  /* set function pointers to integrator-supplied ATimes routine
     and data, and return with success */
  PCG_CONTENT(S)->ATimes = ATimes;
  PCG_CONTENT(S)->ATData = ATData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetPreconditioner_PCG(SUNLinearSolver S, void* PData,
                                          SUNPSetupFn Psetup, SUNPSolveFn Psolve)
{
  /* set function pointers to integrator-supplied Psetup and PSolve
     routines and data, and return with success */
  PCG_CONTENT(S)->Psetup = Psetup;
  PCG_CONTENT(S)->Psolve = Psolve;
  PCG_CONTENT(S)->PData = PData;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetScalingVectors_PCG(SUNLinearSolver S, N_Vector s,
                                          N_Vector nul)
{
  /* set N_Vector pointer to integrator-supplied scaling vector
     (only use the first one), and return with success */
  PCG_CONTENT(S)->s = s;
  return SUN_SUCCESS;
}


SUNErrCode SUNLinSolSetZeroGuess_PCG(SUNLinearSolver S, booleantype onoff)
{
  /* set flag indicating a zero initial guess */
  PCG_CONTENT(S)->zeroguess = onoff;
  return SUN_SUCCESS;
}


SUNLsStatus SUNLinSolSetup_PCG(SUNLinearSolver S, SUNMatrix nul)
{
  SUNLsStatus status;
  SUNPSetupFn Psetup;
  void* PData;

  /* Set shortcuts to PCG memory structures */
  Psetup = PCG_CONTENT(S)->Psetup;
  PData = PCG_CONTENT(S)->PData;

  /* no solver-specific setup is required, but if user-supplied
     Psetup routine exists, call that here */
  if (Psetup != NULL) {
    status = Psetup(PData);
    if (status != 0) {
      LASTFLAG(S) = (status < 0) ?
        SUNLS_PSET_FAIL_UNREC : SUNLS_PSET_FAIL_REC;
      return(LASTFLAG(S));
    }
  }

  /* return with success */
  LASTFLAG(S) = SUNLS_SUCCESS;
  return(SUNLS_SUCCESS);
}


SUNLsStatus SUNLinSolSolve_PCG(SUNLinearSolver S, SUNMatrix nul, N_Vector x,
                               N_Vector b, realtype delta)
{
  /* local data and shortcut variables */
  realtype alpha, beta, r0_norm, rho, rz, rz_old;
  N_Vector r, p, z, Ap, w;
  booleantype UsePrec, UseScaling, converged;
  booleantype *zeroguess;
  int l, l_max, pretype;
  void *A_data, *P_data;
  SUNATimesFn atimes;
  SUNPSolveFn psolve;
  realtype *res_norm;
  int *nli;
  SUNLsStatus status;
  SUNContext sunctx = S->sunctx;

  /* Make local shorcuts to solver variables. */
  l_max        = PCG_CONTENT(S)->maxl;
  r            = PCG_CONTENT(S)->r;
  p            = PCG_CONTENT(S)->p;
  z            = PCG_CONTENT(S)->z;
  Ap           = PCG_CONTENT(S)->Ap;
  w            = PCG_CONTENT(S)->s;
  A_data       = PCG_CONTENT(S)->ATData;
  P_data       = PCG_CONTENT(S)->PData;
  atimes       = PCG_CONTENT(S)->ATimes;
  psolve       = PCG_CONTENT(S)->Psolve;
  pretype      = PCG_CONTENT(S)->pretype;
  zeroguess    = &(PCG_CONTENT(S)->zeroguess);
  nli          = &(PCG_CONTENT(S)->numiters);
  res_norm     = &(PCG_CONTENT(S)->resnorm);

  /* Initialize counters and convergence flag */
  *nli = 0;
  converged = SUNFALSE;

  /* set booleantype flags for internal solver options */
  UsePrec = ( (pretype == SUN_PREC_BOTH) ||
              (pretype == SUN_PREC_LEFT) ||
              (pretype == SUN_PREC_RIGHT) );
  UseScaling = (w != NULL);

#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
  if (PCG_CONTENT(S)->print_level && PCG_CONTENT(S)->info_file
      && (PCG_CONTENT(S)->info_file != S->sunctx->logger->info_fp))
    fprintf(PCG_CONTENT(S)->info_file, "SUNLINSOL_PCG:\n");
#endif

  /* Check if Atimes function has been set */
  SUNAssert(atimes, SUN_ERR_ARG_CORRUPT, S->sunctx);

  /* If preconditioning, check if psolve has been set */
  SUNAssert(!UsePrec || psolve, SUN_ERR_ARG_CORRUPT, S->sunctx);

  /* Set r to initial residual r_0 = b - A*x_0 */
  if (*zeroguess) {
    SUNCheckCallLastErr(N_VScale(ONE, b, r), sunctx);
  } else {
    status = atimes(A_data, x, r);
    if (status != 0) {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ?
        SUNLS_ATIMES_FAIL_UNREC : SUNLS_ATIMES_FAIL_REC;
      return(LASTFLAG(S));
    }
    SUNCheckCallLastErr(N_VLinearSum(ONE, b, -ONE, r, r), sunctx);
  }

  /* Set rho to scaled L2 norm of r, and return if small */
  if (UseScaling) {
    SUNCheckCallLastErr(N_VProd(r, w, Ap), sunctx);
  } else {
    SUNCheckCallLastErr(N_VScale(ONE, r, Ap), sunctx);
  }
  rho = SUNCheckCallLastErr(N_VDotProd(Ap, Ap), sunctx);
  *res_norm = r0_norm = rho = SUNRsqrt(rho);

#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
  /* print initial residual */
  if (PCG_CONTENT(S)->print_level && PCG_CONTENT(S)->info_file
      && (PCG_CONTENT(S)->info_file != S->sunctx->logger->info_fp))
  {
    fprintf(PCG_CONTENT(S)->info_file,
            SUNLS_MSG_RESIDUAL,
            (long int) 0, *res_norm);
  }
  SUNLogger_QueueMsg(S->sunctx->logger, SUN_LOGLEVEL_INFO,
    "SUNLinSolSolve_PCG", "initial-residual",
    "nli = %li, resnorm = %.16g", (long int) 0, *res_norm);
#endif

  if (rho <= delta) {
    *zeroguess  = SUNFALSE;
    LASTFLAG(S) = SUNLS_SUCCESS;
    return(LASTFLAG(S));
  }

  /* Apply preconditioner and b-scaling to r = r_0 */
  if (UsePrec) {
    status = psolve(P_data, r, z, delta, SUN_PREC_LEFT);   /* z = P^{-1}r */
    if (status != 0) {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ?
        SUNLS_PSOLVE_FAIL_UNREC : SUNLS_PSOLVE_FAIL_REC;
      return(LASTFLAG(S));
    }
  } else {
    SUNCheckCallLastErr(N_VScale(ONE, r, z), sunctx);
  }


  /* Initialize rz to <r,z> */
  rz = SUNCheckCallLastErr(N_VDotProd(r, z), sunctx);

  /* Copy z to p */
  SUNCheckCallLastErr(N_VScale(ONE, z, p), sunctx);

  /* Begin main iteration loop */
  for(l=0; l<l_max; l++) {

    /* increment counter */
    (*nli)++;

    /* Generate Ap = A*p */
    status = atimes(A_data, p, Ap);
    if (status != 0) {
      *zeroguess  = SUNFALSE;
      LASTFLAG(S) = (status < 0) ?
        SUNLS_ATIMES_FAIL_UNREC : SUNLS_ATIMES_FAIL_REC;
      return(LASTFLAG(S));
    }

    /* Calculate alpha = <r,z> / <Ap,p> */
    alpha = SUNCheckCallLastErr(N_VDotProd(Ap, p), sunctx);
    alpha = rz / alpha;

    /* Update x = x + alpha*p */
    if (l == 0 && *zeroguess) {
      SUNCheckCallLastErr(N_VScale(alpha, p, x), sunctx);
    } else {
      SUNCheckCallLastErr(N_VLinearSum(ONE, x, alpha, p, x), sunctx);
    }

    /* Update r = r - alpha*Ap */
    SUNCheckCallLastErr(N_VLinearSum(ONE, r, -alpha, Ap, r), sunctx);

    /* Set rho and check convergence */
    if (UseScaling) {
      SUNCheckCallLastErr(N_VProd(r, w, Ap), sunctx);
    } else {
      SUNCheckCallLastErr(N_VScale(ONE, r, Ap), sunctx);
    }
    rho = SUNCheckCallLastErr(N_VDotProd(Ap, Ap), sunctx);
    *res_norm = rho = SUNRsqrt(rho);

#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO
    /* print current iteration number and the residual */
    if (PCG_CONTENT(S)->print_level && PCG_CONTENT(S)->info_file
      && (PCG_CONTENT(S)->info_file != S->sunctx->logger->info_fp))
    {
      fprintf(PCG_CONTENT(S)->info_file,
              SUNLS_MSG_RESIDUAL,
              (long int) *nli, *res_norm);
    }
    SUNLogger_QueueMsg(S->sunctx->logger, SUN_LOGLEVEL_INFO,
      "SUNLinSolSolve_PCG", "iterate-residual",
      "nli = %li, resnorm = %.16g", (long int) 0, *res_norm);
#endif

    if (rho <= delta) {
      converged = SUNTRUE;
      break;
    }

    /* Exit early on last iteration */
    if (l == l_max - 1) break;

    /* Apply preconditioner:  z = P^{-1}*r */
    if (UsePrec) {
      status = psolve(P_data, r, z, delta, SUN_PREC_LEFT);
      if (status != 0) {
        *zeroguess  = SUNFALSE;
        LASTFLAG(S) = (status < 0) ?
          SUNLS_PSOLVE_FAIL_UNREC : SUNLS_PSOLVE_FAIL_REC;
        return(LASTFLAG(S));
      }
    } else {
      SUNCheckCallLastErr(N_VScale(ONE, r, z), sunctx);
    }

    /* update rz */
    rz_old = rz;
    rz = SUNCheckCallLastErr(N_VDotProd(r, z), sunctx);

    /* Calculate beta = <r,z> / <r_old,z_old> */
    beta = rz / rz_old;

    /* Update p = z + beta*p */
    SUNCheckCallLastErr(N_VLinearSum(ONE, z, beta, p, p), sunctx);
  }

  /* Main loop finished, return with result */
  *zeroguess = SUNFALSE;
  if (converged == SUNTRUE) {
    LASTFLAG(S) = SUNLS_SUCCESS;
  } else if (rho < r0_norm) {
    LASTFLAG(S) = SUNLS_RES_REDUCED;
  } else {
    LASTFLAG(S) = SUNLS_CONV_FAIL;
  }
  return(LASTFLAG(S));
}


int SUNLinSolNumIters_PCG(SUNLinearSolver S) SUNDIALS_NOEXCEPT
{
  /* return the stored 'numiters' value */
  return (PCG_CONTENT(S)->numiters);
}


realtype SUNLinSolResNorm_PCG(SUNLinearSolver S) SUNDIALS_NOEXCEPT
{
  /* return the stored 'resnorm' value */
  return (PCG_CONTENT(S)->resnorm);
}


N_Vector SUNLinSolResid_PCG(SUNLinearSolver S) SUNDIALS_NOEXCEPT
{
  /* return the stored 'r' vector */
  return (PCG_CONTENT(S)->r);
}


sunindextype SUNLinSolLastFlag_PCG(SUNLinearSolver S) SUNDIALS_NOEXCEPT
{
  /* return the stored 'last_flag' value */
  return (LASTFLAG(S));
}

SUNErrCode SUNLinSolSpace_PCG(SUNLinearSolver S, long int* lenrwLS,
                              long int* leniwLS)
{
  sunindextype liw1, lrw1;
  SUNCheckCallLastErr(N_VSpace(PCG_CONTENT(S)->r, &lrw1, &liw1), S->sunctx);
  *lenrwLS = 1 + lrw1*4;
  *leniwLS = 4 + liw1*4;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolFree_PCG(SUNLinearSolver S)
{
  if (S == NULL) return SUN_SUCCESS;

  if (S->content) {
    /* delete items from within the content structure */
    if (PCG_CONTENT(S)->r) {
      N_VDestroy(PCG_CONTENT(S)->r);
      PCG_CONTENT(S)->r = NULL;
    }
    if (PCG_CONTENT(S)->p) {
      N_VDestroy(PCG_CONTENT(S)->p);
      PCG_CONTENT(S)->p = NULL;
    }
    if (PCG_CONTENT(S)->z) {
      N_VDestroy(PCG_CONTENT(S)->z);
      PCG_CONTENT(S)->z = NULL;
    }
    if (PCG_CONTENT(S)->Ap) {
      N_VDestroy(PCG_CONTENT(S)->Ap);
      PCG_CONTENT(S)->Ap = NULL;
    }
    free(S->content); S->content = NULL;
  }
  if (S->ops) { free(S->ops); S->ops = NULL; }
  free(S); S = NULL;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetInfoFile_PCG(SUNLinearSolver S, FILE* info_file)
{
  PCG_CONTENT(S)->info_file = info_file;
  return SUN_SUCCESS;
}

SUNErrCode SUNLinSolSetPrintLevel_PCG(SUNLinearSolver S, int print_level)
{
  /* check for valid print level */
  SUNAssert(print_level >= 0 && print_level <= 1, SUN_ERR_ARG_OUTOFRANGE,
            S->sunctx);
  PCG_CONTENT(S)->print_level = print_level;
  return SUN_SUCCESS;
}
