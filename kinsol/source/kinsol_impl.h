/*
 * -----------------------------------------------------------------
 * $Revision: 1.1 $
 * $Date: 2004-06-02 23:22:30 $
 * -----------------------------------------------------------------
 * Programmer(s): Allan Taylor, Alan Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California
 * Produced at the Lawrence Livermore National Laboratory
 * All rights reserved
 * For details, see sundials/kinsol/LICENSE
 * -----------------------------------------------------------------
 * KINSOL solver module header file (private version)
 * -----------------------------------------------------------------
 */

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

#ifndef _kinsol_impl_h
#define _kinsol_impl_h

#include "kinsol.h"
#include "sundialstypes.h"
#include "nvector.h"

/*
 * -----------------------------------------------------------------
 * Types : struct KINMemRec and struct *KINMem
 * -----------------------------------------------------------------
 * A variable declaration of type struct *KINMem denotes a
 * pointer to a data structure of type struct KINMemRec. The
 * KINMemRec structure contains numerous fields that must be
 * accessible by KINSOL solver module routines.
 * -----------------------------------------------------------------
 */

typedef struct KINMemRec {

  realtype kin_uround; /* machine epsilon (or unit roundoff error) 
			  (defined in shared/include/sundialstypes.h)          */

  /* problem specification data */

  SysFn kin_func;              /* nonlinear system function implementation     */
  void *kin_f_data;            /* work space available to func routine         */
  realtype kin_fnormtol;       /* stopping tolerance on L2-norm of function
				  value                                        */
  realtype kin_scsteptol;      /* scaled step length tolerance                 */
  int kin_globalstrategy;      /* choices are INEXACT_NEWTON and LINESEARCH    */
  int kin_printfl;             /* level of verbosity of output                 */
  long int kin_mxiter;         /* maximum number of nonlinear iterations       */
  long int kin_msbpre;         /* maximum number of nonlinear iterations that
				  may be performed between calls to the
				  preconditioner setup routine (pset)          */
  int kin_etaflag;             /* choices are ETACONSTANT, ETACHOICE1 and
				  ETACHOICE2                                   */
  booleantype kin_noMinEps;    /* flag controlling whether or not the value
				  of eps is bounded below                      */
  booleantype kin_precondflag;     /* flag indicating if using preconditioning */
  booleantype kin_setupNonNull;    /* flag indicating if preconditioning setup
				      routine is non-null and if preconditioning
				      is being used                            */
  booleantype kin_constraintsSet;  /* flag indicating if constraints are being
				      used                                     */
  booleantype kin_precondcurrent;  /* flag indicating if the preconditioner is
				     current                                   */
  booleantype kin_callForcingTerm; /* flag set if using either ETACHOICE1 or
				      ETACHOICE2                               */
  realtype kin_mxnewtstep;     /* maximum allowable scaled step length         */
  realtype kin_sqrt_relfunc;   /* relative error bound for func(u)             */
  realtype kin_stepl;          /* scaled length of current step                */
  realtype kin_stepmul;        /* step scaling factor                          */
  realtype kin_eps;            /* current value of eps                         */
  realtype kin_eta;            /* current value of eta                         */
  realtype kin_eta_gamma;      /* gamma value used in eta calculation
				  (choice #2)                                  */
  realtype kin_eta_alpha;      /* alpha value used in eta calculation
			          (choice #2)                                  */
  booleantype kin_noPrecInit;  /* flag controlling whether or not the KINSol
				  routine makes an initial call to the
				  preconditioner setup routine (pset)          */
  realtype kin_pthrsh;         /* threshold value for calling preconditioner   */

  /* counters */

  long int  kin_nni;           /* number of nonlinear iterations               */
  long int  kin_nfe;           /* number of calls made to func routine         */
  long int  kin_nnilpre;       /* value of nni counter when the preconditioner
				  was last called                              */
  long int  kin_nbcf;          /* number of times the beta-condition could not 
                                  be met in KINLineSearch                      */
  long int  kin_nbktrk;        /* number of backtracks performed by
				  KINLineSearch                                */
  long int  kin_ncscmx;        /* number of consecutive steps of size
                                  mxnewtstep taken                             */

  /* vectors */

  N_Vector kin_uu;          /* solution vector/current iterate (initially
			       contains initial guess, but holds approximate
			       solution upon completion if no errors occurred) */
  N_Vector kin_unew;        /* next iterate (unew = uu+pp)                     */
  N_Vector kin_fval;        /* vector containing result of nonlinear system
			       function evaluated at a given iterate
			       (fval = func(uu))                               */
  N_Vector kin_uscale;      /* iterate scaling vector                          */
  N_Vector kin_fscale;      /* fval scaling vector                             */
  N_Vector kin_pp;          /* incremental change vector (pp = unew-uu)        */
  N_Vector kin_constraints; /* constraints vector                              */ 
  N_Vector kin_vtemp1;      /* scratch vector #1                               */
  N_Vector kin_vtemp2;      /* scratch vector #2                               */

  /* space requirements for vector storage */ 

  long int kin_lrw1;        /* number of realtype-sized memory blocks needed
			       for a single N_Vector                           */ 
  long int kin_liw1;        /* number of int-sized memory blocks needed for
			       a single N_Vecotr                               */ 
  long int kin_lrw;         /* total number of realtype-sized memory blocks
			       needed for all KINSOL work vectors              */
  long int kin_liw;         /* total number of int-sized memory blocks needed
			       for all KINSOL work vectors                     */

  /* linear solver data */
 
  /* function prototypes (pointers) */

  int (*kin_linit)(struct KINMemRec *kin_mem);

  int (*kin_lsetup)(struct KINMemRec *kin_mem);

  int (*kin_lsolve)(struct KINMemRec *kin_mem, N_Vector xx, N_Vector bb, 
                    realtype *res_norm );

  int (*kin_lfree)(struct KINMemRec *kin_mem);

  void *kin_lmem;        /* pointer to linear solver memory block              */

  realtype kin_fnorm;    /* value of L2-norm of fscale*fval                    */
  realtype kin_f1norm;   /* f1norm = 0.5*(fnorm)^2                             */
  realtype kin_res_norm; /* value of L2-norm of residual (set by linear
			    solver)                                            */
  realtype kin_sfdotJp;  /* value of scaled func(u) vector (fscale*fval)
			    dotted with scaled J(u)*pp vector                  */
  realtype kin_sJpnorm;  /* value of L2-norm of fscale*(J(u)*pp)               */
  
/*
 * -----------------------------------------------------------------
 * Note: The KINLineSearch subroutine scales the values of the
 * variables sfdotJp and sJpnorm by a factor rl (lambda) that is
 * chosen by the line search algorithm such that the sclaed Newton
 * step satisfies the following conditions:
 *
 *  F(u_k+1) <= F(u_k) + alpha*(F(u_k)^T * J(u_k))*p*rl
 *
 *  F(u_k+1) >= F(u_k) + beta*(F(u_k)^T * J(u_k))*p*rl
 *
 * where alpha = 1.0e-4, beta = 0.9, u_k+1 = u_k + rl*p,
 * 0 < rl <= 1, J denotes the system Jacobian, and F represents
 * the nonliner system function.
 * -----------------------------------------------------------------
 */

  booleantype kin_MallocDone; /* flag indicating if KINMalloc has been
				 called yet                                    */

  /* message files */
  
  FILE *kin_errfp;  /* where KINSol error/warning messages are sent            */
  FILE *kin_infofp; /* where KINSol info messages are sent                     */

  NV_Spec kin_nvspec; /* pointer to vector specification structure             */

} *KINMem;

#endif

#ifdef __cplusplus
}
#endif
