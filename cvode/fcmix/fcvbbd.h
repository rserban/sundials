/*
 * -----------------------------------------------------------------
 * $Revision: 1.24 $
 * $Date: 2005-10-11 16:02:39 $
 * ----------------------------------------------------------------- 
 * Programmer(s): Alan Hindmarsh, Radu Serban and
 *                Aaron Collier @ LLNL
 * -----------------------------------------------------------------
 * Copyright (c) 2002, The Regents of the University of California.
 * Produced at the Lawrence Livermore National Laboratory.
 * All rights reserved.
 * For details, see sundials/cvode/LICENSE.
 * -----------------------------------------------------------------
 * This is the Fortran interface include file for the BBD
 * preconditioner (CVBBDPRE)
 * -----------------------------------------------------------------
 */

/*
 * ==============================================================================
 *
 *                   FCVBBD Interface Package
 * 
 * The FCVBBD Interface Package is a package of C functions which,
 * together with the FCVODE Interface Package, support the use of the
 * CVODE solver (parallel MPI version) with the CVBBDPRE preconditioner module,
 * for the solution of ODE systems in a mixed Fortran/C setting.  The
 * combination of CVODE and CVBBDPRE solves systems dy/dt = f(t,y) with
 * the SPGMR (scaled preconditioned GMRES), SPTFQMR (scaled preconditioned TFQMR),
 * or SPBCG (scaled preconditioned Bi-CGSTAB) method for the linear systems that
 * arise, and with a preconditioner that is block-diagonal with banded blocks.
 * While CVODE and CVBBDPRE are written in C, it is assumed here that the user's
 * calling program and user-supplied problem-defining routines are written in
 * Fortran.
 * 
 * The user-callable functions in this package, with the corresponding
 * CVODE and CVBBDPRE functions, are as follows: 
 *   FCVBBDININT   interfaces to CVBBDPrecAlloc
 *   FCVBBDSPTFQMR interfaces to CVBBDSptfqmr
 *   FCVBBDSPBCG   interfaces to CVBBDSpbcg
 *   FCVBBDPSGMR   interfaces to CVBBDSpgmr
 *   FCVBBDREINIT  interfaces to CVBBDPrecReInit
 *   FCVBBDOPT     accesses optional outputs
 *   FCVBBDFREE    interfaces to CVBBDPrecFree
 * 
 * In addition to the Fortran right-hand side function FCVFUN, the
 * user-supplied functions used by this package, are listed below,
 * each with the corresponding interface function which calls it (and its
 * type within CVBBDPRE or CVODE):
 *   FCVLOCFN  is called by the interface function FCVgloc of type CVLocalFn
 *   FCVCOMMF  is called by the interface function FCVcfn of type CVCommFn
 *   FCVJTIMES (optional) is called by the interface function FCVJtimes of 
 *             type CVSpilsJtimesFn
 * (The names of all user-supplied routines here are fixed, in order to
 * maximize portability for the resulting mixed-language program.)
 * 
 * Important note on portability.
 * In this package, the names of the interface functions, and the names of
 * the Fortran user routines called by them, appear as dummy names
 * which are mapped to actual values by a series of definitions in the
 * header file fcvbbd.h.
 * 
 * ==============================================================================
 * 
 *               Usage of the FCVODE/FCVBBD Interface Packages
 * 
 * The usage of the combined interface packages FCVODE and FCVBBD requires
 * calls to seven to twelve interface functions, and three or four user-supplied
 * routines which define the problem to be solved and indirectly define
 * the preconditioner.  These function calls and user routines are
 * summarized separately below.
 * 
 * Some details are omitted, and the user is referred to the CVODE user document 
 * for more complete information.
 * 
 * (1) User-supplied right-hand side routine: FCVFUN
 * The user must in all cases supply the following Fortran routine
 *       SUBROUTINE FCVFUN (T, Y, YDOT, IPAR, RPAR)
 *       DIMENSION Y(*), YDOT(*), IPAR(*), RPAR(*)
 * It must set the YDOT array to f(t,y), the right-hand side of the ODE
 * system, as function of T = t and the array Y = y.  Here Y and YDOT
 * are distributed vectors.
 * 
 * (2) User-supplied routines to define preconditoner: FCVLOCFN and FCVCOMMF
 * 
 * The routines in the CVBBDPRE module provide a preconditioner matrix
 * for CVODE that is block-diagonal with banded blocks.  The blocking
 * corresponds to the distribution of the dependent variable vector y
 * among the processors.  Each preconditioner block is generated from the
 * Jacobian of the local part (on the current processor) of a given
 * function g(t,y) approximating f(t,y).  The blocks are generated by a
 * difference quotient scheme on each processor independently, utilizing
 * an assumed banded structure with given half-bandwidths.  A separate
 * pair of half-bandwidths defines the band matrix retained.
 * 
 * (2.1) Local approximate function FCVLOCFN.
 * The user must supply a subroutine of the form
 *       SUBROUTINE FCVLOCFN (NLOC, T, YLOC, GLOC, IPAR, RPAR)
 *       DIMENSION YLOC(*), GLOC(*), IPAR(*), RPAR(*)
 * to compute the function g(t,y) which approximates the right-hand side
 * function f(t,y).  This function is to be computed locally, i.e. without 
 * interprocess communication.  (The case where g is mathematically
 * identical to f is allowed.)  It takes as input the local vector length
 * NLOC, the independent variable value T = t, and the local realtype
 * dependent variable array YLOC.  It is to compute the local part of
 * g(t,y) and store this in the realtype array GLOC.
 * 
 * (2.2) Communication function FCVCOMMF.
 * The user must also supply a subroutine of the form
 *       SUBROUTINE FCVCOMMF (NLOC, T, YLOC, IPAR, RPAR)
 *       DIMENSION YLOC(*), IPAR(*), RPAR(*)
 * which is to perform all interprocess communication necessary to
 * evaluate the approximate right-hand side function g described above.
 * This function takes as input the local vector length NLOC, the
 * independent variable value T = t, and the local real dependent
 * variable array YLOC.  It is expected to save communicated data in 
 * work space defined by the user, and made available to CVLOCFN.
 * Each call to the FCVCOMMF is preceded by a call to FCVFUN with the same
 * (t,y) arguments.  Thus FCVCOMMF can omit any communications done by 
 * FCVFUN if relevant to the evaluation of g.
 * 
 * (3) Optional user-supplied Jacobian-vector product routine: FCVJTIMES
 * As an option, the user may supply a routine that computes the product
 * of the system Jacobian J = df/dy and a given vector v.  If supplied, it
 * must have the following form:
 *       SUBROUTINE FCVJTIMES (V, FJV, T, Y, FY, EWT, IPAR, RPAR, WORK, IER)
 *       DIMENSION V(*), FJV(*), Y(*), FY(*), EWT(*), IPAR(*), RPAR(*), WORK(*)
 * Typically this routine will use only NEQ, T, Y, V, and FJV.  It must
 * compute the product vector Jv, where the vector v is stored in V, and store
 * the product in FJV.  On return, set IER = 0 if FCVJTIMES was successful,
 * and nonzero otherwise.
 * 
 * (4) Initialization:  FNVINITP, FCVMALLOC, FCVBBDINIT.
 * 
 * (4.1) To initialize the parallel vector specification, the user must make 
 * the following call:
 *        CALL FNVINITP (NLOCAL, NGLOBAL, IER)
 * The arguments are:
 * NLOCAL  = local size of vectors on this processor
 * NGLOBAL = the system size, and the global size of vectors (the sum 
 *           of all values of NLOCAL)
 * IER     = return completion flag. Values are 0 = success, -1 = failure.
 * 
 * Note: If MPI was initialized by the user, the communicator must be
 * set to MPI_COMM_WORLD.  If not, this routine initializes MPI and sets
 * the communicator equal to MPI_COMM_WORLD.
 * 
 * (4.2) To set various problem and solution parameters and allocate
 * internal memory for CVODE, make the following call:
 *       CALL FCVMALLOC(T0, Y0, METH, ITMETH, IATOL, RTOL, ATOL,
 *      1               IOUT, ROUT, IPAR, RPAR, IER)
 * The arguments are:
 * T0     = initial value of t
 * Y0     = array of initial conditions
 * METH   = basic integration method: 1 = Adams (nonstiff), 2 = BDF (stiff)
 * ITMETH = nonlinear iteration method: 1 = functional iteration, 2 = Newton iter.
 * IATOL  = type for absolute tolerance ATOL: 1 = scalar, 2 = array
 * RTOL   = relative tolerance (scalar)
 * ATOL   = absolute tolerance (scalar or array)
 * IOUT   = array of length 21 for integer optional outputs
 *          (declare as INTEGER*4 or INTEGER*8 according to C type long int)
 * ROUT   = array of length 6 for real optional outputs
 * IPAR   = array with user integer data
 *          (declare as INTEGER*4 or INTEGER*8 according to C type long int)
 * RPAR   = array with user real data
 * IER    = return completion flag.  Values are 0 = success, and -1 = failure.
 *          See printed message for details in case of failure.
 * 
 * (4.3) To allocate memory and initialize data associated with the CVBBDPRE
 * preconditioner, make the following call:
 *       CALL FCVBBDINIT(NLOCAL, MUDQ, MLDQ, MU, ML, DQRELY, IER)
 * 
 * The arguments are:
 * NLOCAL    = local size of vectors on this processor
 * MUDQ,MLDQ = upper and lower half-bandwidths to be used in the computation
 *             of the local Jacobian blocks by difference quotients.
 *             These may be smaller than the true half-bandwidths of the
 *             Jacobian of the local block of g, when smaller values may
 *             provide greater efficiency.
 * MU, ML    = upper and lower half-bandwidths of the band matrix that 
 *             is retained as an approximation of the local Jacobian block.
 *             These may be smaller than MUDQ and MLDQ.
 * DQRELY    = relative increment factor in y for difference quotients
 *             (optional). 0.0 indicates the default, sqrt(unit roundoff).
 * IER       = return completion flag: IER=0: success, IER<0: an error occurred
 *
 * (4.4A) To specify the SPGMR linear system solver and use it with the CVBBDPRE
 * preconditioner, make the following call:
 *       CALL FCVBBDSPGMR(IPRETYPE, IGSTYPE, MAXL, DELT, IER)
 * The arguments are:
 * IPRETYPE  = preconditioner type: 
 *             0 = none
 *             1 = left only
 *             2 = right only
 *             3 = both sides.
 * IGSTYPE   = Gram-schmidt process type: 1 = modified G-S, 1 = classical G-S.
 * MAXL      = maximum Krylov subspace dimension; 0 indicates default.
 * DELT      = linear convergence tolerance factor; 0.0 indicates default.
 * IER       = return completion flag: IER=0: success, IER<0: an error occurred
 *
 * (4.4B) To specify the SPBCG linear system solver and use it with the CVBBDPRE
 * preconditioner, make the following call:
 *       CALL FCVBBDSPBCG(IPRETYPE, MAXL, DELT, IER)
 * The arguments are:
 * IPRETYPE  = preconditioner type: 
 *             0 = none
 *             1 = left only
 *             2 = right only
 *             3 = both sides.
 * MAXL      = maximum Krylov subspace dimension; 0 indicates default.
 * DELT      = linear convergence tolerance factor; 0.0 indicates default.
 * IER       = return completion flag: IER=0: success, IER<0: an error occurred
 *
 * (4.4C) To specify the SPTFQMR linear system solver and use it with the CVBBDPRE
 * preconditioner, make the following call:
 *       CALL FCVBBDSPTFQMR(IPRETYPE, MAXL, DELT, IER)
 * The arguments are:
 * IPRETYPE  = preconditioner type: 
 *             0 = none
 *             1 = left only
 *             2 = right only
 *             3 = both sides.
 * MAXL      = maximum Krylov subspace dimension; 0 indicates default.
 * DELT      = linear convergence tolerance factor; 0.0 indicates default.
 * IER       = return completion flag: IER=0: success, IER<0: an error occurred
 *
 * (4.5A) To specify whether GMRES should use the supplied FCVJTIMES or the 
 * internal finite difference approximation, make the call
 *        CALL FCVSPGMRSETJAC(FLAG, IER)
 * where FLAG=0 for finite differences approximation or
 *       FLAG=1 to use the supplied routine FCVJTIMES
 * 
 * (4.5B) To specify whether Bi-CGSTAB should use the supplied FCVJTIMES or the 
 * internal finite difference approximation, make the call
 *        CALL FCVSPBCGSETJAC(FLAG, IER)
 * where FLAG=0 for finite differences approximation or
 *       FLAG=1 to use the supplied routine FCVJTIMES
 *
 * (4.5C) To specify whether TFQMR should use the supplied FCVJTIMES or the 
 * internal finite difference approximation, make the call
 *        CALL FCVSPTFQMRSETJAC(FLAG, IER)
 * where FLAG=0 for finite differences approximation or
 *       FLAG=1 to use the supplied routine FCVJTIMES
 *
 * (5) Re-initialization: FCVREINIT, FCVBBDREINIT
 * If a sequence of problems of the same size is being solved using the SPGMR, SPBCG,
 * SPTFQMR linear solver in combination with the CVBBDPRE preconditioner, then the
 * CVODE package can be reinitialized for the second and subsequent problems
 * so as to avoid further memory allocation.  First, in place of the call
 * to FCVMALLOC, make the following call:
 *       CALL FCVREINIT(T0, Y0, IATOL, RTOL, ATOL, IER)
 * The arguments have the same names and meanings as those of FCVMALLOC, except
 * that METH and ITMETH have been omitted from the argument list (being unchanged
 * for the new problem).  FCVREINIT performs the same initializations as
 * FCVMALLOC, but does no memory allocation, using instead the existing
 * internal memory created by the previous FCVMALLOC call.
 * Following the call to FCVREINIT, a call to FCVBBDINIT may or may not be needed.
 * If the input arguments are the same, no FCVBBDINIT call is needed.
 * If there is a change in input arguments, then make the call
 *      CALL FCVBBDREINIT(NLOCAL, MUDQ, MLDQ, DQRELY, IER)
 * This reinitializes the SP* linear solver, but without reallocating its memory.
 * The arguments of the have the same names and meanings as FCVBBDINIT.  
 * If the value of MU or ML is being changed, then a call to FCVBBDINIT must
 * be made.  If there is a change in any of the linear solver arguments, then
 * a call to FCVBBDSPGMR, FCVBBDSPBCG, or FCVBBDSPTFQMR must also be made;
 * in this case the linear solver memory is reallocated. 
 * 
 * (6) The integrator: FCVODE
 * Carrying out the integration is accomplished by making calls as follows:
 *       CALL FCVODE (TOUT, T, Y, ITASK, IER)
 * The arguments are:
 * TOUT  = next value of t at which a solution is desired (input)
 * T     = value of t reached by the solver on output
 * Y     = array containing the computed solution on output
 * ITASK = task indicator: 
 *         1 = normal mode (overshoot TOUT and interpolate)
 *         2 = one-step mode (return after each internal step taken)
 *         3 = normal mode with TSTOP check
 *         4 = one-step mode with TSTOP check
 * IER   = completion flag: 0 = success, 1 = TSTOP return, 2 = root return,
 *         negative values are various failure modes (see CVODE User Guide).
 * The current values of the optional outputs are available in IOUT and ROUT.
 * 
 * (7) Optional outputs: FCVBBDOPT
 * Optional outputs specific to the SP* solver are LRW, LIW, LFLG, NFEDQ, NJTV,
 * NPE, NPS, NLI, NCFL, stored in IOUT(13)...IOUT(21).
 * To obtain the optional outputs associated with the CVBBDPRE module, make
 * the following call:
 *       CALL FCVBBDOPT (LENRPW, LENIPW, NGE)
 * The arguments returned are:
 * LENRPW = length of real preconditioner work space, in realtype words.
 *          This size is local to the current processor.
 * LENIPW = length of integer preconditioner work space, in integer words.
 *          This size is local to the current processor.
 * NGE    = number of g(t,y) evaluations (calls to CVLOCFN) so far.
 * 
 * (8) Computing solution derivatives: FCVDKY
 * To obtain a derivative of the solution (optionally), of order up to
 * the current method order, make the following call:
 *       CALL FCVDKY (T, K, DKY)
 * The arguments are:
 * T   = value of t at which solution derivative is desired
 * K   = derivative order (0 .le. K .le. QU)
 * DKY = array containing computed K-th derivative of y on return
 * 
 * (9) Memory freeing: FCVBBDFREE and FCVFREE
 *   To the free the internal memory created by the calls to FNVINITP,
 * FCVMALLOC, and FCVBBDINIT, make the following calls, in this order:
 *       CALL FCVBBDFREE
 *       CALL FCVFREE
 *
 * ==============================================================================
 */

#ifndef _FCVBBD_H
#define _FCVBBD_H

#ifdef __cplusplus  /* wrapper to enable C++ usage */
extern "C" {
#endif

/* header files  */

#include "nvector.h"       /* definition of type N_Vector */
#include "sundialstypes.h" /* definition of type realtype */

/* Definitions of interface function names */

#if defined(F77_FUNC)

#define FCV_BBDINIT    F77_FUNC(fcvbbdinit, FCVBBDINIT)
#define FCV_BBDSPTFQMR F77_FUNC(fcvbbdsptfqmr, FCVBBDSPTFQMR)
#define FCV_BBDSPBCG   F77_FUNC(fcvbbdspbcg, FCVBBDSPBCG)
#define FCV_BBDSPGMR   F77_FUNC(fcvbbdspgmr, FCVBBDSPGMR)
#define FCV_BBDREINIT  F77_FUNC(fcvbbdreinit, FCVBBDREINIT)
#define FCV_BBDOPT     F77_FUNC(fcvbbdopt, FCVBBDOPT)
#define FCV_BBDFREE    F77_FUNC(fcvbbdfree, FCVBBDFREE)
#define FCV_GLOCFN     F77_FUNC(fcvglocfn, FCVGLOCFN)
#define FCV_COMMFN     F77_FUNC(fcvcommfn, FCVCOMMFN)

#elif defined(SUNDIALS_UNDERSCORE_NONE) && defined(SUNDIALS_CASE_LOWER)

#define FCV_BBDINIT    fcvbbdinit
#define FCV_BBDSPTFQMR fcvbbdsptfqmr
#define FCV_BBDSPBCG   fcvbbdspbcg
#define FCV_BBDSPGMR   fcvbbdspgmr
#define FCV_BBDREINIT  fcvbbdreinit
#define FCV_BBDOPT     fcvbbdopt
#define FCV_BBDFREE    fcvbbdfree
#define FCV_GLOCFN     fcvglocfn
#define FCV_COMMFN     fcvcommfn

#elif defined(SUNDIALS_UNDERSCORE_NONE) && defined(SUNDIALS_CASE_UPPER)

#define FCV_BBDINIT    FCVBBDINIT
#define FCV_BBDSPTFQMR FCVBBDSPTFQMR
#define FCV_BBDSPBCG   FCVBBDSPBCG
#define FCV_BBDSPGMR   FCVBBDSPGMR
#define FCV_BBDREINIT  FCVBBDREINIT
#define FCV_BBDOPT     FCVBBDOPT
#define FCV_BBDFREE    FCVBBDFREE
#define FCV_GLOCFN     FCVGLOCFN
#define FCV_COMMFN     FCVCOMMFN

#elif defined(SUNDIALS_UNDERSCORE_ONE) && defined(SUNDIALS_CASE_LOWER)

#define FCV_BBDINIT    fcvbbdinit_
#define FCV_BBDSPTFQMR fcvbbdsptfqmr_
#define FCV_BBDSPBCG   fcvbbdspbcg_
#define FCV_BBDSPGMR   fcvbbdspgmr_
#define FCV_BBDREINIT  fcvbbdreinit_
#define FCV_BBDOPT     fcvbbdopt_
#define FCV_BBDFREE    fcvbbdfree_
#define FCV_GLOCFN     fcvglocfn_
#define FCV_COMMFN     fcvcommfn_

#elif defined(SUNDIALS_UNDERSCORE_ONE) && defined(SUNDIALS_CASE_UPPER)

#define FCV_BBDINIT    FCVBBDINIT_
#define FCV_BBDSPTFQMR FCVBBDSPTFQMR_
#define FCV_BBDSPBCG   FCVBBDSPBCG_
#define FCV_BBDSPGMR   FCVBBDSPGMR_
#define FCV_BBDREINIT  FCVBBDREINIT_
#define FCV_BBDOPT     FCVBBDOPT_
#define FCV_BBDFREE    FCVBBDFREE_
#define FCV_GLOCFN     FCVGLOCFN_
#define FCV_COMMFN     FCVCOMMFN_

#elif defined(SUNDIALS_UNDERSCORE_TWO) && defined(SUNDIALS_CASE_LOWER)

#define FCV_BBDINIT    fcvbbdinit__
#define FCV_BBDSPTFQMR fcvbbdsptfqmr__
#define FCV_BBDSPBCG   fcvbbdspbcg__
#define FCV_BBDSPGMR   fcvbbdspgmr__
#define FCV_BBDREINIT  fcvbbdreinit__
#define FCV_BBDOPT     fcvbbdopt__
#define FCV_BBDFREE    fcvbbdfree__
#define FCV_GLOCFN     fcvglocfn__
#define FCV_COMMFN     fcvcommfn__

#elif defined(SUNDIALS_UNDERSCORE_TWO) && defined(SUNDIALS_CASE_UPPER)

#define FCV_BBDINIT    FCVBBDINIT__
#define FCV_BBDSPTFQMR FCVBBDSPTFQMR__
#define FCV_BBDSPBCG   FCVBBDSPBCG__
#define FCV_BBDSPGMR   FCVBBDSPGMR__
#define FCV_BBDREINIT  FCVBBDREINIT__
#define FCV_BBDOPT     FCVBBDOPT__
#define FCV_BBDFREE    FCVBBDFREE__
#define FCV_GLOCFN     FCVGLOCFN__
#define FCV_COMMFN     FCVCOMMFN__

#endif

/* Prototypes of exported functions */

void FCV_BBDINIT(long int *Nloc, long int *mudq, long int *mldq, 
                 long int *mu, long int *ml, realtype* dqrely, int *ier);
void FCV_BBDSPTFQMR(int *pretype, int *maxl, realtype *delt, int *ier);
void FCV_BBDSPBCG(int *pretype, int *maxl, realtype *delt, int *ier);
void FCV_BBDSPGMR(int *pretype, int *gstype, int *maxl, realtype *delt, int *ier);
void FCV_BBDREINIT(long int *Nloc, long int *mudq, long int *mldq, 
                   realtype* dqrely, int *ier);
void FCV_BBDOPT(long int *lenrpw, long int *lenipw, long int *nge);
void FCV_BBDFREE(void);

/* Prototypes: Functions Called by the CVBBDPRE Module */

void FCVgloc(long int Nloc, realtype t, N_Vector yloc, N_Vector gloc,
             void *f_data);

void FCVcfn(long int Nloc, realtype t, N_Vector y, void *f_data);


/* Declarations for global variables, shared among various routines */

void *CVBBD_Data;

#ifdef __cplusplus
}
#endif

#endif
