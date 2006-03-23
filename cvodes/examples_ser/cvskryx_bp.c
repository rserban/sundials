/*
 * -----------------------------------------------------------------
 * $Revision: 1.6 $
 * $Date: 2006-03-23 01:21:42 $
 * -----------------------------------------------------------------
 * Programmer(s): Scott D. Cohen, Alan C. Hindmarsh and
 *                Radu Serban @LLNL
 * -----------------------------------------------------------------
 * Example problem:
 *
 * An ODE system is generated from the following 2-species diurnal
 * kinetics advection-diffusion PDE system in 2 space dimensions:
 *
 * dc(i)/dt = Kh*(d/dx)^2 c(i) + V*dc(i)/dx + (d/dy)(Kv(y)*dc(i)/dy)
 *                 + Ri(c1,c2,t)      for i = 1,2,   where
 *   R1(c1,c2,t) = -q1*c1*c3 - q2*c1*c2 + 2*q3(t)*c3 + q4(t)*c2 ,
 *   R2(c1,c2,t) =  q1*c1*c3 - q2*c1*c2 - q4(t)*c2 ,
 *   Kv(y) = Kv0*exp(y/5) ,
 * Kh, V, Kv0, q1, q2, and c3 are constants, and q3(t) and q4(t)
 * vary diurnally. The problem is posed on the square
 *   0 <= x <= 20,    30 <= y <= 50   (all in km),
 * with homogeneous Neumann boundary conditions, and for time t in
 *   0 <= t <= 86400 sec (1 day).
 * The PDE system is treated by central differences on a uniform
 * 10 x 10 mesh, with simple polynomial initial profiles.
 * The problem is solved with CVODES, with the BDF/GMRES
 * method (i.e. using the CVSPGMR linear solver) and a banded
 * preconditioner, generated by difference quotients, using the
 * module CVBANDPRE. The problem is solved with left and right
 * preconditioning.
 * -----------------------------------------------------------------
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cvodes.h"         /* main integrator header file */
#include "nvector_serial.h" /* serial N_Vector types, fct. and macros */
#include "cvodes_spgmr.h"   /* prototypes & constants for CVSPGMR solver */
#include "cvodes_bandpre.h" /* prototypes & consts. for CVBANDPRE module */
#include "sundials_types.h" /* definition of realtype */
#include "sundials_math.h"  /* contains the macros ABS, SQR, and EXP */

/* Problem Constants */

#define ZERO RCONST(0.0)
#define ONE  RCONST(1.0)
#define TWO  RCONST(2.0)

#define NUM_SPECIES  2                 /* number of species         */
#define KH           RCONST(4.0e-6)    /* horizontal diffusivity Kh */
#define VEL          RCONST(0.001)     /* advection velocity V      */
#define KV0          RCONST(1.0e-8)    /* coefficient in Kv(y)      */
#define Q1           RCONST(1.63e-16)  /* coefficients q1, q2, c3   */ 
#define Q2           RCONST(4.66e-16)
#define C3           RCONST(3.7e16)
#define A3           RCONST(22.62)     /* coefficient in expression for q3(t) */
#define A4           RCONST(7.601)     /* coefficient in expression for q4(t) */
#define C1_SCALE     RCONST(1.0e6)     /* coefficients in initial profiles    */
#define C2_SCALE     RCONST(1.0e12)

#define T0           ZERO                /* initial time */
#define NOUT         12                  /* number of output times */
#define TWOHR        RCONST(7200.0)      /* number of seconds in two hours  */
#define HALFDAY      RCONST(4.32e4)      /* number of seconds in a half day */
#define PI       RCONST(3.1415926535898) /* pi */ 

#define XMIN         ZERO                /* grid boundaries in x  */
#define XMAX         RCONST(20.0)           
#define YMIN         RCONST(30.0)        /* grid boundaries in y  */
#define YMAX         RCONST(50.0)
#define XMID         RCONST(10.0)        /* grid midpoints in x,y */          
#define YMID         RCONST(40.0)

#define MX           10             /* MX = number of x mesh points */
#define MY           10             /* MY = number of y mesh points */
#define NSMX         20             /* NSMX = NUM_SPECIES*MX */
#define MM           (MX*MY)        /* MM = MX*MY */

/* CVodeMalloc Constants */

#define RTOL    RCONST(1.0e-5)   /* scalar relative tolerance */
#define FLOOR   RCONST(100.0)    /* value of C1 or C2 at which tolerances */
                                 /* change from relative to absolute      */
#define ATOL    (RTOL*FLOOR)     /* scalar absolute tolerance */
#define NEQ     (NUM_SPECIES*MM) /* NEQ = number of equations */

/* User-defined vector and matrix accessor macros: IJKth, IJth */

/* IJKth is defined in order to isolate the translation from the
   mathematical 3-dimensional structure of the dependent variable vector
   to the underlying 1-dimensional storage. IJth is defined in order to
   write code which indexes into small dense matrices with a (row,column)
   pair, where 1 <= row, column <= NUM_SPECIES.   
   
   IJKth(vdata,i,j,k) references the element in the vdata array for
   species i at mesh point (j,k), where 1 <= i <= NUM_SPECIES,
   0 <= j <= MX-1, 0 <= k <= MY-1. The vdata array is obtained via
   the macro call vdata = NV_DATA_S(v), where v is an N_Vector. 
   For each mesh point (j,k), the elements for species i and i+1 are
   contiguous within vdata.

   IJth(a,i,j) references the (i,j)th entry of the small matrix realtype **a,
   where 1 <= i,j <= NUM_SPECIES. The small matrix routines in dense.h
   work with matrices stored by column in a 2-dimensional array. In C,
   arrays are indexed starting at 0, not 1. */

#define IJKth(vdata,i,j,k) (vdata[i-1 + (j)*NUM_SPECIES + (k)*NSMX])
#define IJth(a,i,j)        (a[j-1][i-1])

/* Type : UserData 
   contains preconditioner blocks, pivot arrays, and problem constants */

typedef struct {
  realtype q4, om, dx, dy, hdco, haco, vdco;
} *UserData;

/* Private Helper Functions */

static void InitUserData(UserData data);
static void SetInitialProfiles(N_Vector u, realtype dx, realtype dy);
static void PrintIntro(int mu, int ml);
static void PrintOutput(void *cvode_mem, N_Vector u, realtype t);
static void PrintFinalStats(void *cvode_mem, void *bpdata);

/* Private function to check function return values */
static int check_flag(void *flagvalue, char *funcname, int opt);

/* Function Called by the Solver */

static int f(realtype t, N_Vector u, N_Vector udot, void *f_data);

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

int main()
{
  realtype abstol, reltol, t, tout;
  N_Vector u;
  UserData data;
  void *bpdata;
  void *cvode_mem;
  int flag, ml, mu, iout, jpre;

  u = NULL;
  data = NULL;
  bpdata = cvode_mem = NULL;

  /* Allocate and initialize u, and set problem data and tolerances */ 
  u = N_VNew_Serial(NEQ);
  if(check_flag((void *)u, "N_VNew_Serial", 0)) return(1);
  data = (UserData) malloc(sizeof *data);
  if(check_flag((void *)data, "malloc", 2)) return(1);
  InitUserData(data);
  SetInitialProfiles(u, data->dx, data->dy);
  abstol = ATOL; 
  reltol = RTOL;

  /* Call CvodeCreate to create the solver memory 

     CV_BDF     specifies the Backward Differentiation Formula
     CV_NEWTON  specifies a Newton iteration

     A pointer to the integrator memory is returned and stored in cvode_mem. */
  cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);
  if(check_flag((void *)cvode_mem, "CVodeCreate", 0)) return(1);

  /* Set the pointer to user-defined data */
  flag = CVodeSetFdata(cvode_mem, data);
  if(check_flag(&flag, "CVodeSetFdata", 1)) return(1);

  /* Call CVodeMalloc to initialize the integrator memory: 
     f       is the user's right hand side function in u'=f(t,u)
     T0      is the initial time
     u       is the initial dependent variable vector
     CV_SS   specifies scalar relative and absolute tolerances
     reltol  is the relative tolerance
     &abstol is a  pointer to the scalar absolutetolerance      */
  flag = CVodeMalloc(cvode_mem, f, T0, u, CV_SS, reltol, &abstol);
  if(check_flag(&flag, "CVodeMalloc", 1)) return(1);

  /* Call CVBandPreAlloc to initialize band preconditioner */
  ml = mu = 2;
  bpdata = CVBandPrecAlloc (cvode_mem, NEQ, mu, ml);
  if(check_flag((void *)bpdata, "CVBandPrecAlloc", 0)) return(1);

  /* Call CVBPSpgmr to specify the linear solver CVSPGMR 
     with left preconditioning and the maximum Krylov dimension maxl */
  flag = CVBPSpgmr(cvode_mem, PREC_LEFT, 0, bpdata);
  if(check_flag(&flag, "CVBPSpgmr", 1)) return(1);

  PrintIntro(mu, ml);

  /* Loop over jpre (= PREC_LEFT, PREC_RIGHT), and solve the problem */

  for (jpre = PREC_LEFT; jpre <= PREC_RIGHT; jpre++) {
    
    /* On second run, re-initialize u, the solver, and CVSPGMR */
    
    if (jpre == PREC_RIGHT) {
      
      SetInitialProfiles(u, data->dx, data->dy);
      
      flag = CVodeReInit(cvode_mem, f, T0, u, CV_SS, reltol, &abstol);
      if(check_flag(&flag, "CVodeReInit", 1)) return(1);

      flag = CVSpilsSetPrecType(cvode_mem, PREC_RIGHT);
      check_flag(&flag, "CVSpilsSetPrecType", 1);
      
      printf("\n\n-------------------------------------------------------");
      printf("------------\n");
    }
    
    printf("\n\nPreconditioner type is:  jpre = %s\n\n",
           (jpre == PREC_LEFT) ? "PREC_LEFT" : "PREC_RIGHT");
    
    /* In loop over output points, call CVode, print results, test for error */
    
    for (iout = 1, tout = TWOHR; iout <= NOUT; iout++, tout += TWOHR) {
      flag = CVode(cvode_mem, tout, u, &t, CV_NORMAL);
      check_flag(&flag, "CVode", 1);
      PrintOutput(cvode_mem, u, t);
      if (flag != CV_SUCCESS) {
        break;
      }
    }
    
    /* Print final statistics */
    
    PrintFinalStats(cvode_mem, bpdata);
    
  } /* End of jpre loop */

  /* Free memory */
  N_VDestroy_Serial(u);
  free(data);
  CVBandPrecFree(&bpdata);
  CVodeFree(&cvode_mem);

  return(0);
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

/* Load problem constants in data */

static void InitUserData(UserData data)
{
  data->om = PI/HALFDAY;
  data->dx = (XMAX-XMIN)/(MX-1);
  data->dy = (YMAX-YMIN)/(MY-1);
  data->hdco = KH/SQR(data->dx);
  data->haco = VEL/(TWO*data->dx);
  data->vdco = (ONE/SQR(data->dy))*KV0;
}

/* Set initial conditions in u */

static void SetInitialProfiles(N_Vector u, realtype dx, realtype dy)
{
  int jx, jy;
  realtype x, y, cx, cy;
  realtype *udata;

  /* Set pointer to data array in vector u. */

  udata = NV_DATA_S(u);

  /* Load initial profiles of c1 and c2 into u vector */

  for (jy = 0; jy < MY; jy++) {
    y = YMIN + jy*dy;
    cy = SQR(RCONST(0.1)*(y - YMID));
    cy = ONE - cy + RCONST(0.5)*SQR(cy);
    for (jx = 0; jx < MX; jx++) {
      x = XMIN + jx*dx;
      cx = SQR(RCONST(0.1)*(x - XMID));
      cx = ONE - cx + RCONST(0.5)*SQR(cx);
      IJKth(udata,1,jx,jy) = C1_SCALE*cx*cy; 
      IJKth(udata,2,jx,jy) = C2_SCALE*cx*cy;
    }
  }
}

static void PrintIntro(int mu, int ml)
{
  printf("2-species diurnal advection-diffusion problem, %d by %d mesh\n",
         MX, MY);
  printf("SPGMR solver; band preconditioner; mu = %d, ml = %d\n\n",
         mu, ml);

  return;
}

/* Print current t, step count, order, stepsize, and sampled c1,c2 values */

static void PrintOutput(void *cvode_mem, N_Vector u,realtype t)
{
  long int nst;
  int qu, flag;
  realtype hu, *udata;
  int mxh = MX/2 - 1, myh = MY/2 - 1, mx1 = MX - 1, my1 = MY - 1;

  udata = NV_DATA_S(u);

  flag = CVodeGetNumSteps(cvode_mem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetLastOrder(cvode_mem, &qu);
  check_flag(&flag, "CVodeGetLastOrder", 1);
  flag = CVodeGetLastStep(cvode_mem, &hu);
  check_flag(&flag, "CVodeGetLastStep", 1);

#if defined(SUNDIALS_EXTENDED_PRECISION)
  printf("t = %.2Le   no. steps = %ld   order = %d   stepsize = %.2Le\n",
         t, nst, qu, hu);
  printf("c1 (bot.left/middle/top rt.) = %12.3Le  %12.3Le  %12.3Le\n",
         IJKth(udata,1,0,0), IJKth(udata,1,mxh,myh), IJKth(udata,1,mx1,my1));
  printf("c2 (bot.left/middle/top rt.) = %12.3Le  %12.3Le  %12.3Le\n\n",
         IJKth(udata,2,0,0), IJKth(udata,2,mxh,myh), IJKth(udata,2,mx1,my1));
#elif defined(SUNDIALS_DOUBLE_PRECISION)
  printf("t = %.2le   no. steps = %ld   order = %d   stepsize = %.2le\n",
         t, nst, qu, hu);
  printf("c1 (bot.left/middle/top rt.) = %12.3le  %12.3le  %12.3le\n",
         IJKth(udata,1,0,0), IJKth(udata,1,mxh,myh), IJKth(udata,1,mx1,my1));
  printf("c2 (bot.left/middle/top rt.) = %12.3le  %12.3le  %12.3le\n\n",
         IJKth(udata,2,0,0), IJKth(udata,2,mxh,myh), IJKth(udata,2,mx1,my1));
#else
  printf("t = %.2e   no. steps = %ld   order = %d   stepsize = %.2e\n",
         t, nst, qu, hu);
  printf("c1 (bot.left/middle/top rt.) = %12.3e  %12.3e  %12.3e\n",
         IJKth(udata,1,0,0), IJKth(udata,1,mxh,myh), IJKth(udata,1,mx1,my1));
  printf("c2 (bot.left/middle/top rt.) = %12.3e  %12.3e  %12.3e\n\n",
         IJKth(udata,2,0,0), IJKth(udata,2,mxh,myh), IJKth(udata,2,mx1,my1));
#endif
}

/* Get and print final statistics */

static void PrintFinalStats(void *cvode_mem, void *bpdata)
{
  long int lenrw, leniw ;
  long int lenrwLS, leniwLS;
  long int lenrwBP, leniwBP;
  long int nst, nfe, nsetups, nni, ncfn, netf;
  long int nli, npe, nps, ncfl, nfeLS;
  long int nfeBP;
  int flag;

  flag = CVodeGetWorkSpace(cvode_mem, &lenrw, &leniw);
  check_flag(&flag, "CVodeGetWorkSpace", 1);
  flag = CVodeGetNumSteps(cvode_mem, &nst);
  check_flag(&flag, "CVodeGetNumSteps", 1);
  flag = CVodeGetNumRhsEvals(cvode_mem, &nfe);
  check_flag(&flag, "CVodeGetNumRhsEvals", 1);
  flag = CVodeGetNumLinSolvSetups(cvode_mem, &nsetups);
  check_flag(&flag, "CVodeGetNumLinSolvSetups", 1);
  flag = CVodeGetNumErrTestFails(cvode_mem, &netf);
  check_flag(&flag, "CVodeGetNumErrTestFails", 1);
  flag = CVodeGetNumNonlinSolvIters(cvode_mem, &nni);
  check_flag(&flag, "CVodeGetNumNonlinSolvIters", 1);
  flag = CVodeGetNumNonlinSolvConvFails(cvode_mem, &ncfn);
  check_flag(&flag, "CVodeGetNumNonlinSolvConvFails", 1);

  flag = CVSpilsGetWorkSpace(cvode_mem, &lenrwLS, &leniwLS);
  check_flag(&flag, "CVSpilsGetWorkSpace", 1);
  flag = CVSpilsGetNumLinIters(cvode_mem, &nli);
  check_flag(&flag, "CVSpilsGetNumLinIters", 1);
  flag = CVSpilsGetNumPrecEvals(cvode_mem, &npe);
  check_flag(&flag, "CVSpilsGetNumPrecEvals", 1);
  flag = CVSpilsGetNumPrecSolves(cvode_mem, &nps);
  check_flag(&flag, "CVSpilsGetNumPrecSolves", 1);
  flag = CVSpilsGetNumConvFails(cvode_mem, &ncfl);
  check_flag(&flag, "CVSpilsGetNumConvFails", 1);
  flag = CVSpilsGetNumRhsEvals(cvode_mem, &nfeLS);
  check_flag(&flag, "CVSpilsGetNumRhsEvals", 1);

  flag = CVBandPrecGetWorkSpace(bpdata, &lenrwBP, &leniwBP);
  check_flag(&flag, "CVBandPrecGetWorkSpace", 1);
  flag = CVBandPrecGetNumRhsEvals(bpdata, &nfeBP);
  check_flag(&flag, "CVBandPrecGetNumRhsEvals", 1);

  printf("\nFinal Statistics.. \n\n");
  printf("lenrw   = %5ld     leniw   = %5ld\n", lenrw, leniw);
  printf("lenrwls = %5ld     leniwls = %5ld\n", lenrwLS, leniwLS);
  printf("lenrwbp = %5ld     leniwbp = %5ld\n", lenrwBP, leniwBP);
  printf("nst     = %5ld\n"                  , nst);
  printf("nfe     = %5ld     nfetot  = %5ld\n"  , nfe, nfe+nfeLS+nfeBP);
  printf("nfeLS   = %5ld     nfeBP   = %5ld\n"  , nfeLS, nfeBP);
  printf("nni     = %5ld     nli     = %5ld\n"  , nni, nli);
  printf("nsetups = %5ld     netf    = %5ld\n"  , nsetups, netf);
  printf("npe     = %5ld     nps     = %5ld\n"  , npe, nps);
  printf("ncfn    = %5ld     ncfl    = %5ld\n\n", ncfn, ncfl);
}

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag >= 0
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, char *funcname, int opt)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if flag < 0 */
  else if (opt == 1) {
    errflag = (int *) flagvalue;
    if (*errflag < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with flag = %d\n\n",
              funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);
}

/*
 *-------------------------------
 * Function called by the solver
 *-------------------------------
 */

/* f routine. Compute f(t,u). */

static int f(realtype t, N_Vector u, N_Vector udot,void *f_data)
{
  realtype q3, c1, c2, c1dn, c2dn, c1up, c2up, c1lt, c2lt;
  realtype c1rt, c2rt, cydn, cyup, hord1, hord2, horad1, horad2;
  realtype qq1, qq2, qq3, qq4, rkin1, rkin2, s, vertd1, vertd2, ydn, yup;
  realtype q4coef, dely, verdco, hordco, horaco;
  realtype *udata, *dudata;
  int idn, iup, ileft, iright, jx, jy;
  UserData data;

  data = (UserData) f_data;
  udata = NV_DATA_S(u);
  dudata = NV_DATA_S(udot);

  /* Set diurnal rate coefficients. */

  s = sin(data->om*t);
  if (s > ZERO) {
    q3 = EXP(-A3/s);
    data->q4 = EXP(-A4/s);
  } else {
    q3 = ZERO;
    data->q4 = ZERO;
  }

  /* Make local copies of problem variables, for efficiency. */

  q4coef = data->q4;
  dely = data->dy;
  verdco = data->vdco;
  hordco  = data->hdco;
  horaco  = data->haco;

  /* Loop over all grid points. */

  for (jy = 0; jy < MY; jy++) {

    /* Set vertical diffusion coefficients at jy +- 1/2 */

    ydn = YMIN + (jy - RCONST(0.5))*dely;
    yup = ydn + dely;
    cydn = verdco*EXP(RCONST(0.2)*ydn);
    cyup = verdco*EXP(RCONST(0.2)*yup);
    idn = (jy == 0) ? 1 : -1;
    iup = (jy == MY-1) ? -1 : 1;
    for (jx = 0; jx < MX; jx++) {

      /* Extract c1 and c2, and set kinetic rate terms. */

      c1 = IJKth(udata,1,jx,jy); 
      c2 = IJKth(udata,2,jx,jy);
      qq1 = Q1*c1*C3;
      qq2 = Q2*c1*c2;
      qq3 = q3*C3;
      qq4 = q4coef*c2;
      rkin1 = -qq1 - qq2 + TWO*qq3 + qq4;
      rkin2 = qq1 - qq2 - qq4;

      /* Set vertical diffusion terms. */

      c1dn = IJKth(udata,1,jx,jy+idn);
      c2dn = IJKth(udata,2,jx,jy+idn);
      c1up = IJKth(udata,1,jx,jy+iup);
      c2up = IJKth(udata,2,jx,jy+iup);
      vertd1 = cyup*(c1up - c1) - cydn*(c1 - c1dn);
      vertd2 = cyup*(c2up - c2) - cydn*(c2 - c2dn);

      /* Set horizontal diffusion and advection terms. */

      ileft = (jx == 0) ? 1 : -1;
      iright =(jx == MX-1) ? -1 : 1;
      c1lt = IJKth(udata,1,jx+ileft,jy); 
      c2lt = IJKth(udata,2,jx+ileft,jy);
      c1rt = IJKth(udata,1,jx+iright,jy);
      c2rt = IJKth(udata,2,jx+iright,jy);
      hord1 = hordco*(c1rt - TWO*c1 + c1lt);
      hord2 = hordco*(c2rt - TWO*c2 + c2lt);
      horad1 = horaco*(c1rt - c1lt);
      horad2 = horaco*(c2rt - c2lt);

      /* Load all terms into udot. */

      IJKth(dudata, 1, jx, jy) = vertd1 + hord1 + horad1 + rkin1; 
      IJKth(dudata, 2, jx, jy) = vertd2 + hord2 + horad2 + rkin2;
    }
  }

  return(0);
}
