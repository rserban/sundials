 /************************************************************************
 * File        : kinwebbbd.d                                             *
 * Programmers : Allan G. Taylor, Alan C. Hindmarsh, Radu Serban @ LLNL  *
 * Version of  : 5 August 2003                                           *
 *-----------------------------------------------------------------------*
 * Example problem for KINSol, parallel machine case, BBD preconditioner.
 * This example solves a nonlinear system that arises from a system of  
 * partial differential equations. The PDE system is a food web         
 * population model, with predator-prey interaction and diffusion on the
 * unit square in two dimensions. The dependent variable vector is      
 * 
 *       1   2         ns
 * c = (c , c ,  ..., c  )              (denoted by the variable cc)
 * 
 * and the PDE's are as follows:
 *
 *                    i       i      
 *         0 = d(i)*(c     + c    )  +  f  (x,y,c)   (i=1,...,ns)
 *                    xx      yy         i
 *
 *   where
 *
 *                   i             ns         j  
 *   f  (x,y,c)  =  c  * (b(i)  + sum a(i,j)*c )
 *    i                           j=1
 *
 * The number of species is ns = 2 * np, with the first np being prey and
 * the last np being predators. The number np is both the number of prey and
 * predator species. The coefficients a(i,j), b(i), d(i) are
 *
 *   a(i,i) = -AA  (all i)
 *   a(i,j) = -GG  (i <= np , j >  np)
 *   a(i,j) =  EE  (i >  np,  j <= np)
 *   b(i) = BB * (1 + alpha * x * y)  (i <= np)
 *   b(i) =-BB * (1 + alpha * x * y)  (i >  np)
 *   d(i) = DPREY  (i <= np)
 *   d(i) = DPRED  ( i > np)
 *
 *  The various scalar parameters are set using define's 
 *  or in routine InitUserData.
 *  The boundary conditions are: normal derivative  =  0.
 *  The initial guess is constant in x and y, although the final
 *  solution is not.
 *
 *  The PDEs are discretized by central differencing on a MX by MY mesh.
 * 
 *  The nonlinear system is solved by KINSOL using the method specified in
 *  local variable globalstrat .
 *
 *  The preconditioner matrix is a band-block-diagonal matrix using the
 *  KINBBDPRE module.  The half-bandwidths are ml = mu = 2*ns - 1.
 * 
 * 
 * References:
 *
 * 1. Peter N. Brown and Youcef Saad,
 *    Hybrid Krylov Methods for Nonlinear Systems of Equations
 *    LLNL report UCRL-97645, November 1987.
 *  
 * 2. Peter N. Brown and Alan C. Hindmarsh,
 *    Reduced Storage Matrix Methods in Stiff ODE systems,
 *    Lawrence Livermore National Laboratory Report  UCRL-95088, Rev. 1,
 *    June 1987, and  Journal of Applied Mathematics and Computation, Vol. 31
 *    (May 1989), pp. 40-91. (Presents a description of the time-dependent
 *    version of this test problem.)
 *
 *
 *  Run command line: mpirun -np N -machinefile machines kinwebbbd
 *      where N = NPEX * NPEY  is the number of processors (see below)
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "sundialstypes.h"     /* def's of realtype and booleantype           */
#include "kinsol.h"            /* main KINSol header file                     */
#include "iterative.h"         /* enum for types of preconditioning           */
#include "kinspgmr.h"          /* use KINSpgmr linear solver                  */
#include "smalldense.h"        /* use generic DENSE solver for preconditioning*/
#include "nvector_parallel.h"  /* def's of type N_Vector, macro NV_DATA_P     */
#include "sundialsmath.h"      /* contains RSqrt routine                      */
#include "mpi.h"               /* MPI include file                            */
#include "kinbbdpre.h"         /* band preconditioner function prototypes     */


/* Problem Constants */

#define NUM_SPECIES     6  /*  must equal 2*(number of prey or predators)
                               number of prey =  number of predators      */

#define PI       3.1415926535898   /* pi */ 

#define NPEX        2            /* number of processors in the x-direction */
#define NPEY        2            /* number of processors in the y-direction */
#define MXSUB       10           /* number of x mesh points per subgrid     */
#define MYSUB       10           /* number of y mesh points per subgrid     */
#define MX          (NPEX*MXSUB) /* number of grid points in x-direction    */
#define MY          (NPEY*MYSUB) /* number of grid points in y-direction    */
#define NSMXSUB     (NUM_SPECIES * MXSUB)
#define NSMXSUB2    (NUM_SPECIES * (MXSUB+2))
#define NEQ         (NUM_SPECIES*MX*MY)  /* number of equations in system   */ 
#define AA          RCONST(1.0)    /* value of coefficient AA in above eqns */
#define EE          RCONST(10000.) /* value of coefficient EE in above eqns */
#define GG          RCONST(0.5e-6) /* value of coefficient GG in above eqns */
#define BB          RCONST(1.0)    /* value of coefficient BB in above eqns */
#define DPREY       RCONST(1.0)    /* value of coefficient dprey above */
#define DPRED       RCONST(0.5)    /* value of coefficient dpred above */
#define ALPHA       RCONST(1.0)    /* value of coefficient alpha above */
#define AX          RCONST(1.0)    /* total range of x variable */
#define AY          RCONST(1.0)    /* total range of y variable */
#define FTOL        RCONST(1.e-7)  /*  ftol tolerance */
#define STOL        RCONST(1.e-13) /*  stol tolerance */
#define THOUSAND    RCONST(1000.0) /* one thousand */
#define ZERO        RCONST(0.)     /* 0. */
#define ONE         RCONST(1.0)    /* 1. */
#define PREYIN      RCONST(1.0)    /* initial guess for prey concentrations. */
#define PREDIN      RCONST(30000.0)/* initial guess for predator concs.      */


/* User-defined vector access macro: IJ_Vptr */

/* IJ_Vptr is defined in order to translate from the underlying 3D structure
   of the dependent variable vector to the 1D storage scheme for an N-vector.
   IJ_Vptr(vv,i,j) returns a pointer to the location in vv corresponding to 
   indices is = 0, jx = i, jy = j.    */

#define IJ_Vptr(vv,i,j)   (&NV_Ith_P(vv, i*NUM_SPECIES + j*NSMXSUB))

/* Type : UserData 
   contains preconditioner blocks, pivot arrays, and problem constants */

typedef struct {
  realtype **acoef, *bcoef;
  N_Vector rates;
  realtype *cox, *coy;
  realtype ax, ay, dx, dy;
  long int Nlocal, mx, my, ns, np;
  realtype cext[NUM_SPECIES * (MXSUB+2)*(MYSUB+2)];
  long int my_pe, isubx, isuby, nsmxsub, nsmxsub2;
  MPI_Comm comm;
} *UserData;


/* Private Helper Functions */

static UserData AllocUserData(void);
static void InitUserData(long int my_pe, long int Nlocal, MPI_Comm comm, UserData data);
static void FreeUserData(UserData data);
static void SetInitialProfiles(N_Vector cc, N_Vector sc);
static void PrintOutput(long int my_pe, MPI_Comm comm, N_Vector cc);
static void PrintFinalStats(void *kmem);
static void WebRate(realtype xx, realtype yy, realtype *cxy, realtype *ratesxy,
                    void *f_data);
static realtype DotProd(long int size, realtype *x1, realtype *x2);
static void BSend(MPI_Comm comm, long int my_pe, long int isubx,
                  long int isuby, long int dsizex, long int dsizey,
                  realtype *cdata);
static void BRecvPost(MPI_Comm comm, MPI_Request request[], long int my_pe,
                      long int isubx, long int isuby,
                      long int dsizex, long int dsizey,
                      realtype *cext, realtype *buffer);
static void BRecvWait(MPI_Request request[], long int isubx,
                      long int isuby, long int dsizex, realtype *cext,
                      realtype *buffer);


/* Function called by the KINSol Solver */

static void funcprpr(N_Vector cc, N_Vector fval, void *f_data);

static void ccomm(long int Nlocal, N_Vector cc, void *data);

static void fcalcprpr(long int Nlocal, N_Vector cc, N_Vector fval, void *f_data);


/* Private function to check function return values */

static int check_flag(void *flagvalue, char *funcname, int opt, int id);


/***************************** Main Program ******************************/

int main(int argc, char *argv[])
{
  MPI_Comm comm;
  NV_Spec nvSpec;
  void *kmem, *pdata;
  UserData data;
  N_Vector cc, sc, constraints;
  int globalstrategy;
  long int Nlocal;
  realtype fnormtol, scsteptol, dq_rel_uu;
  int flag, mu, ml, maxl, maxlrst;
  int my_pe, npes, npelast = NPEX*NPEY-1;

  data = NULL;
  nvSpec = NULL;
  kmem = pdata = NULL;
  cc = sc = constraints = NULL;

  /* Get processor number and total number of pe's */
  MPI_Init(&argc, &argv);
  comm = MPI_COMM_WORLD;
  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &my_pe);

  if (npes != NPEX*NPEY) {
    if (my_pe == 0)
      printf("\nMPI_ERROR(0): npes=%d is not equal to NPEX*NPEY=%d\n", npes, NPEX*NPEY);
    return(1);
  }

  /* Allocate memory, and set problem data, initial values, tolerances */ 

  /* Set local length */
  Nlocal = NUM_SPECIES*MXSUB*MYSUB;

  /* Set nvSpec block */
  nvSpec = NV_SpecInit_Parallel(comm, Nlocal, NEQ, &argc, &argv);
  if (nvSpec==NULL) {
    if (my_pe == 0) check_flag((void *)nvSpec, "NV_SpecInit", 0, my_pe);
    MPI_Finalize();
    return(1); }

  /* Allocate and initialize user data block */
  data = AllocUserData();
  if (check_flag((void *)data, "AllocUserData", 2, my_pe)) MPI_Abort(comm, 1);
  InitUserData(my_pe, Nlocal, comm, data);

  /* Choose global strategy */
  globalstrategy = INEXACT_NEWTON;

  /* Allocate and initialize vectors */
  cc = N_VNew(nvSpec);
  if (check_flag((void *)cc, "N_VNew", 0, my_pe)) MPI_Abort(comm, 1);
  sc = N_VNew(nvSpec);
  if (check_flag((void *)sc, "N_VNew", 0, my_pe)) MPI_Abort(comm, 1);
  data->rates = N_VNew(nvSpec);
  if (check_flag((void *)data->rates, "N_VNew", 0, my_pe)) MPI_Abort(comm, 1);
  constraints = N_VNew(nvSpec);
  if (check_flag((void *)constraints, "N_VNew", 0, my_pe)) MPI_Abort(comm, 1);
  N_VConst(0.,constraints);
  
  SetInitialProfiles(cc, sc);

  fnormtol = FTOL; scsteptol = STOL;

  /* Call KINCreate/KINMalloc to initialize KINSOL: 
     nvSpec  points to machine environment data
     A pointer to KINSOL problem memory is returned and stored in kmem. */
  kmem = KINCreate();
  if (check_flag((void *)kmem, "KINCreate", 0, my_pe)) MPI_Abort(comm, 1);
  flag = KINMalloc(kmem, funcprpr, nvSpec);
  if (check_flag(&flag, "KINMalloc", 1, my_pe)) MPI_Abort(comm, 1);

  flag = KINSetFdata(kmem, data);
  if (check_flag(&flag, "KINSetFdata", 1, my_pe)) MPI_Abort(comm, 1);
  flag = KINSetConstraints(kmem, constraints);
  if (check_flag(&flag, "KINSetConstraints", 1, my_pe)) MPI_Abort(comm, 1);
  flag = KINSetFuncNormTol(kmem, fnormtol);
  if (check_flag(&flag, "KINSetFuncNormTol", 1, my_pe)) MPI_Abort(comm, 1);
  flag = KINSetScaledStepTol(kmem, scsteptol);
  if (check_flag(&flag, "KINSetScaledStepTol", 1, my_pe)) MPI_Abort(comm, 1);
  
  /* Call KBBDAlloc to initialize and allocate memory for the band-block-
     diagonal preconditioner, and specify the local and communication
     functions fcalcprpr and ccomm. */
  dq_rel_uu = ZERO;
  mu = ml = 2*NUM_SPECIES - 1;

  pdata = KBBDPrecAlloc(kmem, Nlocal, mu, ml, dq_rel_uu, fcalcprpr, ccomm);
  if (check_flag((void *)pdata, "KBBDPrecAlloc", 0, my_pe)) MPI_Abort(comm, 1);

  /* Call KBBDSpgmr to specify the linear solver KINSPGMR 
     with preconditioner KINBBDPRE */
  maxl = 20; maxlrst = 2;
  flag = KBBDSpgmr(kmem, maxl, pdata);
  if (check_flag(&flag, "KBBDSpgmr", 1, my_pe)) MPI_Abort(comm, 1);

  flag = KINSpgmrSetMaxRestarts(kmem, maxlrst);
  if (check_flag(&flag, "KINSpgmrSetMaxRestarts", 1, my_pe)) MPI_Abort(comm, 1);

  /* Print out the problem size, solution parameters, initial guess. */
  if (my_pe == 0) {
    printf("\nPredator-prey test problem--  KINSol (parallel-BBD version)\n\n");
    
    printf("Mesh dimensions = %d X %d\n", MX, MY);
    printf("Number of species = %d\n", NUM_SPECIES);
    printf("Total system size = %d\n\n", NEQ);
    printf("Subgrid dimensions = %d X %d\n", MXSUB, MYSUB);
    printf("Processor array is %d X %d\n\n", NPEX, NPEY);
    printf("Flag globalstrategy = %ld (0 = Inex. Newton, 1 = Linesearch)\n",
           globalstrategy);
    printf("Linear solver is SPGMR with maxl = %d, maxlrst = %d\n",
           maxl, maxlrst);
    printf("Preconditioning uses band-block-diagonal matrix from KINBBDPRE\n");
    printf("  with matrix half-bandwidths ml, mu = %d  %d\n", ml, mu);
    printf("Tolerance parameters:  fnormtol = %g   scsteptol = %g\n",
           fnormtol, scsteptol);
    
    printf("\nInitial profile of concentration\n");
    printf("At all mesh points:  %g %g %g   %g %g %g\n", PREYIN,PREYIN,PREYIN,
           PREDIN,PREDIN,PREDIN);
  }

  /* call KINSol and print output concentration profile */
  flag = KINSol(kmem,           /* KINSol memory block */
                cc,             /* initial guesss on input; solution vector */
                globalstrategy, /* global stragegy choice */
                sc,             /* scaling vector, for the variable cc */
                sc);            /* scaling vector for function values fval */
  if (check_flag(&flag, "KINSol", 1, my_pe)) MPI_Abort(comm, 1);

  if (my_pe == 0) printf("\n\n\nComputed equilibrium species concentrations:\n");
  if (my_pe == 0 || my_pe==npelast) PrintOutput(my_pe, comm, cc);
  
  /* Print final statistics and free memory */
  if (my_pe == 0) PrintFinalStats(kmem);

  N_VFree(cc);
  N_VFree(sc);
  N_VFree(constraints);
  KBBDPrecFree(pdata);
  KINFree(kmem);
  FreeUserData(data);
  NV_SpecFree_Parallel(nvSpec);

  MPI_Finalize();

  return(0);

} /* end of Main *********************************************************/


/*********************** Private Helper Functions ************************/

/* Allocate memory for data structure of type UserData */

static UserData AllocUserData(void)
{
  UserData data;
  
  data = (UserData) malloc(sizeof *data);
  
  (data->acoef) = denalloc(NUM_SPECIES);
  (data->bcoef) = (realtype *)malloc(NUM_SPECIES * sizeof(realtype));
  (data->cox)   = (realtype *)malloc(NUM_SPECIES * sizeof(realtype));
  (data->coy)   = (realtype *)malloc(NUM_SPECIES * sizeof(realtype));
  
  return(data);
  
} /* end of routine AllocUserData ****************************************/


/* Readability definitions used in other routines below */
#define acoef  (data->acoef)
#define bcoef  (data->bcoef)
#define cox    (data->cox)
#define coy    (data->coy)


/* Load problem constants in data */

static void InitUserData(long int my_pe, long int Nlocal, MPI_Comm comm, UserData data)
{
  int i, j, np;
  realtype *a1,*a2, *a3, *a4, dx2, dy2;

  data->mx = MX;
  data->my = MY;
  data->ns = NUM_SPECIES;
  data->np = NUM_SPECIES/2;
  data->ax = AX;
  data->ay = AY;
  data->dx = (data->ax)/(MX-1);
  data->dy = (data->ay)/(MY-1);
  data->my_pe = my_pe;
  data->Nlocal = Nlocal;
  data->comm = comm;
  data->isuby = my_pe/NPEX;
  data->isubx = my_pe - data->isuby*NPEX;
  data->nsmxsub = NUM_SPECIES * MXSUB;
  data->nsmxsub2 = NUM_SPECIES * (MXSUB+2);
  
  /* Set up the coefficients a and b plus others found in the equations */
  np = data->np;

  dx2=(data->dx)*(data->dx); dy2=(data->dy)*(data->dy);

  for (i = 0; i < np; i++) {
    a1= &(acoef[i][np]);
    a2= &(acoef[i+np][0]);
    a3= &(acoef[i][0]);
    a4= &(acoef[i+np][np]);

    /*  Fill in the portion of acoef in the four quadrants, row by row */
    for (j = 0; j < np; j++) {
      *a1++ =  -GG;
      *a2++ =   EE;
      *a3++ = ZERO;
      *a4++ = ZERO;
    }

    /* and then change the diagonal elements of acoef to -AA */
    acoef[i][i]=-AA;
    acoef[i+np][i+np] = -AA;

    bcoef[i] = BB;
    bcoef[i+np] = -BB;

    cox[i]=DPREY/dx2;
    cox[i+np]=DPRED/dx2;

    coy[i]=DPREY/dy2;
    coy[i+np]=DPRED/dy2;
  }

} /* end of routine InitUserData *****************************************/

/* Free data memory */

static void FreeUserData(UserData data)
{

  denfree(acoef);
  free(bcoef);
  free(cox); free(coy);
  N_VFree(data->rates);

  free(data);

} /* end of routine FreeUserData *****************************************/

/* Set initial conditions in cc */

static void SetInitialProfiles(N_Vector cc, N_Vector sc)
{
  int i, jx, jy;
  realtype *cloc, *sloc;
  realtype  ctemp[NUM_SPECIES], stemp[NUM_SPECIES];

  /* Initialize arrays ctemp and stemp used in the loading process */
  for (i = 0; i < NUM_SPECIES/2; i++) {
    ctemp[i] = PREYIN;
    stemp[i] = ONE;
  }
  for (i = NUM_SPECIES/2; i < NUM_SPECIES; i++) {
    ctemp[i] = PREDIN;
    stemp[i] = 0.00001;
  }

  /* Load initial profiles into cc and sc vector from ctemp and stemp. */
  for (jy = 0; jy < MYSUB; jy++) {
    for (jx=0; jx < MXSUB; jx++) {
      cloc = IJ_Vptr(cc,jx,jy);
      sloc = IJ_Vptr(sc,jx,jy);
      for (i = 0; i < NUM_SPECIES; i++){
        cloc[i] = ctemp[i];
        sloc[i] = stemp[i];
      }
    }
  }
  
} /* end of routine SetInitialProfiles ***********************************/

/* Print sample of current cc values */

static void PrintOutput(long int my_pe, MPI_Comm comm, N_Vector cc)
{
  int is, i0, npelast;
  realtype  *ct, tempc[NUM_SPECIES];
  MPI_Status status;
  
  npelast = NPEX*NPEY - 1;
  
  ct = NV_DATA_P(cc);
  
  /* Send the cc values (for all species) at the top right mesh point to PE 0 */
  if (my_pe == npelast) {
    i0 = NUM_SPECIES*(MXSUB*MYSUB-1);
    if (npelast!=0)
      MPI_Send(&ct[i0],NUM_SPECIES,PVEC_REAL_MPI_TYPE,0,0,comm);
    else  /* single processor case */
      for (is = 0; is < NUM_SPECIES; is++) tempc[is]=ct[i0+is];   
  }
  
  /* On PE 0, receive the cc values at top right, then print performance data 
     and sampled solution values */
  if (my_pe == 0) {
    
    if (npelast != 0)
      MPI_Recv(&tempc[0],NUM_SPECIES,PVEC_REAL_MPI_TYPE,npelast,0,comm,&status);
    
    printf("\nAt bottom left:");
    for (is = 0; is < NUM_SPECIES; is++){
      if ((is%6)*6== is) printf("\n");
      printf(" %g",ct[is]);
    }
    
    printf("\n\nAt top right:");
    for (is = 0; is < NUM_SPECIES; is++) {
      if ((is%6)*6 == is) printf("\n");
      printf(" %g",tempc[is]);
    }
    printf("\n\n");
  }
  
} /* end of routine PrintOutput ******************************************/

/* Print final statistics contained in iopt */

static void PrintFinalStats(void *kmem)
{
  int nni, nfe, nli, npe, nps, ncfl, nfeSG, flag;
  
  flag = KINGetNumNonlinSolvIters(kmem, &nni);
  check_flag(&flag, "KINGetNumNonlinSolvIters", 1, 0);
  flag = KINGetNumFuncEvals(kmem, &nfe);
  check_flag(&flag, "KINGetNumFuncEvals", 1, 0);
  flag = KINSpgmrGetNumLinIters(kmem, &nli);
  check_flag(&flag, "KINSpgmrGetNumLinIters", 1, 0);
  flag = KINSpgmrGetNumPrecEvals(kmem, &npe);
  check_flag(&flag, "KINSpgmrGetNumPrecEvals", 1, 0);
  flag = KINSpgmrGetNumPrecSolves(kmem, &nps);
  check_flag(&flag, "KINSpgmrGetNumPrecSolves", 1, 0);
  flag = KINSpgmrGetNumConvFails(kmem, &ncfl);
  check_flag(&flag, "KINSpgmrGetNumConvFails", 1, 0);
  flag = KINSpgmrGetNumFuncEvals(kmem, &nfeSG);
  check_flag(&flag, "KINSpgmrGetNumFuncEvals", 1, 0);

  printf("\nFinal Statistics.. \n\n");
  printf("nni    = %5d    nli   = %5d\n", nni, nli);
  printf("nfe    = %5d    nfeSG = %5d\n", nfe, nfeSG);
  printf("nps    = %5d    npe   = %5d     ncfl  = %5d\n", nps, npe, ncfl);
  
} /* end of routine PrintFinalStats **************************************/

/* Routine to send boundary data to neighboring PEs */

static void BSend(MPI_Comm comm, long int my_pe, 
                  long int isubx, long int isuby, 
                  long int dsizex, long int dsizey, realtype *cdata)
{
  int i, ly;
  long int offsetc, offsetbuf;
  realtype bufleft[NUM_SPECIES*MYSUB], bufright[NUM_SPECIES*MYSUB];

  /* If isuby > 0, send data from bottom x-line of u */
  if (isuby != 0)
    MPI_Send(&cdata[0], dsizex, PVEC_REAL_MPI_TYPE, my_pe-NPEX, 0, comm);

  /* If isuby < NPEY-1, send data from top x-line of u */
  if (isuby != NPEY-1) {
    offsetc = (MYSUB-1)*dsizex;
    MPI_Send(&cdata[offsetc], dsizex, PVEC_REAL_MPI_TYPE, my_pe+NPEX, 0, comm);
  }

  /* If isubx > 0, send data from left y-line of u (via bufleft) */
  if (isubx != 0) {
    for (ly = 0; ly < MYSUB; ly++) {
      offsetbuf = ly*NUM_SPECIES;
      offsetc = ly*dsizex;
      for (i = 0; i < NUM_SPECIES; i++)
        bufleft[offsetbuf+i] = cdata[offsetc+i];
    }
    MPI_Send(&bufleft[0], dsizey, PVEC_REAL_MPI_TYPE, my_pe-1, 0, comm);   
  }

  /* If isubx < NPEX-1, send data from right y-line of u (via bufright) */
  if (isubx != NPEX-1) {
    for (ly = 0; ly < MYSUB; ly++) {
      offsetbuf = ly*NUM_SPECIES;
      offsetc = offsetbuf*MXSUB + (MXSUB-1)*NUM_SPECIES;
      for (i = 0; i < NUM_SPECIES; i++)
        bufright[offsetbuf+i] = cdata[offsetc+i];
    }
    MPI_Send(&bufright[0], dsizey, PVEC_REAL_MPI_TYPE, my_pe+1, 0, comm);   
  }

} /* end of routine BSend ************************************************/

/* Routine to start receiving boundary data from neighboring PEs.
   Notes:
   1) buffer should be able to hold 2*NUM_SPECIES*MYSUB realtype entries,
   should be passed to both the BRecvPost and BRecvWait functions, and
   should not be manipulated between the two calls.
   2) request should have 4 entries, and should be passed in both calls also. */

static void BRecvPost(MPI_Comm comm, MPI_Request request[], long int my_pe,
                      long int isubx, long int isuby,
                      long int dsizex, long int dsizey,
                      realtype *cext, realtype *buffer)
{
  long int offsetce;

  /* Have bufleft and bufright use the same buffer */
  realtype *bufleft = buffer, *bufright = buffer+NUM_SPECIES*MYSUB;
  
  /* If isuby > 0, receive data for bottom x-line of cext */
  if (isuby != 0)
    MPI_Irecv(&cext[NUM_SPECIES], dsizex, PVEC_REAL_MPI_TYPE,
              my_pe-NPEX, 0, comm, &request[0]);
  
  /* If isuby < NPEY-1, receive data for top x-line of cext */
  if (isuby != NPEY-1) {
    offsetce = NUM_SPECIES*(1 + (MYSUB+1)*(MXSUB+2));
    MPI_Irecv(&cext[offsetce], dsizex, PVEC_REAL_MPI_TYPE,
              my_pe+NPEX, 0, comm, &request[1]);
  }
  
  /* If isubx > 0, receive data for left y-line of cext (via bufleft) */
  if (isubx != 0) {
    MPI_Irecv(&bufleft[0], dsizey, PVEC_REAL_MPI_TYPE,
              my_pe-1, 0, comm, &request[2]);
  }
  
  /* If isubx < NPEX-1, receive data for right y-line of cext (via bufright) */
  if (isubx != NPEX-1) {
    MPI_Irecv(&bufright[0], dsizey, PVEC_REAL_MPI_TYPE,
              my_pe+1, 0, comm, &request[3]);
  }
  
} /* end of routine BRecvPost ********************************************/

/* Routine to finish receiving boundary data from neighboring PEs.
   Notes:
   1) buffer should be able to hold 2*NUM_SPECIES*MYSUB realtype entries,
   should be passed to both the BRecvPost and BRecvWait functions, and
   should not be manipulated between the two calls.
   2) request should have 4 entries, and should be passed in both calls also. */

static void BRecvWait(MPI_Request request[], long int isubx,
                      long int isuby, long int dsizex, realtype *cext,
                      realtype *buffer)
{
  int i, ly;
  long int dsizex2, offsetce, offsetbuf;
  realtype *bufleft = buffer, *bufright = buffer+NUM_SPECIES*MYSUB;
  MPI_Status status;
  
  dsizex2 = dsizex + 2*NUM_SPECIES;
  
  /* If isuby > 0, receive data for bottom x-line of cext */
  if (isuby != 0)
    MPI_Wait(&request[0],&status);
  
  /* If isuby < NPEY-1, receive data for top x-line of cext */
  if (isuby != NPEY-1)
    MPI_Wait(&request[1],&status);

  /* If isubx > 0, receive data for left y-line of cext (via bufleft) */
  if (isubx != 0) {
    MPI_Wait(&request[2],&status);
    
    /* Copy the buffer to cext */
    for (ly = 0; ly < MYSUB; ly++) {
      offsetbuf = ly*NUM_SPECIES;
      offsetce = (ly+1)*dsizex2;
      for (i = 0; i < NUM_SPECIES; i++)
        cext[offsetce+i] = bufleft[offsetbuf+i];
    }
  }
  
  /* If isubx < NPEX-1, receive data for right y-line of cext (via bufright) */
  if (isubx != NPEX-1) {
    MPI_Wait(&request[3],&status);
    
    /* Copy the buffer to cext */
    for (ly = 0; ly < MYSUB; ly++) {
      offsetbuf = ly*NUM_SPECIES;
      offsetce = (ly+2)*dsizex2 - NUM_SPECIES;
      for (i = 0; i < NUM_SPECIES; i++)
        cext[offsetce+i] = bufright[offsetbuf+i];
    }
  }
  
} /* end of routine BRecvWait ********************************************/


/***************** Functions Called by the KINSol Solver ******************/

/* ccomm routine.  This routine performs all communication 
   between processors of data needed to calculate f. */

static void ccomm(long int Nlocal, N_Vector cc, void *userdata)
{

  realtype *cdata, *cext, buffer[2*NUM_SPECIES*MYSUB];
  UserData data;
  MPI_Comm comm;
  long int my_pe, isubx, isuby, nsmxsub, nsmysub;
  MPI_Request request[4];

  /* Get comm, my_pe, subgrid indices, data sizes, extended array cext */
  data = (UserData) userdata;
  comm = data->comm;  my_pe = data->my_pe;
  isubx = data->isubx;   isuby = data->isuby;
  nsmxsub = data->nsmxsub;
  nsmysub = NUM_SPECIES*MYSUB;
  cext = data->cext;

  cdata = NV_DATA_P(cc);

  /* Start receiving boundary data from neighboring PEs */
  BRecvPost(comm, request, my_pe, isubx, isuby, nsmxsub, nsmysub, cext, buffer);

  /* Send data from boundary of local grid to neighboring PEs */
  BSend(comm, my_pe, isubx, isuby, nsmxsub, nsmysub, cdata);

  /* Finish receiving boundary data from neighboring PEs */
  BRecvWait(request, isubx, isuby, nsmxsub, cext, buffer);

} /* end of routine ccomm ************************************************/

/* System function for predator-prey system - calculation part */

static void fcalcprpr(long int Nlocal, N_Vector cc, N_Vector fval, void *f_data)
{
  realtype xx, yy, *cxy, *rxy, *fxy, dcydi, dcyui, dcxli, dcxri;
  realtype *cext, dely, delx, *cdata;
  long int i, jx, jy, is, ly;
  long int isubx, isuby, nsmxsub, nsmxsub2;
  long int shifty, offsetc, offsetce, offsetcl, offsetcr, offsetcd, offsetcu;
  UserData data;
  
  data = (UserData)f_data;
  cdata = NV_DATA_P(cc);
  
  /* Get subgrid indices, data sizes, extended work array cext */
  isubx = data->isubx;   isuby = data->isuby;
  nsmxsub = data->nsmxsub; nsmxsub2 = data->nsmxsub2;
  cext = data->cext;
  
  /* Copy local segment of cc vector into the working extended array cext */
  offsetc = 0;
  offsetce = nsmxsub2 + NUM_SPECIES;
  for (ly = 0; ly < MYSUB; ly++) {
    for (i = 0; i < nsmxsub; i++) cext[offsetce+i] = cdata[offsetc+i];
    offsetc = offsetc + nsmxsub;
    offsetce = offsetce + nsmxsub2;
  }
  
  /* To facilitate homogeneous Neumann boundary conditions, when this is a
     boundary PE, copy data from the first interior mesh line of cc to cext */
  
  /* If isuby = 0, copy x-line 2 of cc to cext */
  if (isuby == 0) {
    for (i = 0; i < nsmxsub; i++) cext[NUM_SPECIES+i] = cdata[nsmxsub+i];
  }
  
  /* If isuby = NPEY-1, copy x-line MYSUB-1 of cc to cext */
  if (isuby == NPEY-1) {
    offsetc = (MYSUB-2)*nsmxsub;
    offsetce = (MYSUB+1)*nsmxsub2 + NUM_SPECIES;
    for (i = 0; i < nsmxsub; i++) cext[offsetce+i] = cdata[offsetc+i];
  }
  
  /* If isubx = 0, copy y-line 2 of cc to cext */
  if (isubx == 0) {
    for (ly = 0; ly < MYSUB; ly++) {
      offsetc = ly*nsmxsub + NUM_SPECIES;
      offsetce = (ly+1)*nsmxsub2;
      for (i = 0; i < NUM_SPECIES; i++) cext[offsetce+i] = cdata[offsetc+i];
    }
  }
  
  /* If isubx = NPEX-1, copy y-line MXSUB-1 of cc to cext */
  if (isubx == NPEX-1) {
    for (ly = 0; ly < MYSUB; ly++) {
      offsetc = (ly+1)*nsmxsub - 2*NUM_SPECIES;
      offsetce = (ly+2)*nsmxsub2 - NUM_SPECIES;
      for (i = 0; i < NUM_SPECIES; i++) cext[offsetce+i] = cdata[offsetc+i];
    }
  }

  /* Loop over all mesh points, evaluating rate arra at each point */
  delx = data->dx;
  dely = data->dy;
  shifty = (MXSUB+2)*NUM_SPECIES;

  for (jy = 0; jy < MYSUB; jy++) {

    yy = dely*(jy + isuby * MYSUB);

    for (jx = 0; jx < MXSUB; jx++) {

      xx = delx * (jx + isubx * MXSUB);
      cxy = IJ_Vptr(cc,jx,jy);
      rxy = IJ_Vptr(data->rates,jx,jy);
      fxy = IJ_Vptr(fval,jx,jy);
      
      WebRate(xx, yy, cxy, rxy, f_data);

      offsetc = (jx+1)*NUM_SPECIES + (jy+1)*NSMXSUB2;
      offsetcd = offsetc - shifty;
      offsetcu = offsetc + shifty;
      offsetcl = offsetc - NUM_SPECIES;
      offsetcr = offsetc + NUM_SPECIES;
      
      for (is = 0; is < NUM_SPECIES; is++) {
        
        /* differencing in x */
        dcydi = cext[offsetc+is]  - cext[offsetcd+is];
        dcyui = cext[offsetcu+is] - cext[offsetc+is];
        
        /* differencing in y */
        dcxli = cext[offsetc+is]  - cext[offsetcl+is];
        dcxri = cext[offsetcr+is] - cext[offsetc+is];
        
        /* compute the value at xx , yy */
        fxy[is] = (coy)[is] * (dcyui - dcydi) +
          (cox)[is] * (dcxri - dcxli) + rxy[is];
        
      } /* end of is loop */
      
    } /* end of jx loop */
    
  } /* end of jy loop */
  
} /* end of routine fcalcprpr ********************************************/

/* System function routine.  Evaluate funcprpr(cc).  First call ccomm to do 
  communication of subgrid boundary data into cext.  Then calculate funcprpr
  by a call to fcalcprpr. */

static void funcprpr(N_Vector cc, N_Vector fval, void *f_data)
{
  UserData data;

  data = (UserData) f_data;

  /* Call ccomm to do inter-processor communicaiton */
  ccomm (data->Nlocal, cc, data);

  /* Call fcalcprpr to calculate all right-hand sides */
  fcalcprpr (data->Nlocal, cc, fval, data);
  
} /* end of routine funcprpr *********************************************/

/* Interaction rate function routine */

static void WebRate(realtype xx, realtype yy, realtype *cxy, realtype *ratesxy, 
                    void *f_data)
{
  long int i;
  realtype fac;
  UserData data;
  
  data = (UserData)f_data;
  
  for (i = 0; i<NUM_SPECIES; i++)
    ratesxy[i] = DotProd(NUM_SPECIES, cxy, acoef[i]);
  
  fac = ONE + ALPHA * xx * yy;
  
  for (i = 0; i < NUM_SPECIES; i++)
    ratesxy[i] = cxy[i] * ( bcoef[i] * fac + ratesxy[i] );
  
} /* end of routine WebRate **********************************************/

/* Dot product routine for realtype arrays */

static realtype DotProd(long int size, realtype *x1, realtype *x2)
{
  long int i;
  realtype *xx1, *xx2, temp = ZERO;
  
  xx1 = x1; xx2 = x2;
  for (i = 0; i < size; i++) temp += (*xx1++) * (*xx2++);
  return(temp);
  
} /* end of routine DotProd **********************************************/

/*********************** Private Helper Function ************************/

/* Check function return value...
     opt == 0 means SUNDIALS function allocates memory so check if
              returned NULL pointer
     opt == 1 means SUNDIALS function returns a flag so check if
              flag == SUCCESS
     opt == 2 means function allocates memory so check if returned
              NULL pointer */

static int check_flag(void *flagvalue, char *funcname, int opt, int id)
{
  int *errflag;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && flagvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1); }

  /* Check if flag != SUCCESS */
  else if (opt == 1) {
    errflag = flagvalue;
    if (*errflag != SUCCESS) {
      fprintf(stderr, "\nSUNDIALS_ERROR(%d): %s() failed with flag = %d\n\n",
	      id, funcname, *errflag);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && flagvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR(%d): %s() failed - returned NULL pointer\n\n",
	    id, funcname);
    return(1); }

  return(0);
}
