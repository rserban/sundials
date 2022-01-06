#!/usr/bin/env python3

"""
 -----------------------------------------------------------------
 Programmer(s): Cody J. Balos @ LLNL
 -----------------------------------------------------------------
 Based on the example kinRoberts_fp.c by Carol Woodward @ LLNL
 -----------------------------------------------------------------
 Example problem:

 The following is a simple example problem, with the coding
 needed for its solution by the accelerated fixed point solver in
 KINSOL.
 The problem is from chemical kinetics, and consists of solving
 the first time step in a Backward Euler solution for the
 following three rate equations:
    dy1/dt = -.04*y1 + 1.e4*y2*y3
    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2
    dy3/dt = 3.e2*(y2)^2
 on the interval from t = 0.0 to t = 0.1, with initial
 conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
 Run statistics (optional outputs) are printed at the end.
 -----------------------------------------------------------------
"""

import ctypes
import kinsol as kin
import numpy as np
import numba as nb
from numba import cfunc
from timeit import default_timer as timer
import faulthandler

class Problem():

  def __init__(self):
    # constants
    self.NEQ    = 3     # number of equations
    self.Y10    = 1.0   # initial y components
    self.Y20    = 0.0
    self.Y30    = 0.0
    self.TOL    = 1e-10 # function tolerance
    self.DSTEP  = 0.1   # size of the single time step used
    self.PRIORS = 2

  def funcRoberts(self, y, y_len, g, g_len, udata):
    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    yd1 = self.DSTEP * (-0.04*y1 + 1.0e4*y2*y3)
    yd3 = self.DSTEP * 3.0e2*y2*y2

    g[0] = yd1 + self.Y10
    g[1] = -yd1 - yd3 + self.Y20
    g[2] = yd3 + self.Y30

    return 0


def funcRobertsCtypes(y, y_len, g, g_len, udata):
  # constants
  NEQ    = udata.NEQ   # number of equations
  Y10    = udata.Y10   # initial y components
  Y20    = udata.Y20
  Y30    = udata.Y30
  TOL    = udata.TOL   # function tolerance
  DSTEP  = udata.DSTEP # size of the single time step used
  PRIORS = udata.PRIORS

  y1 = y[0]
  y2 = y[1]
  y3 = y[2]

  yd1 = DSTEP * (-0.04*y1 + 1.0e4*y2*y3)
  yd3 = DSTEP * 3.0e2*y2*y2

  g[0] = yd1 + Y10
  g[1] = -yd1 - yd3 + Y20
  g[2] = yd3 + Y30

  return 0


SysFnSpec = nb.types.int32(nb.types.CPointer(nb.types.double),
                           nb.types.int32,
                           nb.types.CPointer(nb.types.double),
                           nb.types.int32,
                           nb.types.CPointer(nb.types.void))
@cfunc(SysFnSpec, nopython=True)
def funcRobertsNumba(y, y_len, g, g_len, udata):
  # constants
  NEQ    = 3     # number of equations
  Y10    = 1.0   # initial y components
  Y20    = 0.0
  Y30    = 0.0
  TOL    = 1e-10 # function tolerance
  DSTEP  = 0.1   # size of the single time step used
  PRIORS = 2

  y1 = y[0]
  y2 = y[1]
  y3 = y[2]

  yd1 = DSTEP * (-0.04*y1 + 1.0e4*y2*y3)
  yd3 = DSTEP * 3.0e2*y2*y2

  g[0] = yd1 + Y10
  g[1] = -yd1 - yd3 + Y20
  g[2] = yd3 + Y30

  return 0


def kinErrHandler(error_code, module, function, msg, user_data):
  raise Exception(f'ERROR code {error_code} encountered in {module}::{function}: {msg}')


def solve(problem, callback_option):
  print("------------------------------------------------------------\n")
  print("------------------------------------------------------------\n")
  print("Using callback option %d\n" % callback_option)
  print("Example problem from chemical kinetics solving")
  print("the first time step in a Backward Euler solution for the")
  print("following three rate equations:")
  print("    dy1/dt = -.04*y1 + 1.e4*y2*y3")
  print("    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2")
  print("    dy3/dt = 3.e2*(y2)^2")
  print("on the interval from t = 0.0 to t = 0.1, with initial")
  print("conditions: y1 = 1.0, y2 = y3 = 0.")
  print("Solution method: Anderson accelerated fixed point iteration.\n")

  start = timer()

  sunctx = kin.Context().get()

  # --------------------------------------
  # Create vectors for solution and scales
  # --------------------------------------

  # Underlying data arrays for the solution vector
  y = np.zeros(problem.NEQ)

  # Create N_Vector objects
  sunvec_y = kin.N_VMake_Serial(y, sunctx)
  sunvec_scale = kin.N_VNew_Serial(problem.NEQ, sunctx)
  kin.N_VConst(0.0, sunvec_scale)

  # Underlying data array for the scale vector
  flag, scale = kin.N_VArrayView(sunvec_scale)
  print(scale)

  # -----------------------------------------
  # Initialize and allocate memory for KINSOL
  # -----------------------------------------

  kmem = kin.KINCreate(sunctx)

  # Set number of prior residuals used in Anderson acceleration.
  flag = kin.KINSetMAA(kmem, problem.PRIORS)
  if flag < 0: raise RuntimeError(f'KINSetMAA returned {flag}')

  # Attach the problem object as the kinsol user_data so that
  # it is provided to the callback functions.
  kin.KINSetUserData(kmem, problem)

  #----------------------------------------------------------------------------
  # It is required to wrap Python callback functions so that they are callable
  # from C. The kin.Register<>Fn functions should be used to do this. There are
  # a few options:
  #----------------------------------------------------------------------------

  # (1) Use ctypes.
  if callback_option == 1:
    sysfn = kin.RegisterFn(funcRobertsCtypes, kin.cfunctypes.KINSysFn)

  # (2) Note, with ctypes, you can even pass a capturing lambda:
  if callback_option == 2:
    sysfn = kin.RegisterFn(lambda y,y_len,g,g_len,udata: problem.funcRoberts(y,y_len,g,g_len,udata), kin.cfunctypes.KINSysFn)

  # (3) Use a numba.cfunc to try and achieve the best performance.
  #     The drawback here is that the numba functions cannot access the problem data.
  if callback_option == 3:
    sysfn = kin.RegisterNumbaFn(funcRobertsNumba, kin.cfunctypes.KINSysFn)

  # initialize the solver
  flag = kin.KINInit(kmem, sysfn, sunvec_y)
  if flag < 0: raise RuntimeError(f'KINInitPy returned {flag}')

  # --------------------------
  # Set KINSOL optional inputs
  # --------------------------

  # specify stopping tolerance based on residual
  fnormtol = problem.TOL
  flag = kin.KINSetFuncNormTol(kmem, fnormtol)
  if flag < 0: raise RuntimeError(f'KINSetFuncNormTol returned {flag}')

  flag = kin.KINSetErrFilename(kmem, "error.log")
  if flag < 0: raise RuntimeError(f'KINSetErrFilename returned {flag}')

  # TODO: determine why this causes a segfault
  # flag = kin.KINSetErrHandlerFn(kmem, kin.RegisterFn(kinErrHandler, kin.cfunctypes.KINErrHandlerFn), problem)
  # if flag < 0: raise RuntimeError(f'KINSetErrHandlerFn returned {flag}')

  # -------------
  # Initial guess
  # -------------

  y[0] = 1.0

  # ----------------------------
  # Call KINSOL to solve problem
  # ----------------------------

  # no scaling used
  kin.N_VConst(1.0, sunvec_scale)

  # call main solver
  flag = kin.KINSol(kmem,          # KINSOL memory block
                    sunvec_y,      # initial guess on input; solution vector
                    kin.KIN_FP,    # global strategy choice
                    sunvec_scale,  # scaling vector for the variable cc
                    sunvec_scale)  # scaling vector for function values fval
  if flag < 0: raise RuntimeError(f'KINSol returned {flag}')

  # ------------------------------------
  # Print solution and solver statistics
  # ------------------------------------
  flag, fnorm = kin.KINGetFuncNorm(kmem)
  if flag < 0: raise RuntimeError(f'KINGetFuncNorm returned {flag}')

  print('Computed solution (||F|| = %Lg):\n' % fnorm)
  print('y = %14.6e  %14.6e  %14.6e' % (y[0], y[1], y[2]))
  PrintFinalStats(kmem)

  # -----------
  # Free memory
  # -----------

  kin.N_VDestroy(sunvec_y)
  kin.N_VDestroy(sunvec_scale)
  kin.KINFree(kmem)

  end = timer()
  print("\nElapsed time: %g\n" % (end - start))


def PrintFinalStats(kmem):
  flag, nni = kin.KINGetNumNonlinSolvIters(kmem)
  if flag < 0: raise RuntimeError(f'KINGetNumNonlinSolvIters returned {flag}')
  flag, nfe = kin.KINGetNumFuncEvals(kmem)
  if flag < 0: raise RuntimeError(f'KINGetNumFuncEvals returned {flag}')

  print('\nFinal Statistics...')
  print('nni      = %6ld      nfe     = %6ld' % (nni, nfe))


if __name__ == '__main__':
  faulthandler.enable()
  funcRobertsNumba(np.zeros(3), 3, np.zeros(3), 3, None) # force compilation
  solve(Problem(), 1)
  solve(Problem(), 2)
  solve(Problem(), 3)