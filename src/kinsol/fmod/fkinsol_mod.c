/* ----------------------------------------------------------------------------
 * This file was automatically generated by SWIG (http://www.swig.org).
 * Version 4.0.0
 *
 * This file is not intended to be easily readable and contains a number of
 * coding conventions designed to improve portability and efficiency. Do not make
 * changes to this file unless you know what you are doing--modify the SWIG
 * interface file instead.
 * ----------------------------------------------------------------------------- */

/* ---------------------------------------------------------------
 * Programmer(s): Auto-generated by swig.
 * ---------------------------------------------------------------
 * SUNDIALS Copyright Start
 * Copyright (c) 2002-2022, Lawrence Livermore National Security
 * and Southern Methodist University.
 * All rights reserved.
 *
 * See the top-level LICENSE and NOTICE files for details.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * SUNDIALS Copyright End
 * -------------------------------------------------------------*/

/* -----------------------------------------------------------------------------
 *  This section contains generic SWIG labels for method/variable
 *  declarations/attributes, and other compiler dependent labels.
 * ----------------------------------------------------------------------------- */

/* template workaround for compilers that cannot correctly implement the C++ standard */
#ifndef SWIGTEMPLATEDISAMBIGUATOR
# if defined(__SUNPRO_CC) && (__SUNPRO_CC <= 0x560)
#  define SWIGTEMPLATEDISAMBIGUATOR template
# elif defined(__HP_aCC)
/* Needed even with `aCC -AA' when `aCC -V' reports HP ANSI C++ B3910B A.03.55 */
/* If we find a maximum version that requires this, the test would be __HP_aCC <= 35500 for A.03.55 */
#  define SWIGTEMPLATEDISAMBIGUATOR template
# else
#  define SWIGTEMPLATEDISAMBIGUATOR
# endif
#endif

/* inline attribute */
#ifndef SWIGINLINE
# if defined(__cplusplus) || (defined(__GNUC__) && !defined(__STRICT_ANSI__))
#   define SWIGINLINE inline
# else
#   define SWIGINLINE
# endif
#endif

/* attribute recognised by some compilers to avoid 'unused' warnings */
#ifndef SWIGUNUSED
# if defined(__GNUC__)
#   if !(defined(__cplusplus)) || (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4))
#     define SWIGUNUSED __attribute__ ((__unused__))
#   else
#     define SWIGUNUSED
#   endif
# elif defined(__ICC)
#   define SWIGUNUSED __attribute__ ((__unused__))
# else
#   define SWIGUNUSED
# endif
#endif

#ifndef SWIG_MSC_UNSUPPRESS_4505
# if defined(_MSC_VER)
#   pragma warning(disable : 4505) /* unreferenced local function has been removed */
# endif
#endif

#ifndef SWIGUNUSEDPARM
# ifdef __cplusplus
#   define SWIGUNUSEDPARM(p)
# else
#   define SWIGUNUSEDPARM(p) p SWIGUNUSED
# endif
#endif

/* internal SWIG method */
#ifndef SWIGINTERN
# define SWIGINTERN static SWIGUNUSED
#endif

/* internal inline SWIG method */
#ifndef SWIGINTERNINLINE
# define SWIGINTERNINLINE SWIGINTERN SWIGINLINE
#endif

/* qualifier for exported *const* global data variables*/
#ifndef SWIGEXTERN
# ifdef __cplusplus
#   define SWIGEXTERN extern
# else
#   define SWIGEXTERN
# endif
#endif

/* exporting methods */
#if defined(__GNUC__)
#  if (__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 4)
#    ifndef GCC_HASCLASSVISIBILITY
#      define GCC_HASCLASSVISIBILITY
#    endif
#  endif
#endif

#ifndef SWIGEXPORT
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   if defined(STATIC_LINKED)
#     define SWIGEXPORT
#   else
#     define SWIGEXPORT __declspec(dllexport)
#   endif
# else
#   if defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
#     define SWIGEXPORT __attribute__ ((visibility("default")))
#   else
#     define SWIGEXPORT
#   endif
# endif
#endif

/* calling conventions for Windows */
#ifndef SWIGSTDCALL
# if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
#   define SWIGSTDCALL __stdcall
# else
#   define SWIGSTDCALL
# endif
#endif

/* Deal with Microsoft's attempt at deprecating C standard runtime functions */
#if !defined(SWIG_NO_CRT_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_CRT_SECURE_NO_DEPRECATE)
# define _CRT_SECURE_NO_DEPRECATE
#endif

/* Deal with Microsoft's attempt at deprecating methods in the standard C++ library */
#if !defined(SWIG_NO_SCL_SECURE_NO_DEPRECATE) && defined(_MSC_VER) && !defined(_SCL_SECURE_NO_DEPRECATE)
# define _SCL_SECURE_NO_DEPRECATE
#endif

/* Deal with Apple's deprecated 'AssertMacros.h' from Carbon-framework */
#if defined(__APPLE__) && !defined(__ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES)
# define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#endif

/* Intel's compiler complains if a variable which was never initialised is
 * cast to void, which is a common idiom which we use to indicate that we
 * are aware a variable isn't used.  So we just silence that warning.
 * See: https://github.com/swig/swig/issues/192 for more discussion.
 */
#ifdef __INTEL_COMPILER
# pragma warning disable 592
#endif

/*  Errors in SWIG */
#define  SWIG_UnknownError    	   -1
#define  SWIG_IOError        	   -2
#define  SWIG_RuntimeError   	   -3
#define  SWIG_IndexError     	   -4
#define  SWIG_TypeError      	   -5
#define  SWIG_DivisionByZero 	   -6
#define  SWIG_OverflowError  	   -7
#define  SWIG_SyntaxError    	   -8
#define  SWIG_ValueError     	   -9
#define  SWIG_SystemError    	   -10
#define  SWIG_AttributeError 	   -11
#define  SWIG_MemoryError    	   -12
#define  SWIG_NullReferenceError   -13




#include <assert.h>
#define SWIG_exception_impl(DECL, CODE, MSG, RETURNNULL) \
 { printf("In " DECL ": " MSG); assert(0); RETURNNULL; }


#include <stdio.h>
#if defined(_MSC_VER) || defined(__BORLANDC__) || defined(_WATCOM)
# ifndef snprintf
#  define snprintf _snprintf
# endif
#endif


/* Support for the `contract` feature.
 *
 * Note that RETURNNULL is first because it's inserted via a 'Replaceall' in
 * the fortran.cxx file.
 */
#define SWIG_contract_assert(RETURNNULL, EXPR, MSG) \
 if (!(EXPR)) { SWIG_exception_impl("$decl", SWIG_ValueError, MSG, RETURNNULL); } 


#define SWIGVERSION 0x040000 
#define SWIG_VERSION SWIGVERSION


#define SWIG_as_voidptr(a) (void *)((const void *)(a)) 
#define SWIG_as_voidptrptr(a) ((void)SWIG_as_voidptr(*a),(void**)(a)) 


#include "kinsol/kinsol.h"
#include "kinsol/kinsol_bbdpre.h"
#include "kinsol/kinsol_ls.h"


#include <stdlib.h>
#ifdef _MSC_VER
# ifndef strtoull
#  define strtoull _strtoui64
# endif
# ifndef strtoll
#  define strtoll _strtoi64
# endif
#endif


typedef struct {
    void* data;
    size_t size;
} SwigArrayWrapper;


SWIGINTERN SwigArrayWrapper SwigArrayWrapper_uninitialized() {
  SwigArrayWrapper result;
  result.data = NULL;
  result.size = 0;
  return result;
}


#include <string.h>

SWIGEXPORT void * _wrap_FKINCreate(void *farg1) {
  void * fresult ;
  SUNContext arg1 = (SUNContext) 0 ;
  void *result = 0 ;
  
  arg1 = (SUNContext)(farg1);
  result = (void *)KINCreate(arg1);
  fresult = result;
  return fresult;
}


SWIGEXPORT int _wrap_FKINInit(void *farg1, KINSysFn farg2, N_Vector farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINSysFn arg2 = (KINSysFn) 0 ;
  N_Vector arg3 = (N_Vector) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINSysFn)(farg2);
  arg3 = (N_Vector)(farg3);
  result = (int)KINInit(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSol(void *farg1, N_Vector farg2, int const *farg3, N_Vector farg4, N_Vector farg5) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  N_Vector arg2 = (N_Vector) 0 ;
  int arg3 ;
  N_Vector arg4 = (N_Vector) 0 ;
  N_Vector arg5 = (N_Vector) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (N_Vector)(farg2);
  arg3 = (int)(*farg3);
  arg4 = (N_Vector)(farg4);
  arg5 = (N_Vector)(farg5);
  result = (int)KINSol(arg1,arg2,arg3,arg4,arg5);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetUserData(void *farg1, void *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  void *arg2 = (void *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (void *)(farg2);
  result = (int)KINSetUserData(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetDamping(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetDamping(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetMAA(void *farg1, long const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long)(*farg2);
  result = (int)KINSetMAA(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetOrthAA(void *farg1, int const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  int arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (int)(*farg2);
  result = (int)KINSetOrthAA(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetDelayAA(void *farg1, long const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long)(*farg2);
  result = (int)KINSetDelayAA(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetDampingAA(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetDampingAA(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetReturnNewest(void *farg1, int const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  int arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (int)(*farg2);
  result = (int)KINSetReturnNewest(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetNumMaxIters(void *farg1, long const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long)(*farg2);
  result = (int)KINSetNumMaxIters(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetNoInitSetup(void *farg1, int const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  int arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (int)(*farg2);
  result = (int)KINSetNoInitSetup(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetNoResMon(void *farg1, int const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  int arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (int)(*farg2);
  result = (int)KINSetNoResMon(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetMaxSetupCalls(void *farg1, long const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long)(*farg2);
  result = (int)KINSetMaxSetupCalls(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetMaxSubSetupCalls(void *farg1, long const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long)(*farg2);
  result = (int)KINSetMaxSubSetupCalls(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetEtaForm(void *farg1, int const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  int arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (int)(*farg2);
  result = (int)KINSetEtaForm(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetEtaConstValue(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetEtaConstValue(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetEtaParams(void *farg1, double const *farg2, double const *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  realtype arg3 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  arg3 = (realtype)(*farg3);
  result = (int)KINSetEtaParams(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetResMonParams(void *farg1, double const *farg2, double const *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  realtype arg3 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  arg3 = (realtype)(*farg3);
  result = (int)KINSetResMonParams(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetResMonConstValue(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetResMonConstValue(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetNoMinEps(void *farg1, int const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  int arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (int)(*farg2);
  result = (int)KINSetNoMinEps(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetMaxNewtonStep(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetMaxNewtonStep(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetMaxBetaFails(void *farg1, long const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long)(*farg2);
  result = (int)KINSetMaxBetaFails(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetRelErrFunc(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetRelErrFunc(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetFuncNormTol(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetFuncNormTol(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetScaledStepTol(void *farg1, double const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype)(*farg2);
  result = (int)KINSetScaledStepTol(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetConstraints(void *farg1, N_Vector farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  N_Vector arg2 = (N_Vector) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (N_Vector)(farg2);
  result = (int)KINSetConstraints(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetSysFunc(void *farg1, KINSysFn farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINSysFn arg2 = (KINSysFn) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINSysFn)(farg2);
  result = (int)KINSetSysFunc(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetErrHandlerFn(void *farg1, KINErrHandlerFn farg2, void *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINErrHandlerFn arg2 = (KINErrHandlerFn) 0 ;
  void *arg3 = (void *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINErrHandlerFn)(farg2);
  arg3 = (void *)(farg3);
  result = (int)KINSetErrHandlerFn(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetErrFile(void *farg1, void *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  FILE *arg2 = (FILE *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (FILE *)(farg2);
  result = (int)KINSetErrFile(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetInfoHandlerFn(void *farg1, KINInfoHandlerFn farg2, void *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINInfoHandlerFn arg2 = (KINInfoHandlerFn) 0 ;
  void *arg3 = (void *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINInfoHandlerFn)(farg2);
  arg3 = (void *)(farg3);
  result = (int)KINSetInfoHandlerFn(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetInfoFile(void *farg1, void *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  FILE *arg2 = (FILE *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (FILE *)(farg2);
  result = (int)KINSetInfoFile(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetPrintLevel(void *farg1, int const *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  int arg2 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (int)(*farg2);
  result = (int)KINSetPrintLevel(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetDebugFile(void *farg1, void *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  FILE *arg2 = (FILE *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (FILE *)(farg2);
  result = (int)KINSetDebugFile(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetWorkSpace(void *farg1, long *farg2, long *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  long *arg3 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  arg3 = (long *)(farg3);
  result = (int)KINGetWorkSpace(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumNonlinSolvIters(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumNonlinSolvIters(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumFuncEvals(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumFuncEvals(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumBetaCondFails(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumBetaCondFails(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumBacktrackOps(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumBacktrackOps(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetFuncNorm(void *farg1, double *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype *arg2 = (realtype *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype *)(farg2);
  result = (int)KINGetFuncNorm(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetStepLength(void *farg1, double *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  realtype *arg2 = (realtype *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (realtype *)(farg2);
  result = (int)KINGetStepLength(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetUserData(void *farg1, void *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  void **arg2 = (void **) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (void **)(farg2);
  result = (int)KINGetUserData(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINPrintAllStats(void *farg1, void *farg2, int const *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  FILE *arg2 = (FILE *) 0 ;
  SUNOutputFormat arg3 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (FILE *)(farg2);
  arg3 = (SUNOutputFormat)(*farg3);
  result = (int)KINPrintAllStats(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT SwigArrayWrapper _wrap_FKINGetReturnFlagName(long const *farg1) {
  SwigArrayWrapper fresult ;
  long arg1 ;
  char *result = 0 ;
  
  arg1 = (long)(*farg1);
  result = (char *)KINGetReturnFlagName(arg1);
  fresult.size = strlen((const char*)(result));
  fresult.data = (char *)(result);
  return fresult;
}


SWIGEXPORT void _wrap_FKINFree(void *farg1) {
  void **arg1 = (void **) 0 ;
  
  arg1 = (void **)(farg1);
  KINFree(arg1);
}


SWIGEXPORT int _wrap_FKINSetJacTimesVecSysFn(void *farg1, KINSysFn farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINSysFn arg2 = (KINSysFn) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINSysFn)(farg2);
  result = (int)KINSetJacTimesVecSysFn(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINBBDPrecInit(void *farg1, int64_t const *farg2, int64_t const *farg3, int64_t const *farg4, int64_t const *farg5, int64_t const *farg6, double const *farg7, KINBBDLocalFn farg8, KINBBDCommFn farg9) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  sunindextype arg2 ;
  sunindextype arg3 ;
  sunindextype arg4 ;
  sunindextype arg5 ;
  sunindextype arg6 ;
  realtype arg7 ;
  KINBBDLocalFn arg8 = (KINBBDLocalFn) 0 ;
  KINBBDCommFn arg9 = (KINBBDCommFn) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (sunindextype)(*farg2);
  arg3 = (sunindextype)(*farg3);
  arg4 = (sunindextype)(*farg4);
  arg5 = (sunindextype)(*farg5);
  arg6 = (sunindextype)(*farg6);
  arg7 = (realtype)(*farg7);
  arg8 = (KINBBDLocalFn)(farg8);
  arg9 = (KINBBDCommFn)(farg9);
  result = (int)KINBBDPrecInit(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINBBDPrecGetWorkSpace(void *farg1, long *farg2, long *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  long *arg3 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  arg3 = (long *)(farg3);
  result = (int)KINBBDPrecGetWorkSpace(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINBBDPrecGetNumGfnEvals(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINBBDPrecGetNumGfnEvals(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetLinearSolver(void *farg1, SUNLinearSolver farg2, SUNMatrix farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  SUNLinearSolver arg2 = (SUNLinearSolver) 0 ;
  SUNMatrix arg3 = (SUNMatrix) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (SUNLinearSolver)(farg2);
  arg3 = (SUNMatrix)(farg3);
  result = (int)KINSetLinearSolver(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetJacFn(void *farg1, KINLsJacFn farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINLsJacFn arg2 = (KINLsJacFn) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINLsJacFn)(farg2);
  result = (int)KINSetJacFn(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetPreconditioner(void *farg1, KINLsPrecSetupFn farg2, KINLsPrecSolveFn farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINLsPrecSetupFn arg2 = (KINLsPrecSetupFn) 0 ;
  KINLsPrecSolveFn arg3 = (KINLsPrecSolveFn) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINLsPrecSetupFn)(farg2);
  arg3 = (KINLsPrecSolveFn)(farg3);
  result = (int)KINSetPreconditioner(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINSetJacTimesVecFn(void *farg1, KINLsJacTimesVecFn farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  KINLsJacTimesVecFn arg2 = (KINLsJacTimesVecFn) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (KINLsJacTimesVecFn)(farg2);
  result = (int)KINSetJacTimesVecFn(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetJac(void *farg1, void *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  SUNMatrix *arg2 = (SUNMatrix *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (SUNMatrix *)(farg2);
  result = (int)KINGetJac(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetJacNumIters(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetJacNumIters(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetLinWorkSpace(void *farg1, long *farg2, long *farg3) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  long *arg3 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  arg3 = (long *)(farg3);
  result = (int)KINGetLinWorkSpace(arg1,arg2,arg3);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumJacEvals(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumJacEvals(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumLinFuncEvals(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumLinFuncEvals(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumPrecEvals(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumPrecEvals(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumPrecSolves(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumPrecSolves(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumLinIters(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumLinIters(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumLinConvFails(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumLinConvFails(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetNumJtimesEvals(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetNumJtimesEvals(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT int _wrap_FKINGetLastLinFlag(void *farg1, long *farg2) {
  int fresult ;
  void *arg1 = (void *) 0 ;
  long *arg2 = (long *) 0 ;
  int result;
  
  arg1 = (void *)(farg1);
  arg2 = (long *)(farg2);
  result = (int)KINGetLastLinFlag(arg1,arg2);
  fresult = (int)(result);
  return fresult;
}


SWIGEXPORT SwigArrayWrapper _wrap_FKINGetLinReturnFlagName(long const *farg1) {
  SwigArrayWrapper fresult ;
  long arg1 ;
  char *result = 0 ;
  
  arg1 = (long)(*farg1);
  result = (char *)KINGetLinReturnFlagName(arg1);
  fresult.size = strlen((const char*)(result));
  fresult.data = (char *)(result);
  return fresult;
}



