// ---------------------------------------------------------------
// Programmer: Cody J. Balos @ LLNL
// ---------------------------------------------------------------
// SUNDIALS Copyright Start
// Copyright (c) 2002-2019, Lawrence Livermore National Security
// and Southern Methodist University.
// All rights reserved.
//
// See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
// SUNDIALS Copyright End
// ---------------------------------------------------------------
// Swig interface file
// ---------------------------------------------------------------

%module kinsol

%include "stdint.i"

// ----------------------------------
// numpy swig setup stuff.
// This must only happen in one file.
// ----------------------------------

%{
#define SWIG_FILE_WITH_INIT
%}
%include "numpy.i"
%init %{
import_array();
%}

// -------------------------------
// Bring in shared sundials stuff.
// -------------------------------

%include "../sundials/sundials.i"

// ---------------------
// KINSOL specific stuff
// ---------------------

%{
#include "kinsol/kinsol.h"
#include "kinsol/kinsol_py.h"
#include "kinsol/kinsol_bbdpre.h"
#include "kinsol/kinsol_ls.h"
%}

// Apply typemap for Get functions that use an argout variable
%typemap(in, numinputs=0) realtype* (realtype temp) {
  $1 = &temp;
}
%typemap(argout) realtype* {
  $result = SWIG_Python_AppendOutput($result, PyFloat_FromDouble(*$1));
}
%typemap(in, numinputs=0) long* (long temp) {
  $1 = &temp;
}
%typemap(argout) long* {
  $result = SWIG_Python_AppendOutput($result, PyLong_FromLong(*$1));
}

// We need this typemap for KINSetUserData otherwise the type-safety check
// that swig does will fail. One potential problem with this is that the
// user data (a python) object could go out of scope and be deleted.
// The user must make sure that the object is not out of scope before
// KINSOL is done currently.
%typemap(in) (void *user_data) {
  $1 = (void *) $input;
}
%typemap(in) (void *eh_data) {
  $1 = (void *) $input;
}

// Since KINCreate returns a void* not void**, we have
// to get the address of the kinmem passed into KINFree.
// This typemap extracts the void* kinmem and then takes
// it address.
%typemap(in) (void **kinmem) {
  void* argp1 = 0;
  int res1 = 0;
  res1 = SWIG_ConvertPtr(swig_obj[0], &argp1, SWIGTYPE_p_void, 0 | 0 );
  if (!SWIG_IsOK(res1)) {
    SWIG_exception_fail(SWIG_ArgError(res1), "in method '" "KINFree" "', argument " "1"" of type '" "void **""'");
  }
  $1 = &argp1;
}

// A typemap for a callback, it expects the argument to be an integer
// whose value is the address of an appropriate callback function
%define %callback_function(name)
%typemap(in) name {
  $1 = (name)PyLong_AsVoidPtr($input);
}
%enddef

%include "kinsol_callbacks.i"

// Process definitions from these files
%include "kinsol/kinsol.h"
%include "kinsol/kinsol_py.h"
%include "kinsol/kinsol_bbdpre.h"
%include "kinsol/kinsol_ls.h"

%pythoncode
%{

import ctypes

# We provide the ctypes for all the callback functions in KINSol here as
# a convenience to our users. They could always define it themselves too.
class cfunctypes():
  KINSysFn = [ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.py_object),
               _kinsol.KINPyRegister_KINPySysFn]

  KINErrHandlerFn = [ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.py_object),
                     _kinsol.KINPyRegister_KINPyErrHandlerFn]



def RegisterFn(py_callback, py_callback_tuple):
  py_callback_type, kinpy_register_fn = py_callback_tuple
  f_in = py_callback_type(py_callback)
  f_in_ptr = ctypes.cast(f_in, ctypes.c_void_p).value
  try:
    return kinpy_register_fn(f_in_ptr)
  except:
    raise ValueError("Unknown callback function encountered")

def RegisterNumbaFn(py_callback, py_callback_tuple):
  _, kinpy_register_fn = py_callback_tuple
  f_in = py_callback.ctypes
  f_in_ptr = ctypes.cast(f_in, ctypes.c_void_p).value
  try:
    return kinpy_register_fn(f_in_ptr)
  except:
    raise ValueError("Unknown callback function encountered")

%}
