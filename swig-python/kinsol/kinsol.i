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

// KINInit cannot be called from Python.
// Instead, users should call KINInitPy.
// %ignore KINInit;

// We hijack KINSetUserData to pass out director class
// objects. So, hide the function from users.
// %ignore KINSetUserData;

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

// A typemap for the callback, it expects the argument to be an integer
// whose value is the address of an appropriate callback function
%typemap(in) KINPySysFn {
  $1 = (KINPySysFn)PyLong_AsVoidPtr($input);
}

// We need this typemap for KINSetUserData otherwise the type-safety check
// that swig does will fail. One potential problem with this is that the
// user data (a python) object could go out of scope and be deleted.
// The user must make sure that the object is not out of scope before
// KINSOL is done.
%typemap(in) (void *user_data) {
  $1 = (void *) $input;
};

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
  KINSysFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.py_object)

def RegisterFn(py_callback, py_callback_type):
  f_in = py_callback_type(py_callback)
  f_in_ptr = ctypes.cast(f_in, ctypes.c_void_p).value
  if py_callback_type == cfunctypes.KINSysFn:
    return _kinsol.KINPyRegisterKINPySysFn(f_in_ptr)

def RegisterNumbaFn(py_callback, py_callback_type):
  f = py_callback.ctypes
  f_ptr = ctypes.cast(f, ctypes.c_void_p).value

  if py_callback_type == cfunctypes.KINSysFn:
    return  _kinsol.KINPyRegisterKINPySysFn(f_ptr)
%}
