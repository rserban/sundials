! This file was automatically generated by SWIG (http://www.swig.org).
! Version 4.0.0
!
! Do not make changes to this file unless you know what you are doing--modify
! the SWIG interface file instead.

! ---------------------------------------------------------------
! Programmer(s): Auto-generated by swig.
! ---------------------------------------------------------------
! SUNDIALS Copyright Start
! Copyright (c) 2002-2022, Lawrence Livermore National Security
! and Southern Methodist University.
! All rights reserved.
!
! See the top-level LICENSE and NOTICE files for details.
!
! SPDX-License-Identifier: BSD-3-Clause
! SUNDIALS Copyright End
! ---------------------------------------------------------------

module fsunmatrix_band_mod
 use, intrinsic :: ISO_C_BINDING
 use fsundials_matrix_mod
 use fsundials_types_mod
 use fsundials_context_mod
 use fsundials_nvector_mod
 use fsundials_context_mod
 use fsundials_types_mod
 implicit none
 private

 ! DECLARATION CONSTRUCTS
 public :: FSUNBandMatrix
 public :: FSUNBandMatrixStorage
 public :: FSUNBandMatrix_Print
 public :: FSUNBandMatrix_Rows
 public :: FSUNBandMatrix_Columns
 public :: FSUNBandMatrix_LowerBandwidth
 public :: FSUNBandMatrix_UpperBandwidth
 public :: FSUNBandMatrix_StoredUpperBandwidth
 public :: FSUNBandMatrix_LDim
 public :: FSUNBandMatrix_LData
 public :: FSUNBandMatrix_Cols
 public :: FSUNMatGetID_Band
 public :: FSUNMatClone_Band
 public :: FSUNMatDestroy_Band
 public :: FSUNMatZero_Band
 public :: FSUNMatCopy_Band
 public :: FSUNMatScaleAdd_Band
 public :: FSUNMatScaleAddI_Band
 public :: FSUNMatMatvec_Band
 public :: FSUNMatSpace_Band

 public :: FSUNBandMatrix_Data
 public :: FSUNBandMatrix_Column


! WRAPPER DECLARATIONS
interface
function swigc_FSUNBandMatrix(farg1, farg2, farg3, farg4) &
bind(C, name="_wrap_FSUNBandMatrix") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T), intent(in) :: farg1
integer(C_INT64_T), intent(in) :: farg2
integer(C_INT64_T), intent(in) :: farg3
type(C_PTR), value :: farg4
type(C_PTR) :: fresult
end function

function swigc_FSUNBandMatrixStorage(farg1, farg2, farg3, farg4, farg5) &
bind(C, name="_wrap_FSUNBandMatrixStorage") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T), intent(in) :: farg1
integer(C_INT64_T), intent(in) :: farg2
integer(C_INT64_T), intent(in) :: farg3
integer(C_INT64_T), intent(in) :: farg4
type(C_PTR), value :: farg5
type(C_PTR) :: fresult
end function

subroutine swigc_FSUNBandMatrix_Print(farg1, farg2) &
bind(C, name="_wrap_FSUNBandMatrix_Print")
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
type(C_PTR), value :: farg2
end subroutine

function swigc_FSUNBandMatrix_Rows(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_Rows") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T) :: fresult
end function

function swigc_FSUNBandMatrix_Columns(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_Columns") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T) :: fresult
end function

function swigc_FSUNBandMatrix_LowerBandwidth(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_LowerBandwidth") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T) :: fresult
end function

function swigc_FSUNBandMatrix_UpperBandwidth(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_UpperBandwidth") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T) :: fresult
end function

function swigc_FSUNBandMatrix_StoredUpperBandwidth(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_StoredUpperBandwidth") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T) :: fresult
end function

function swigc_FSUNBandMatrix_LDim(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_LDim") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T) :: fresult
end function

function swigc_FSUNBandMatrix_LData(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_LData") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T) :: fresult
end function

function swigc_FSUNBandMatrix_Cols(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_Cols") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
type(C_PTR) :: fresult
end function

function swigc_FSUNMatGetID_Band(farg1) &
bind(C, name="_wrap_FSUNMatGetID_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT) :: fresult
end function

function swigc_FSUNMatClone_Band(farg1) &
bind(C, name="_wrap_FSUNMatClone_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
type(C_PTR) :: fresult
end function

subroutine swigc_FSUNMatDestroy_Band(farg1) &
bind(C, name="_wrap_FSUNMatDestroy_Band")
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
end subroutine

function swigc_FSUNMatZero_Band(farg1) &
bind(C, name="_wrap_FSUNMatZero_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT) :: fresult
end function

function swigc_FSUNMatCopy_Band(farg1, farg2) &
bind(C, name="_wrap_FSUNMatCopy_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
type(C_PTR), value :: farg2
integer(C_INT) :: fresult
end function

function swigc_FSUNMatScaleAdd_Band(farg1, farg2, farg3) &
bind(C, name="_wrap_FSUNMatScaleAdd_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
real(C_DOUBLE), intent(in) :: farg1
type(C_PTR), value :: farg2
type(C_PTR), value :: farg3
integer(C_INT) :: fresult
end function

function swigc_FSUNMatScaleAddI_Band(farg1, farg2) &
bind(C, name="_wrap_FSUNMatScaleAddI_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
real(C_DOUBLE), intent(in) :: farg1
type(C_PTR), value :: farg2
integer(C_INT) :: fresult
end function

function swigc_FSUNMatMatvec_Band(farg1, farg2, farg3) &
bind(C, name="_wrap_FSUNMatMatvec_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
type(C_PTR), value :: farg2
type(C_PTR), value :: farg3
integer(C_INT) :: fresult
end function

function swigc_FSUNMatSpace_Band(farg1, farg2, farg3) &
bind(C, name="_wrap_FSUNMatSpace_Band") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
type(C_PTR), value :: farg2
type(C_PTR), value :: farg3
integer(C_INT) :: fresult
end function


function swigc_FSUNBandMatrix_Data(farg1) &
bind(C, name="_wrap_FSUNBandMatrix_Data") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
type(C_PTR) :: fresult
end function

function swigc_FSUNBandMatrix_Column(farg1, farg2) &
bind(C, name="_wrap_FSUNBandMatrix_Column") &
result(fresult)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), value :: farg1
integer(C_INT64_T), intent(in) :: farg2
type(C_PTR) :: fresult
end function

end interface


contains
 ! MODULE SUBPROGRAMS
function FSUNBandMatrix(n, mu, ml, sunctx) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
type(SUNMatrix), pointer :: swig_result
integer(C_INT64_T), intent(in) :: n
integer(C_INT64_T), intent(in) :: mu
integer(C_INT64_T), intent(in) :: ml
type(C_PTR) :: sunctx
type(C_PTR) :: fresult 
integer(C_INT64_T) :: farg1 
integer(C_INT64_T) :: farg2 
integer(C_INT64_T) :: farg3 
type(C_PTR) :: farg4 

farg1 = n
farg2 = mu
farg3 = ml
farg4 = sunctx
fresult = swigc_FSUNBandMatrix(farg1, farg2, farg3, farg4)
call c_f_pointer(fresult, swig_result)
end function

function FSUNBandMatrixStorage(n, mu, ml, smu, sunctx) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
type(SUNMatrix), pointer :: swig_result
integer(C_INT64_T), intent(in) :: n
integer(C_INT64_T), intent(in) :: mu
integer(C_INT64_T), intent(in) :: ml
integer(C_INT64_T), intent(in) :: smu
type(C_PTR) :: sunctx
type(C_PTR) :: fresult 
integer(C_INT64_T) :: farg1 
integer(C_INT64_T) :: farg2 
integer(C_INT64_T) :: farg3 
integer(C_INT64_T) :: farg4 
type(C_PTR) :: farg5 

farg1 = n
farg2 = mu
farg3 = ml
farg4 = smu
farg5 = sunctx
fresult = swigc_FSUNBandMatrixStorage(farg1, farg2, farg3, farg4, farg5)
call c_f_pointer(fresult, swig_result)
end function

subroutine FSUNBandMatrix_Print(a, outfile)
use, intrinsic :: ISO_C_BINDING
type(SUNMatrix), target, intent(inout) :: a
type(C_PTR) :: outfile
type(C_PTR) :: farg1 
type(C_PTR) :: farg2 

farg1 = c_loc(a)
farg2 = outfile
call swigc_FSUNBandMatrix_Print(farg1, farg2)
end subroutine

function FSUNBandMatrix_Rows(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_Rows(farg1)
swig_result = fresult
end function

function FSUNBandMatrix_Columns(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_Columns(farg1)
swig_result = fresult
end function

function FSUNBandMatrix_LowerBandwidth(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_LowerBandwidth(farg1)
swig_result = fresult
end function

function FSUNBandMatrix_UpperBandwidth(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_UpperBandwidth(farg1)
swig_result = fresult
end function

function FSUNBandMatrix_StoredUpperBandwidth(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_StoredUpperBandwidth(farg1)
swig_result = fresult
end function

function FSUNBandMatrix_LDim(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_LDim(farg1)
swig_result = fresult
end function

function FSUNBandMatrix_LData(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT64_T) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_LData(farg1)
swig_result = fresult
end function

function FSUNBandMatrix_Cols(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
type(C_PTR), pointer :: swig_result
type(SUNMatrix), target, intent(inout) :: a
type(C_PTR) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_Cols(farg1)
call c_f_pointer(fresult, swig_result)
end function

function FSUNMatGetID_Band(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(SUNMatrix_ID) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNMatGetID_Band(farg1)
swig_result = fresult
end function

function FSUNMatClone_Band(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
type(SUNMatrix), pointer :: swig_result
type(SUNMatrix), target, intent(inout) :: a
type(C_PTR) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNMatClone_Band(farg1)
call c_f_pointer(fresult, swig_result)
end function

subroutine FSUNMatDestroy_Band(a)
use, intrinsic :: ISO_C_BINDING
type(SUNMatrix), target, intent(inout) :: a
type(C_PTR) :: farg1 

farg1 = c_loc(a)
call swigc_FSUNMatDestroy_Band(farg1)
end subroutine

function FSUNMatZero_Band(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNMatZero_Band(farg1)
swig_result = fresult
end function

function FSUNMatCopy_Band(a, b) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
type(SUNMatrix), target, intent(inout) :: b
integer(C_INT) :: fresult 
type(C_PTR) :: farg1 
type(C_PTR) :: farg2 

farg1 = c_loc(a)
farg2 = c_loc(b)
fresult = swigc_FSUNMatCopy_Band(farg1, farg2)
swig_result = fresult
end function

function FSUNMatScaleAdd_Band(c, a, b) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT) :: swig_result
real(C_DOUBLE), intent(in) :: c
type(SUNMatrix), target, intent(inout) :: a
type(SUNMatrix), target, intent(inout) :: b
integer(C_INT) :: fresult 
real(C_DOUBLE) :: farg1 
type(C_PTR) :: farg2 
type(C_PTR) :: farg3 

farg1 = c
farg2 = c_loc(a)
farg3 = c_loc(b)
fresult = swigc_FSUNMatScaleAdd_Band(farg1, farg2, farg3)
swig_result = fresult
end function

function FSUNMatScaleAddI_Band(c, a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT) :: swig_result
real(C_DOUBLE), intent(in) :: c
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT) :: fresult 
real(C_DOUBLE) :: farg1 
type(C_PTR) :: farg2 

farg1 = c
farg2 = c_loc(a)
fresult = swigc_FSUNMatScaleAddI_Band(farg1, farg2)
swig_result = fresult
end function

function FSUNMatMatvec_Band(a, x, y) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
type(N_Vector), target, intent(inout) :: x
type(N_Vector), target, intent(inout) :: y
integer(C_INT) :: fresult 
type(C_PTR) :: farg1 
type(C_PTR) :: farg2 
type(C_PTR) :: farg3 

farg1 = c_loc(a)
farg2 = c_loc(x)
farg3 = c_loc(y)
fresult = swigc_FSUNMatMatvec_Band(farg1, farg2, farg3)
swig_result = fresult
end function

function FSUNMatSpace_Band(a, lenrw, leniw) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
integer(C_INT) :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_LONG), dimension(*), target, intent(inout) :: lenrw
integer(C_LONG), dimension(*), target, intent(inout) :: leniw
integer(C_INT) :: fresult 
type(C_PTR) :: farg1 
type(C_PTR) :: farg2 
type(C_PTR) :: farg3 

farg1 = c_loc(a)
farg2 = c_loc(lenrw(1))
farg3 = c_loc(leniw(1))
fresult = swigc_FSUNMatSpace_Band(farg1, farg2, farg3)
swig_result = fresult
end function


function FSUNBandMatrix_Data(a) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
real(C_DOUBLE), dimension(:), pointer :: swig_result
type(SUNMatrix), target, intent(inout) :: a
type(C_PTR) :: fresult 
type(C_PTR) :: farg1 

farg1 = c_loc(a)
fresult = swigc_FSUNBandMatrix_Data(farg1)
call c_f_pointer(fresult, swig_result, [FSUNBandMatrix_LData(a)])
end function

function FSUNBandMatrix_Column(a, j) &
result(swig_result)
use, intrinsic :: ISO_C_BINDING
real(C_DOUBLE), dimension(:), pointer :: swig_result
type(SUNMatrix), target, intent(inout) :: a
integer(C_INT64_T), intent(in) :: j
type(C_PTR) :: fresult 
type(C_PTR) :: farg1 
integer(C_INT64_T) :: farg2 

farg1 = c_loc(a)
farg2 = j
fresult = swigc_FSUNBandMatrix_Column(farg1, farg2)
call c_f_pointer(fresult, swig_result, [FSUNBandMatrix_UpperBandwidth(a)+FSUNBandMatrix_LowerBandwidth(a)+1])
end function


end module
