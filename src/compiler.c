/*****************************************************************************
 *
 *  compiler.c
 *
 *  Provide some details of the current compiler.
 *
 *  Edinburgh Soft Matter and Statistical Phyiscs Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "compiler.h"

/*****************************************************************************
 *
 *  compiler_id
 *
 *  Set compiler id; return 0 on success.
 *
 *****************************************************************************/

int compiler_id(compiler_info_t * compiler) {

  int ierr = -1;

  assert(compiler);
  
#ifdef __cplusplus
  compiler->cplusplus = __cplusplus;
#endif
  
  /* Positive identifications return immediately. Many compilers define one or
     both of __GNUC__ and __clang__, so discretion is required. */

#ifdef __cray__
  /* Clay clang */
  compiler->major = __cray_major__;
  compiler->minor = __cray_minor__;
  compiler->patchlevel = __cray_patchlevel__;
  strncpy(compiler->version, __VERSION__, strnlen(__VERSION__, BUFSIZ-1));
  sprintf(compiler->name, "%s", "Cray Clang");
  return 0;
#endif

#ifdef _CRAYC
  /* Cray "classic" */
  compiler->major = _RELEASE;
  compiler->minor = _RELEASE_MINOR;
  compiler->patchlevel = 0; /* Not provided */
  {
    int len = strnlen(_RELEASE_STRING, BUFSIZ-1);
    strncpy(compiler->version, _RELEASE_STRING, len);
  }
  sprintf(compiler->name, "%s", "Cray Classic");
  return 0;
#endif

#ifdef __INTEL_COMPILER

  /* Intel compiler versioning seems a movable feast... use: */
  compiler->major = __INTEL_COMPILER/100;
  compiler->minor = __INTEL_COMPILER - 100*compiler->major;
  compiler->patchlevel = __INTEL_COMPILER_UPDATE;
  strncpy(compiler->version, __VERSION__, strnlen(__VERSION__, BUFSIZ-1));
  sprintf(compiler->name, "%s", "Intel");
  return 0;
#endif

#ifdef __NVCC__
  compiler->major = __CUDACC_VER_MAJOR__;
  compiler->minor = __CUDACC_VER_MINOR__;
  compiler->patchlevel = __CUDACC_VER_BUILD__;
  sprintf(compiler->version, "%s", "null"); /* None provided */
  sprintf(compiler->name, "%s", "NVIDIA nvcc");
  /* Include __CUDA_ARCH__ */
  return 0;
#endif

#ifdef __clang__
  /* Load details */
  compiler->major = __clang_major__;
  compiler->minor = __clang_minor__;
  compiler->patchlevel = __clang_patchlevel__;
  {
    int len = strnlen(__clang_version__, BUFSIZ-1);
    strncpy(compiler->version, __clang_version__, len);
  }
  sprintf(compiler->name, "%s", "Clang");

  /* CASE */
  /* AMD/AOCC defines no specific macros. */
  /* __clang_version__ 10.0.0 (CLANG: AOCC_2.2.0-Build#93 2020_06_25) */
  /* Could orrow the GPU _VERSION_, if available, as more informative. */

  return 0;
#endif
  
#ifdef __GNUC__
  /* Load details. */
  compiler->major = __GNUC__;
  compiler->minor = __GNUC_MINOR__;
  compiler->patchlevel = __GNUC_PATCHLEVEL__;
  strncpy(compiler->version, __VERSION__, 1 + strlen(__VERSION__));
  sprintf(compiler->name, "%s", "Gnu");
  ierr = 0;

  /* CASE */
  /* __VERSION__ AOCC.LLVM.2.1.0.B1030.2019_11_12 */
  /* __VERSION__ AMD Clang 10.0.0 (CLANG: AOCC_2.2.0-Build#93 2020_06_25) */
#endif
  
  return ierr;
}
