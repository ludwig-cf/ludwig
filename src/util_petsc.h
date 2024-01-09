/*****************************************************************************
 *
 *  util_petsc.h
 *
 *  A stub interface to deal with Petsc availability.
 *  The idea is to use PetscInitialised() [sic].
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_PETSC_H
#define LUDWIG_UTIL_PETSC_H

#ifdef PETSC
#include "petscsys.h"
#else

/* Stub replacements for ... */

typedef enum {PETSC_FALSE, PETSC_TRUE} PetscBool;
typedef int PetscErrorCode;

PetscErrorCode PetscInitialize(int * argc, char *** argv, const char * file,
			       const char * help);
PetscErrorCode PetscFinalize(void);
PetscErrorCode PetscInitialized(PetscBool * isInitialised);
#endif

int PetscInitialised(int * isInitialised);

#endif
