/*****************************************************************************
 *
 *  util_petsc.c
 *
 *  A facade.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "util_petsc.h"

/*****************************************************************************
 *
 *  PetscInitialised
 *
 *  A slightly adapted version.
 *
 *****************************************************************************/

int PetscInitialised(int * isInitialised) {

  PetscBool havePetsc = PETSC_FALSE;
  PetscInitialized(&havePetsc);

  *isInitialised = 0;
  if (havePetsc == PETSC_TRUE) *isInitialised = 1;
  
  return 0;
}

#ifdef PETSC
/* Nothing more required. */
#else

/*****************************************************************************
 *
 *  PetscInitialize
 *
 *****************************************************************************/

PetscErrorCode PetscInitialize(int * argc, char *** argv, const char * file,
			       const char * help) {

  return 0;
}

/*****************************************************************************
 *
 *  PetscFinalize
 *
 *****************************************************************************/

PetscErrorCode PetscFinalize(void) {

  return 0;
}

/*****************************************************************************
 *
 *  PetscInitialized
 *
 *****************************************************************************/

PetscErrorCode PetscInitialized(PetscBool * isInitialised) {

  assert(isInitialised);

  *isInitialised = (PetscBool) 0; /* Always zero */

  return 0;
}

#endif
