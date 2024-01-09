/*****************************************************************************
 *
 *  psi_petsc.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing Authors:
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PSI_SOLVER_PETSC_H
#define LUDWIG_PSI_SOLVER_PETSC_H

#include "psi_solver.h"

typedef struct psi_solver_petsc_s psi_solver_petsc_t;
typedef struct psi_solver_petsc_block_s psi_solver_petsc_block_t;

struct psi_solver_petsc_s {
  psi_solver_t super;                /* Superclass block */
  psi_t * psi;                       /* Retain a reference to psi_t */
  fe_t * fe;                         /* Free energy */
  var_epsilon_ft epsilon;            /* Variable dielectric model */
  psi_solver_petsc_block_t * block;  /* Opaque internal information. */
};

int psi_solver_petsc_create(psi_t * psi, psi_solver_petsc_t ** solver);
int psi_solver_petsc_free(psi_solver_petsc_t ** solver);
int psi_solver_petsc_solve(psi_solver_petsc_t * solver, int ntimestep);

int psi_solver_petsc_var_epsilon_create(psi_t * psi, var_epsilon_t epsilon,
					psi_solver_petsc_t ** solver);
int psi_solver_petsc_var_epsilon_solve(psi_solver_petsc_t * solver, int nt);

#endif

