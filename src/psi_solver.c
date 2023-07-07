/*****************************************************************************
 *
 *  psi_solver.c
 *
 *  Factory function for the solver object.
 *
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

/* Available implementations ... */
#include "psi_petsc.h"
#include "psi_sor.h"

/*****************************************************************************
 *
 *  psi_solver_create
 *
 *  Returns pointer to the abstract solver type.
 *
 *  We must allow that PETSc is not available, so the return value
 *  must be checked by the caller.
 *
 *****************************************************************************/

int psi_solver_create(psi_t * psi, psi_solver_t ** solver) {

  int ifail = 0;

  assert(solver && *solver == NULL);

  switch (psi->solver.psolver) {

  case (PSI_POISSON_SOLVER_PETSC):
    {
      psi_solver_petsc_t * petsc = NULL;
      ifail = psi_solver_petsc_create(psi, &petsc);
      if (ifail == 0) *solver = (psi_solver_t *) petsc;
    }
    break;

  case (PSI_POISSON_SOLVER_SOR):
    {
      psi_solver_sor_t * sor = NULL;
      ifail = psi_solver_sor_create(psi, &sor);
      if (ifail == 0) *solver = (psi_solver_t *) sor;
    }
    break;

  default:
    ifail = -1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_var_epsilon_create
 *
 *  The equivalent for the version allowing electric contrast.
 *  The same comments apply.
 *
 *****************************************************************************/

int psi_solver_var_epsilon_create(psi_t * psi, var_epsilon_t user,
				  psi_solver_t ** solver) {

  int ifail = 0;

  assert(solver && *solver == NULL);

  switch (psi->solver.psolver) {

  case (PSI_POISSON_SOLVER_PETSC):
    {
      psi_solver_petsc_t * petsc = NULL;
      ifail = psi_solver_petsc_var_epsilon_create(psi, user, &petsc);
      if (ifail == 0) *solver = (psi_solver_t *) petsc;
    }
    break;

  case (PSI_POISSON_SOLVER_SOR):
    {
      psi_solver_sor_t * sor = NULL;
      ifail = psi_solver_sor_var_epsilon_create(psi, user, &sor);
      *solver = (psi_solver_t *) sor;
    }
    break;

  default:
    ifail = -1;
  }

  return ifail;
}
