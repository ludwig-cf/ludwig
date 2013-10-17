#ifdef PETSC
/*****************************************************************************
 *
 *  psi_petsc.c
 *
 *  A solution of the Poisson equation for the potential and
 *  charge densities stored in the psi_t object.
 *
 *  The Poisson equation looks like
 *
 *    nabla^2 \psi = - rho_elec / epsilon
 *
 *  where psi is the potential, rho_elec is the free charge density, and
 *  epsilon is a permeability.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2013 The University of Edinburgh
 *
 *  Contributing Authors:
 *    Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

#include "pe.h"
#include "coords.h"
#include "physics.h"
#include "psi_s.h"
#include "psi.h"
#include "psi_sor.h"
#include "map.h"
#include "psi_petsc.h"
#include "petscksp.h"

Vec	x, b, u;      /* approx solution, RHS, exact solution */
Mat	A;            /* linear system matrix */
KSP	ksp;          /* linear solver context */
PC      pc;           /* preconditioner context */

PetscReal  norm,tol=1.e-14;  /* norm of solution error */
PetscErrorCode ierr;
PetscBool  nonzeroguess = PETSC_FALSE;


/*****************************************************************************
 *
 *  psi_petsc_init
 *
 *  Initialises PETSc KSP solver
 *
 *****************************************************************************/

int psi_petsc_init(psi_t * obj){

  int N;
  MPI_Comm comm;

  assert(obj);

  N = N_total(X)*N_total(Y)*N_total(Z);
  comm = cart_comm();

  /* Initialising PETSc vectors and matrix */
  info("\nUsing PETSc Kyrlov Subspace Solver\n");

  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
//  ierr = PetscObjectSetName((PetscObject) x, "Solution");CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_solve
 *
 *  If the f_vare_t argument is NULL, the uniform epsilon solver is used.
 *  If the argument is present, the non-uniform solver is used.
 *
 *****************************************************************************/

int psi_petsc_solve(psi_t * obj, f_vare_t fepsilon) {

  assert(obj);

  if (fepsilon == NULL) psi_petsc_poisson(obj);

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_poisson
 *
 *
 *****************************************************************************/

int psi_petsc_poisson(psi_t * obj) {

  assert(obj);

  return 0;
}

#endif
