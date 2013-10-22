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

PetscReal   norm, tol=1.e-14;  /* norm of solution error */
PetscInt    i, n = 10, col[3], its; 
PetscMPIInt size; 
PetscScalar neg_one=-1.0, one=1.0, value[3]; 
PetscBool   nonzeroguess = PETSC_FALSE; 

/*****************************************************************************
 *
 *  psi_petsc_init
 *
 *  Initialises PETSc vectors, matrices and KSP solver context
 *
 *****************************************************************************/

int psi_petsc_init(psi_t * obj){

  int N;
  MPI_Comm comm;

  assert(obj);

  N = coords_nsites();
  comm = cart_comm();

  info("\nUsing PETSc Kyrlov Subspace Solver\n");

  /* Allocate PETSc vectors and matrix */
  VecCreate(comm,&x);
  VecSetSizes(x,PETSC_DECIDE,N);
  VecSetFromOptions(x);
  VecDuplicate(x,&b);
  VecDuplicate(x,&u);

  MatCreate(comm,&A);
  MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,N,N);
  MatSetType(A,MATAIJ);  
  MatSetFromOptions(A);
  MatSetUp(A);

  /* Create matrix */
  psi_petsc_assemble_matrix(obj);

  /* Initialise solver context and preconditioner */
  KSPCreate(comm,&ksp);	
  KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);

  KSPGetPC(ksp,&pc);
  PCSetType(pc,PCJACOBI);

  KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_assemble_matrix
 *
 *  Creates the matrix.
 *
 *****************************************************************************/

int psi_petsc_assemble_matrix(psi_t * obj) {

  assert(obj);

  MatZeroEntries(A);

  value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;

  for (i=1; i<n-1; i++) {
    col[0] = i-1;
    col[1] = i;
    col[2] = i+1;
    MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);
  }

  i = n -1;
  col[0]=n-2;
  col[1]=n-1;
  MatSetValues(A,1,&i,2,col,value,INSERT_VALUES); 

  i = 0;
  col[0] = 0;
  col[1] = 1;
  value[0] = 2.0;
  value[1] = -1.0;
  MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);

  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

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

  KSPSolve(ksp,b,x);
  info("\nPETSc info:\n");
  KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);

  /* Check the error */
  VecAXPY(x,neg_one,u);
  VecNorm(x,NORM_2,&norm);
  KSPGetIterationNumber(ksp,&its);

  if (norm > tol) {
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G, Iterations %D\n",norm,its);
  }

  return 0;
}

#endif
