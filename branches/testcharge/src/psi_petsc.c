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
#include "petscdmda.h"

DM             da;            /* distributed array */
Vec            x,b,u;         /* approx solution, RHS, exact solution */
Mat            A;             /* linear system matrix */
KSP            ksp;           /* linear solver context */
PetscReal      norm;          /* norm of solution error */
PetscInt       i,j,its;

/*****************************************************************************
 *
 *  psi_petsc_init
 *
 *  Initialises PETSc vectors, matrices and KSP solver context
 *
 *****************************************************************************/

int psi_petsc_init(psi_t * obj){

  int nhalo;
  assert(obj);

  nhalo = coords_nhalo();

  info("\nUsing PETSc Kyrlov Subspace Solver\n");

 /* Create 3D distributed array */ 
  DMDACreate3d(PETSC_COMM_WORLD, \
	DMDA_BOUNDARY_PERIODIC,	DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC,	\
	DMDA_STENCIL_BOX, N_total(X), N_total(Y), N_total(Z), \
	cart_size(X), cart_size(Y), cart_size(Z), 1, nhalo, \
	NULL, NULL, NULL, &da);

  /* Create global vectors */
  DMCreateGlobalVector(da,&u);
  VecDuplicate(u,&b);
  VecDuplicate(u,&x);

  /* Create matrix pre-allocated according to the DMDA */
  DMCreateMatrix(da,MATAIJ,&A);

  /* Initialise solver context and preconditioner */
  KSPCreate(PETSC_COMM_WORLD,&ksp);	
  KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);
  KSPSetTolerances(ksp,1.e-5,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);

  psi_petsc_compute_matrix(obj);

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_compute_matrix
 *
 *  Creates the matrix for KSP solver. 
 *  Note that the matrix ought to be pre-allocated to achieve good performance.
 *
 *****************************************************************************/

int psi_petsc_compute_matrix(psi_t * obj) {

  PetscInt    i,j,k;
  PetscInt    xs,ys,zs,xw,yw,zw,xe,ye,ze;
  PetscInt    nx,ny,nz;
  PetscScalar v[7];
  MatStencil  row, col[3];

  assert(obj);

  DMDAGetInfo(da,0,&nx,&ny,&nz,0,0,0,0,0,0,0,0,0);
  DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  for(k=zs; k<ze; k++){
    for(j=ys; j<ye; j++){
      for(i=xs; i<xe; i++){

	row.i = i;
	row.j = j;
	row.k = k;

	if (i==0 || j==0 || k==0 || i==nx-1 || j==ny-1 || k==nz-1) {
	  v[0] = 2.0;
	  MatSetValuesStencil(A,1,&row,1,&row,v,INSERT_VALUES);
	}
	else{
	  col[0].i = i; col[0].j = j; col[0].k = k-1; v[0] = -1.0;
	  col[1].i = i; col[1].j = j-1; col[1].k = k; v[1] = -1.0;
	  col[2].i = i-1; col[2].j = j; col[2].k = k; v[2] = -1.0;
	  col[3].i = row.i; col[3].j = row.j; col[3].k = row.k; v[3] = 2.0;
	  col[4].i = i+1; col[4].j = j; col[4].k = k; v[4] = -1.0;
	  col[5].i = i; col[5].j = j+1; col[5].k = k; v[5] = -1.0;
	  col[6].i = i; col[6].j = j; col[6].k = k+1; v[6] = -1.0;
	  MatSetValuesStencil(A,1,&row,7,col,v,INSERT_VALUES);
	}

      }
    }
  }

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
 *****************************************************************************/

int psi_petsc_poisson(psi_t * obj) {

  assert(obj);

  KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
  KSPSolve(ksp,b,x);

  /* Error check */
  VecAXPY(x,-1.,u);
  VecNorm(x,NORM_2,&norm);
  KSPGetIterationNumber(ksp,&its);

  PetscPrintf(PETSC_COMM_WORLD,"Norm of error %G iterations %D\n",norm,its);

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_finish
 *
 *  Destroys the solver context and distributed matrix and vectors.
 *
 *****************************************************************************/

int psi_petsc_finish() {

  KSPDestroy(&ksp);
  VecDestroy(&u);
  VecDestroy(&x);
  VecDestroy(&b);
  MatDestroy(&A);
  DMDestroy(&da);

  return 0;
}

#endif
