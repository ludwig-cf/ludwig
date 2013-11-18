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
#include "control.h"
#include "physics.h"
#include "psi_s.h"
#include "psi.h"
#include "psi_sor.h"
#include "map.h"
#include "util.h"
#include "psi_petsc.h"
#include "petscksp.h"
#include "petscdmda.h"

DM             da;            /* distributed array */
Vec            x,b,u;         /* approx solution, RHS, exact solution */
Mat            A;             /* linear system matrix */
KSP            ksp;           /* linear solver context */
PC             pc;            /* preconditioner context */
PetscReal      norm;          /* norm of solution error */
PetscInt       i,j,its;

int view_matrix = 0;
int view_vector = 0;

/*****************************************************************************
 *
 *  psi_petsc_init
 *
 *  Initialises PETSc vectors, matrices and KSP solver context
 *
 *****************************************************************************/

int psi_petsc_init(psi_t * obj, f_vare_t fepsilon){

  MPI_Comm new_comm;
  int new_rank, nhalo;
  double tol_rel;              /* Relative tolerance */
  double tol_abs;              /* Absolute tolerance */
  KSPType solver_type;
  PCType pc_type;
  PetscReal rtol, abstol, dtol;
  PetscInt maxits;

  assert(obj);

  /* In order for the DMDA and the Cartesian MPI communicator 
     to share the same part of the domain decomposition it is 
     necessary to renumber the process ranks of the default 
     PETSc communicator. Default PETSc is column major decomposition. 
  */

  /* Set new rank according to PETSc ordering */
  new_rank = cart_coords(Z)*cart_size(Y)*cart_size(X) \
	+ cart_coords(Y)*cart_size(X) + cart_coords(X);

  /* Create communicator with new ranks according to PETSc ordering */
  MPI_Comm_split(PETSC_COMM_WORLD, 1, new_rank, &new_comm);

  /* Override default PETSc communicator */
  PETSC_COMM_WORLD = new_comm;

 /* Create 3D distributed array */ 
  nhalo = coords_nhalo();

  DMDACreate3d(PETSC_COMM_WORLD, \
	DMDA_BOUNDARY_PERIODIC,	DMDA_BOUNDARY_PERIODIC, DMDA_BOUNDARY_PERIODIC,	\
	DMDA_STENCIL_STAR, N_total(X), N_total(Y), N_total(Z), \
	cart_size(X), cart_size(Y), cart_size(Z), 1, nhalo, \
	NULL, NULL, NULL, &da);

  /* Create global vectors on DM */
  DMCreateGlobalVector(da,&x);
  VecDuplicate(x,&b);

  /* Create matrix on DM pre-allocated according to distributed array structure */
  DMCreateMatrix(da,MATAIJ,&A);

  /* Initialise solver context and preconditioner */

  psi_reltol(obj, &tol_rel);
  psi_abstol(obj, &tol_abs);

  KSPCreate(PETSC_COMM_WORLD,&ksp);	
  KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);
  KSPSetTolerances(ksp,tol_rel,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  KSPSetFromOptions(ksp);
  KSPSetUp(ksp);
  
  KSPGetType(ksp, &solver_type);
  KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &maxits);
  KSPGetPC(ksp, &pc);
  PCGetType(pc, &pc_type);

  info("\nUsing Krylov subspace solver\n");
  info("----------------------------\n");
  info("Solver type %s\n", solver_type);
  info("Tolerances rtol %g  abstol %g  maxits %d\n", rtol, abstol, maxits);
  info("Preconditioner type %s\n", pc_type);

  if (fepsilon == NULL) psi_petsc_compute_laplacian(obj);
  if (fepsilon != NULL) psi_petsc_compute_matrix(obj,fepsilon);
  MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);
 
  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_compute_laplacian
 *
 *  Computes the Laplacian for KSP solver. 
 *  Note that this routine uses the PETSc stencil structure, which permits
 *  local assembly of the matrix.
 *
 *****************************************************************************/

int psi_petsc_compute_laplacian(psi_t * obj) {

  PetscInt    i,j,k;
  PetscInt    xs,ys,zs,xw,yw,zw,xe,ye,ze;
  PetscScalar epsilon;
  PetscScalar v[7];
  MatStencil  row, col[7];

  assert(obj);

  MatZeroEntries(A);

  /* Get details of the distributed array data structure.
     The PETSc directives return global indices, but 
     every process works only on its local copy. */

  DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  psi_epsilon(obj, &epsilon); 

  /* 3D-Laplacian with periodic BCs */
  for(k=zs; k<ze; k++){
    for(j=ys; j<ye; j++){
      for(i=xs; i<xe; i++){

	row.i = i;
	row.j = j;
	row.k = k;

	col[0].i = i;     col[0].j = j;     col[0].k = k-1;   v[0] = - epsilon;
	col[1].i = i;     col[1].j = j-1;   col[1].k = k;     v[1] = - epsilon;
	col[2].i = i-1;   col[2].j = j;     col[2].k = k;     v[2] = - epsilon;
	col[3].i = row.i; col[3].j = row.j; col[3].k = row.k; v[3] = 6.0 * epsilon;
	col[4].i = i+1;   col[4].j = j;     col[4].k = k;     v[4] = - epsilon;
	col[5].i = i;     col[5].j = j+1;   col[5].k = k;     v[5] = - epsilon;
	col[6].i = i;     col[6].j = j;     col[6].k = k+1;   v[6] = - epsilon;
	MatSetValuesStencil(A,1,&row,7,col,v,INSERT_VALUES);

      }
    }
  }

  /* Matrix assembly & halo swap */
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  if (view_matrix) {
    info("\nPETSc output matrix\n");
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matrix.log", &viewer);
    PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_INDEX);
    MatView(A,viewer);;
    PetscViewerDestroy(&viewer);
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_compute_matrix
 *
 *  Computes the matrix for KSP solver. 
 *  Note that this routine uses the PETSc stencil structure, which permits
 *  local assembly of the matrix.
 *
 *****************************************************************************/

int psi_petsc_compute_matrix(psi_t * obj, f_vare_t fepsilon) {

  PetscInt    i,j,k;
  PetscInt    xs,ys,zs,xw,yw,zw,xe,ye,ze;
  PetscScalar epsilon;
  PetscScalar v[7];
  MatStencil  row, col[7];

  assert(obj);

  MatZeroEntries(A);

  /* Get details of the distributed array data structure.
     The PETSc directives return global indices, but 
     every process works only on its local copy. */

  DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  psi_epsilon(obj, &epsilon); 

  /* 3D-Laplacian with periodic BCs */
  for(k=zs; k<ze; k++){
    for(j=ys; j<ye; j++){
      for(i=xs; i<xe; i++){

	row.i = i;
	row.j = j;
	row.k = k;

	col[0].i = i;     col[0].j = j;     col[0].k = k-1;   v[0] = - epsilon;
	col[1].i = i;     col[1].j = j-1;   col[1].k = k;     v[1] = - epsilon;
	col[2].i = i-1;   col[2].j = j;     col[2].k = k;     v[2] = - epsilon;
	col[3].i = row.i; col[3].j = row.j; col[3].k = row.k; v[3] = 6.0 * epsilon;
	col[4].i = i+1;   col[4].j = j;     col[4].k = k;     v[4] = - epsilon;
	col[5].i = i;     col[5].j = j+1;   col[5].k = k;     v[5] = - epsilon;
	col[6].i = i;     col[6].j = j;     col[6].k = k+1;   v[6] = - epsilon;
	MatSetValuesStencil(A,1,&row,7,col,v,INSERT_VALUES);

      }
    }
  }

  /* Matrix assembly & halo swap */
  MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);

  if (view_matrix) {
    info("\nPETSc output matrix\n");
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "matrix.log", &viewer);
    PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_INDEX);
    MatView(A,viewer);;
    PetscViewerDestroy(&viewer);
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_copy_psi_to_da
 *
 *****************************************************************************/

int psi_petsc_copy_psi_to_da(psi_t * obj) {

  PetscInt    ic,jc,kc,index;
  PetscInt    noffset[3];
  PetscInt    i,j,k;
  PetscInt    xs,ys,zs,xw,yw,zw,xe,ye,ze;
  PetscScalar *** psi_3d;

  assert(obj);
  coords_nlocal_offset(noffset);

  DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);
  DMDAVecGetArray(da, x, &psi_3d);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  for (k=zs; k<ze; k++) {
    kc = k - noffset[Z] + 1;
    for (j=ys; j<ye; j++) {
      jc = j - noffset[Y] + 1;
      for (i=xs; i<xe; i++) {
	ic = i - noffset[X] + 1;

	index = coords_index(ic,jc,kc);
	psi_3d[k][j][i] = obj->psi[index];

      }
    }
  }

  DMDAVecRestoreArray(da, x, &psi_3d);

  if (view_vector) {
    info("\nPETSc output DA vector\n");
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "da.log", &viewer);
    PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_INDEX);
    VecView(x,viewer);
    PetscViewerDestroy(&viewer);
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_copy_da_to_psi
 *
 *****************************************************************************/

int psi_petsc_copy_da_to_psi(psi_t * obj) {

  PetscInt    ic,jc,kc,index;
  PetscInt    noffset[3];
  PetscInt    i,j,k;
  PetscInt    xs,ys,zs,xw,yw,zw,xe,ye,ze;
  PetscScalar *** psi_3d;

  assert(obj);
  coords_nlocal_offset(noffset);

  DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);
  DMDAVecGetArray(da, x, &psi_3d);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  for (k=zs; k<ze; k++) {
    kc = k - noffset[Z] + 1;
    for (j=ys; j<ye; j++) {
      jc = j - noffset[Y] + 1;
      for (i=xs; i<xe; i++)  {
	ic = i - noffset[X] + 1;

	index = coords_index(ic,jc,kc);
	obj->psi[index] = psi_3d[k][j][i];

      }
    }
  }

  DMDAVecRestoreArray(da, x, &psi_3d);

  psi_halo_psi(obj);

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_set_rhs
 *
 *****************************************************************************/

int psi_petsc_set_rhs(psi_t * obj) {

   PetscInt    ic,jc,kc,index;
   PetscInt    noffset[3];
   PetscInt    i,j,k;
   PetscInt    xs,ys,zs,xw,yw,zw,xe,ye,ze;
   PetscScalar *** rho_3d;
   PetscScalar rho_elec;
 
   assert(obj);
   coords_nlocal_offset(noffset);
 
   DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);
   DMDAVecGetArray(da, b, &rho_3d);
 
   xe = xs + xw;
   ye = ys + yw;
   ze = zs + zw;

   for (k=zs; k<ze; k++) {
     kc = k - noffset[Z] + 1;
     for (j=ys; j<ye; j++) {
       jc = j - noffset[Y] + 1;
       for (i=xs; i<xe; i++) {
         ic = i - noffset[X] + 1;
 
         index = coords_index(ic,jc,kc);
	 psi_rho_elec(obj, index, &rho_elec);
         rho_3d[k][j][i] = rho_elec;
 
       }
     }
   }
 
   DMDAVecRestoreArray(da, b, &rho_3d);

  if (view_vector) {
    info("\nPETSc output RHS\n");
    PetscViewer viewer;
    PetscViewerASCIIOpen(PETSC_COMM_WORLD, "rhs.log", &viewer);
    PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_INDEX);
    VecView(b,viewer);;
    PetscViewerDestroy(&viewer);
  }
 
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

  if(fepsilon != NULL) psi_petsc_compute_matrix(obj,fepsilon);
  psi_petsc_set_rhs(obj);
  psi_petsc_copy_psi_to_da(obj);
  psi_petsc_poisson(obj);
  psi_petsc_copy_da_to_psi(obj);

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_poisson
 *
 *  Solves the Poisson equation for constant permittivity.
 *  The vectors b, x are distributed arrays (DA).
 *
 *****************************************************************************/

int psi_petsc_poisson(psi_t * obj) {

  assert(obj);

  KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
  KSPSolve(ksp,b,x);

  if (is_statistics_step()) {
    KSPGetResidualNorm(ksp,&norm);
    KSPGetIterationNumber(ksp,&its);
    info("\nKrylov solver\nNorm of residual %g at %d iterations\n",norm,its);
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_petsc_finish
 *
 *  Destroys the solver context, distributed array, matrix and vectors.
 *
 *****************************************************************************/

int psi_petsc_finish() {

  KSPDestroy(&ksp);
  VecDestroy(&x);
  VecDestroy(&b);
  MatDestroy(&A);
  DMDestroy(&da);

  return 0;
}

#endif
