/*****************************************************************************
 *
 *  psi_petsc.c
 *
 *  A solution of the Poisson equation for the potential and
 *  charge densities stored in the psi_t object.
 *
 *  This uses the PETSc library version 3.4.3
 *
 *  The Poisson equation homogeneous permittivity looks like
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
 *  (c) 2013 The University of Edinburgh
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

#ifdef PETSC

#include "pe.h"
#include "coords.h"
#include "control.h"
#include "physics.h"
#include "psi_s.h"
#include "psi.h"
#include "psi_sor.h"
#include "psi_gradients.h"
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
int       i,j,its;

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
  int niteration = 10000;      /* Number of iterations */
  KSPType solver_type;
  PCType pc_type;
  PetscReal rtol, abstol, dtol;
  int maxits;

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
	DMDA_STENCIL_BOX, N_total(X), N_total(Y), N_total(Z), \
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
  psi_maxits(obj, &niteration);

  KSPCreate(PETSC_COMM_WORLD,&ksp);	
  KSPSetOperators(ksp,A,A,SAME_NONZERO_PATTERN);
  KSPSetTolerances(ksp,tol_rel,tol_abs,PETSC_DEFAULT,niteration);
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

  int i, j, k;
  int xs, ys, zs, xw, yw, zw, xe, ye, ze;
  double epsilon;

#ifdef NP_D3Q6
  double v[7];
  MatStencil  row, col[7];
#endif

#ifdef NP_D3Q18
  double v[19];
  MatStencil  row, col[19];
  double r3 = 0.333333333333333, r6 = 0.166666666666667;
#endif

#ifdef NP_D3Q26
  double v[27];
  MatStencil  row, col[27];
  double r10 = 0.1;
  double r30 = 0.033333333333333;
  double r15_7  = 0.46666666666666;
  double r15_64 = 4.26666666666666;
#endif

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

#ifdef NP_D3Q6
	/* 7-point stencil */
	col[0].i = row.i; col[0].j = row.j; col[0].k = row.k; v[0] = 6.0 * epsilon;
	col[1].i = i-1;   col[1].j = j;     col[1].k = k;     v[1] = - epsilon;
	col[2].i = i;     col[2].j = j-1;   col[2].k = k;     v[2] = - epsilon;
	col[3].i = i;     col[3].j = j;     col[3].k = k-1;   v[3] = - epsilon;
	col[4].i = i+1;   col[4].j = j;     col[4].k = k;     v[4] = - epsilon;
	col[5].i = i;     col[5].j = j+1;   col[5].k = k;     v[5] = - epsilon;
	col[6].i = i;     col[6].j = j;     col[6].k = k+1;   v[6] = - epsilon;
	MatSetValuesStencil(A,1,&row,7,col,v,INSERT_VALUES);
#endif

#ifdef NP_D3Q18
	/* 19-point stencil */
	col[0].i  = row.i; col[0].j  = row.j; col[0].k  = row.k; v[0]  =  4.0 * epsilon;
	col[1].i  = i+1;   col[1].j  = j+1;   col[1].k  = k;     v[1]  = - r6 * epsilon;
	col[2].i  = i+1;   col[2].j  = j;     col[2].k  = k+1;   v[2]  = - r6 * epsilon;
	col[3].i  = i+1;   col[3].j  = j;     col[3].k  = k;     v[3]  = - r3 * epsilon;
	col[4].i  = i+1;   col[4].j  = j;     col[4].k  = k-1;   v[4]  = - r6 * epsilon;
	col[5].i  = i+1;   col[5].j  = j-1;   col[5].k  = k;     v[5]  = - r6 * epsilon;
	col[6].i  = i;     col[6].j  = j+1;   col[6].k  = k+1;   v[6]  = - r6 * epsilon;
	col[7].i  = i;     col[7].j  = j+1;   col[7].k  = k;     v[7]  = - r3 * epsilon;
	col[8].i  = i;     col[8].j  = j+1;   col[8].k  = k-1;   v[8]  = - r6 * epsilon;
	col[9].i  = i;     col[9].j  = j;     col[9].k  = k+1;   v[9]  = - r3 * epsilon;
	col[10].i = i;     col[10].j = j;     col[10].k = k-1;   v[10] = - r3 * epsilon;
	col[11].i = i;     col[11].j = j-1;   col[11].k = k+1;   v[11] = - r6 * epsilon;
	col[12].i = i;     col[12].j = j-1;   col[12].k = k;     v[12] = - r3 * epsilon;
	col[13].i = i;     col[13].j = j-1;   col[13].k = k-1;   v[13] = - r6 * epsilon;
	col[14].i = i-1;   col[14].j = j+1;   col[14].k = k;     v[14] = - r6 * epsilon;
	col[15].i = i-1;   col[15].j = j;     col[15].k = k+1;   v[15] = - r6 * epsilon;
	col[16].i = i-1;   col[16].j = j;     col[16].k = k;     v[16] = - r3 * epsilon;
	col[17].i = i-1;   col[17].j = j;     col[17].k = k-1;   v[17] = - r6 * epsilon;
	col[18].i = i-1;   col[18].j = j-1;   col[18].k = k;     v[18] = - r6 * epsilon;
	MatSetValuesStencil(A,1,&row,19,col,v,INSERT_VALUES);
#endif

#ifdef NP_D3Q26
	/* 27-point stencil */
	col[0].i  = row.i; col[0].j  = row.j; col[0].k  = row.k; v[0]  =  r15_64 * epsilon;
	col[1].i  = i-1;   col[1].j  = j-1;   col[1].k  = k-1;   v[1]  = - r30   * epsilon;
	col[2].i  = i-1;   col[2].j  = j-1;   col[2].k  = k;     v[2]  = - r10   * epsilon;
	col[3].i  = i-1;   col[3].j  = j-1;   col[3].k  = k+1;   v[3]  = - r30   * epsilon;
	col[4].i  = i-1;   col[4].j  = j;     col[4].k  = k-1;   v[4]  = - r10   * epsilon;
	col[5].i  = i-1;   col[5].j  = j;     col[5].k  = k;     v[5]  = - r15_7 * epsilon;
	col[6].i  = i-1;   col[6].j  = j;     col[6].k  = k+1;   v[6]  = - r10   * epsilon;
	col[7].i  = i-1;   col[7].j  = j+1;   col[7].k  = k-1;   v[7]  = - r30   * epsilon;
	col[8].i  = i-1;   col[8].j  = j+1;   col[8].k  = k;     v[8]  = - r10   * epsilon;
	col[9].i  = i-1;   col[9].j  = j+1;   col[9].k  = k+1;   v[9]  = - r30   * epsilon;
	col[10].i = i;     col[10].j = j-1;   col[10].k = k-1;   v[10] = - r10   * epsilon;
	col[11].i = i;     col[11].j = j-1;   col[11].k = k;     v[11] = - r15_7 * epsilon;
	col[12].i = i;     col[12].j = j-1;   col[12].k = k+1;   v[12] = - r10   * epsilon;
	col[13].i = i;     col[13].j = j;     col[13].k = k-1;   v[13] = - r15_7 * epsilon;
	col[14].i = i;     col[14].j = j;     col[14].k = k+1;   v[14] = - r15_7 * epsilon;
	col[15].i = i;     col[15].j = j+1;   col[15].k = k-1;   v[15] = - r10   * epsilon;
	col[16].i = i;     col[16].j = j+1;   col[16].k = k;     v[16] = - r15_7 * epsilon;
	col[17].i = i;     col[17].j = j+1;   col[17].k = k+1;   v[17] = - r10   * epsilon;
	col[18].i = i+1;   col[18].j = j-1;   col[18].k = k-1;   v[18] = - r30   * epsilon;
	col[19].i = i+1;   col[19].j = j-1;   col[19].k = k;     v[19] = - r10   * epsilon;
	col[20].i = i+1;   col[20].j = j-1;   col[20].k = k+1;   v[20] = - r30   * epsilon;
	col[21].i = i+1;   col[21].j = j;     col[21].k = k-1;   v[21] = - r10   * epsilon;
	col[22].i = i+1;   col[22].j = j;     col[22].k = k;     v[22] = - r15_7 * epsilon;
	col[23].i = i+1;   col[23].j = j;     col[23].k = k+1;   v[23] = - r10   * epsilon;
	col[24].i = i+1;   col[24].j = j+1;   col[24].k = k-1;   v[24] = - r30   * epsilon;
	col[25].i = i+1;   col[25].j = j+1;   col[25].k = k;     v[25] = - r10   * epsilon;
	col[26].i = i+1;   col[26].j = j+1;   col[26].k = k+1;   v[26] = - r30   * epsilon;
	MatSetValuesStencil(A,1,&row,27,col,v,INSERT_VALUES);
#endif

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

  int ic, jc, kc, p;
  int index, index0, index1;
  int noffset[3];

  int i, j, k, ia;
  int xs, ys, zs, xw, yw, zw, xe, ye, ze;
  double eps, eps1, grad_eps[3];
  

#ifdef NP_D3Q6
  double  v[7];
  MatStencil   row, col[7];
  const double matval[7] = {6.0, 
		       -1.0, -1.0, -1.0, 
		       -1.0, -1.0, -1.0}; 
#endif

#ifdef NP_D3Q18
  double v[19];
  MatStencil  row, col[19];

#define r3 (0.333333333333333)
#define r6 (0.166666666666666)
  const double  matval[19] = {4.0,
			  -r6, -r6, -r3,
			  -r6, -r6, -r6,
			  -r3, -r6, -r3,
			  -r3, -r6, -r3,
			  -r6, -r6, -r6,
			  -r3, -r6, -r6};
#endif

#ifdef NP_D3Q26
  double v[27];
  MatStencil  row, col[27];

#define r10 (0.1)
#define r30 (0.033333333333333)
#define r15_7 (0.46666666666666)
#define r15_64 (4.26666666666666)

  const double  matval[27] = {r15_64, 
			 -r30,  -r10  , -r30, 
			 -r10,  -r15_7, -r10, 
			 -r30,  -r10  , -r30, 
			 -r10,  -r15_7, -r10, 
			 -r15_7,      -r15_7, 
			 -r10,  -r15_7, -r10, 
			 -r30,  -r10  , -r30, 
			 -r10,  -r15_7, -r10, 
			 -r30,  -r10  , -r30};
#endif

  assert(obj);

  MatZeroEntries(A);

  /* Get details of the distributed array data structure.
     The PETSc directives return global indices, but 
     every process works only on its local copy. */

  DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  coords_nlocal_offset(noffset);

  /* 3D-operator with periodic BCs */
  for(k=zs; k<ze; k++){
    for(j=ys; j<ye; j++){
      for(i=xs; i<xe; i++){

	row.i = i;
	row.j = j;
	row.k = k;

#ifdef NP_D3Q6
	/* 7-point stencil */
	col[0].i = row.i; col[0].j = row.j; col[0].k = row.k;
	col[1].i = i-1;   col[1].j = j;     col[1].k = k;
	col[2].i = i;     col[2].j = j-1;   col[2].k = k;
	col[3].i = i;     col[3].j = j;     col[3].k = k-1;
	col[4].i = i+1;   col[4].j = j;     col[4].k = k;
	col[5].i = i;     col[5].j = j+1;   col[5].k = k;
	col[6].i = i;     col[6].j = j;     col[6].k = k+1;

	ic = (col[0].i + 1) - noffset[X];
	jc = (col[0].j + 1) - noffset[Y];
	kc = (col[0].k + 1) - noffset[Z];
	index = coords_index(ic, jc, kc);

	fepsilon(index, &eps); 
	psi_grad_eps_d3qx(fepsilon, index, grad_eps);

	for (p = 0; p < PSI_NGRAD; p++){

	  /* Laplacian part of operator */
	  v[p] = matval[p] * eps;

	  /* Addtional terms in generalised Poisson equation */
	  for(ia = 0; ia < 3; ia++){
	    v[p] -= grad_eps[ia] * psi_gr_wv[p] * psi_gr_rcs2 * psi_gr_cv[p][ia];
	  }

	}

	MatSetValuesStencil(A,1,&row,7,col,v,INSERT_VALUES);
#endif

#ifdef NP_D3Q18
	/* 19-point stencil */
	col[0].i  = row.i; col[0].j  = row.j; col[0].k  = row.k;
	col[1].i  = i+1;   col[1].j  = j+1;   col[1].k  = k;    
	col[2].i  = i+1;   col[2].j  = j;     col[2].k  = k+1;  
	col[3].i  = i+1;   col[3].j  = j;     col[3].k  = k;    
	col[4].i  = i+1;   col[4].j  = j;     col[4].k  = k-1;  
	col[5].i  = i+1;   col[5].j  = j-1;   col[5].k  = k;    
	col[6].i  = i;     col[6].j  = j+1;   col[6].k  = k+1;  
	col[7].i  = i;     col[7].j  = j+1;   col[7].k  = k;    
	col[8].i  = i;     col[8].j  = j+1;   col[8].k  = k-1;  
	col[9].i  = i;     col[9].j  = j;     col[9].k  = k+1;  
	col[10].i = i;     col[10].j = j;     col[10].k = k-1;  
	col[11].i = i;     col[11].j = j-1;   col[11].k = k+1;  
	col[12].i = i;     col[12].j = j-1;   col[12].k = k;    
	col[13].i = i;     col[13].j = j-1;   col[13].k = k-1;  
	col[14].i = i-1;   col[14].j = j+1;   col[14].k = k;    
	col[15].i = i-1;   col[15].j = j;     col[15].k = k+1;  
	col[16].i = i-1;   col[16].j = j;     col[16].k = k;    
	col[17].i = i-1;   col[17].j = j;     col[17].k = k-1;  
	col[18].i = i-1;   col[18].j = j-1;   col[18].k = k;    

        /* Laplacian part of operator */
	ic = (col[0].i + 1) - noffset[X];
	jc = (col[0].j + 1) - noffset[Y];
	kc = (col[0].k + 1) - noffset[Z];
	index = coords_index(ic, jc, kc);

	fepsilon(index, &eps); 
	psi_grad_eps_d3qx(fepsilon, index, grad_eps);

	for (p = 0; p < PSI_NGRAD; p++){

	  /* Laplacian part of operator */
	  v[p] = matval[p] * eps;

	  /* Addtional terms in generalised Poisson equation */
	  for(ia = 0; ia < 3; ia++){
	    v[p] -= grad_eps[ia] * psi_gr_wv[p] * psi_gr_rcs2 * psi_gr_cv[p][ia];
	  }

	}

	MatSetValuesStencil(A,1,&row,19,col,v,INSERT_VALUES);
#endif

#ifdef NP_D3Q26
	/* 27-point stencil */
	col[0].i  = row.i; col[0].j  = row.j; col[0].k  = row.k;
	col[1].i  = i-1;   col[1].j  = j-1;   col[1].k  = k-1;  
	col[2].i  = i-1;   col[2].j  = j-1;   col[2].k  = k;    
	col[3].i  = i-1;   col[3].j  = j-1;   col[3].k  = k+1; 
	col[4].i  = i-1;   col[4].j  = j;     col[4].k  = k-1;
	col[5].i  = i-1;   col[5].j  = j;     col[5].k  = k;     
	col[6].i  = i-1;   col[6].j  = j;     col[6].k  = k+1;  
	col[7].i  = i-1;   col[7].j  = j+1;   col[7].k  = k-1; 
	col[8].i  = i-1;   col[8].j  = j+1;   col[8].k  = k;  
	col[9].i  = i-1;   col[9].j  = j+1;   col[9].k  = k+1;  
	col[10].i = i;     col[10].j = j-1;   col[10].k = k-1; 
	col[11].i = i;     col[11].j = j-1;   col[11].k = k;  
	col[12].i = i;     col[12].j = j-1;   col[12].k = k+1; 
	col[13].i = i;     col[13].j = j;     col[13].k = k-1; 
	col[14].i = i;     col[14].j = j;     col[14].k = k+1; 
	col[15].i = i;     col[15].j = j+1;   col[15].k = k-1;
	col[16].i = i;     col[16].j = j+1;   col[16].k = k; 
	col[17].i = i;     col[17].j = j+1;   col[17].k = k+1; 
	col[18].i = i+1;   col[18].j = j-1;   col[18].k = k-1;
	col[19].i = i+1;   col[19].j = j-1;   col[19].k = k; 
	col[20].i = i+1;   col[20].j = j-1;   col[20].k = k+1; 
	col[21].i = i+1;   col[21].j = j;     col[21].k = k-1;
	col[22].i = i+1;   col[22].j = j;     col[22].k = k; 
	col[23].i = i+1;   col[23].j = j;     col[23].k = k+1;
	col[24].i = i+1;   col[24].j = j+1;   col[24].k = k-1;
	col[25].i = i+1;   col[25].j = j+1;   col[25].k = k; 
	col[26].i = i+1;   col[26].j = j+1;   col[26].k = k+1;

        /* Laplacian part of operator */
	ic = (col[0].i + 1) - noffset[X];
	jc = (col[0].j + 1) - noffset[Y];
	kc = (col[0].k + 1) - noffset[Z];
	index = coords_index(ic, jc, kc);

	fepsilon(index, &eps); 
	psi_grad_eps_d3qx(fepsilon, index, grad_eps);

	for (p = 0; p < PSI_NGRAD; p++){

	  /* Laplacian part of operator */
	  v[p] = matval[p] * eps;

	  /* Addtional terms in generalised Poisson equation */
	  for(ia = 0; ia < 3; ia++){
	    v[p] -= grad_eps[ia] * psi_gr_wv[p] * psi_gr_rcs2 * psi_gr_cv[p][ia];
	  }

	}
	
	MatSetValuesStencil(A,1,&row,27,col,v,INSERT_VALUES);
#endif

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

  int    ic,jc,kc,index;
  int    noffset[3];
  int    i,j,k;
  int    xs,ys,zs,xw,yw,zw,xe,ye,ze;
  double *** psi_3d;

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

  int    ic,jc,kc,index;
  int    noffset[3];
  int    i,j,k;
  int    xs,ys,zs,xw,yw,zw,xe,ye,ze;
  double *** psi_3d;

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

   int    ic,jc,kc,index;
   int    noffset[3];
   int    i,j,k;
   int    xs,ys,zs,xw,yw,zw,xe,ye,ze;
   double *** rho_3d;
   double rho_elec;
 
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
 *  psi_petsc_set_rhs_vare
 *
 *****************************************************************************/

int psi_petsc_set_rhs_vare(psi_t * obj, f_vare_t fepsilon) {

   int    ic,jc,kc;
   int    index, index0, index1;
   int    noffset[3];
   int    i, j, k, ia;
   int    xs,ys,zs,xw,yw,zw,xe,ye,ze;
   double *** rho_3d;
   double rho_elec;
   double eps0, eps1, grad_eps[3];
   double e0[3];   
 
   assert(obj);
   coords_nlocal_offset(noffset);
 
   DMDAGetCorners(da,&xs,&ys,&zs,&xw,&yw,&zw);
   DMDAVecGetArray(da, b, &rho_3d);
 
   physics_e0(e0);

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

	 psi_grad_eps_d3qx(fepsilon, index, grad_eps);

	 for(ia=0; ia<3; ia++){
	   rho_3d[k][j][i] -= grad_eps[ia] * e0[ia];
	 }

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

  if(fepsilon == NULL) {
    psi_petsc_set_rhs(obj);
  }
  if(fepsilon != NULL) {
    psi_petsc_compute_matrix(obj,fepsilon);
    psi_petsc_set_rhs_vare(obj,fepsilon);
  }

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
