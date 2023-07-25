/*****************************************************************************
 *
 *  psi_petsc.c
 *
 *  A solution of the Poisson equation for the potential and
 *  charge densities stored in the psi_t object.
 *
 *  This uses the PETSc library.
 *
 *  The Poisson equation with homogeneous permittivity looks like
 *
 *    nabla^2 \psi = - rho_elec / epsilon
 *
 *  where psi is the potential, rho_elec is the free charge density, and
 *  epsilon is a permittivity.
 *
 *  There is also a version for non-uniform dielectric.
 *
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2023 The University of Edinburgh
 *
 *  Contributing Authors:
 *  Oliver Henrich  (now U. Strathclyde)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "psi_petsc.h"

static psi_solver_vt_t vt_ = {
  (psi_solver_free_ft)  psi_solver_petsc_free,
  (psi_solver_solve_ft) psi_solver_petsc_solve
};

static psi_solver_vt_t vart_ = {
  (psi_solver_free_ft) psi_solver_petsc_free,
  (psi_solver_solve_ft) psi_solver_petsc_var_epsilon_solve
};

int psi_solver_petsc_initialise(psi_t * psi, psi_solver_petsc_t * solver);
int psi_solver_petsc_matrix_set(psi_solver_petsc_t * solver);
int psi_solver_petsc_rhs_set(psi_solver_petsc_t * solver);
int psi_solver_petsc_psi_to_da(psi_solver_petsc_t * solver);
int psi_solver_petsc_da_to_psi(psi_solver_petsc_t * solver);

int psi_solver_petsc_var_epsilon_initialise(psi_t * psi, var_epsilon_t epsilon,
					    psi_solver_petsc_t * solver);
int psi_solver_petsc_var_epsilon_matrix_set(psi_solver_petsc_t * solver);
int psi_solver_petsc_var_epsilon_rhs_set(psi_solver_petsc_t * solver);


/*****************************************************************************
 *
 *  psi_solver_petsc_create
 *
 *****************************************************************************/

int psi_solver_petsc_create(psi_t * psi, psi_solver_petsc_t ** solver) {

  int ifail = -1;                     /* Check PETSC is available */
  int isInitialised = 0;

  PetscInitialised(&isInitialised);

  if (isInitialised) {
    psi_solver_petsc_t * petsc = NULL;

    petsc = (psi_solver_petsc_t *) calloc(1, sizeof(psi_solver_petsc_t));
    assert(petsc);

    if (petsc != NULL) {
      /* initialise ... */
      petsc->super.impl = &vt_;
      petsc->psi = psi;
      ifail = psi_solver_petsc_initialise(psi, petsc);
      if (ifail != 0) free(petsc);
      if (ifail == 0) *solver = petsc;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_var_epsilon_create
 *
 *****************************************************************************/

int psi_solver_petsc_var_epsilon_create(psi_t * psi, var_epsilon_t user,
					psi_solver_petsc_t ** solver) {

  int ifail = -1;                     /* Check PETSC is available */
  int isInitialised = 0;

  PetscInitialised(&isInitialised);

  if (isInitialised) {
    psi_solver_petsc_t * petsc = NULL;

    petsc = (psi_solver_petsc_t *) calloc(1, sizeof(psi_solver_petsc_t));
    assert(petsc);

    if (petsc != NULL) {
      /* initialise ... */
      petsc->super.impl = &vart_;
      petsc->psi = psi;
      petsc->fe = user.fe;
      petsc->epsilon = user.epsilon;
      ifail = psi_solver_petsc_initialise(psi, petsc);
      if (ifail != 0) free(petsc);
      if (ifail == 0) *solver = petsc;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_free
 *
 *****************************************************************************/

int psi_solver_petsc_free(psi_solver_petsc_t ** solver) {

  assert(solver && *solver);

  free(*solver);
  *solver = NULL;

  return 0;
}


#ifndef PETSC

/*****************************************************************************
 *
 *  psi_solver_petsc_initialise
 *
 *  There are two stub routines here for the case that PETSc is not
 *  avialable.
 *
 *****************************************************************************/

int psi_solver_petsc_initialise(psi_t * psi, psi_solver_petsc_t * solver) {

  /* No implementation */
  return -1;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_solve
 *
 *****************************************************************************/

int psi_solver_petsc_solve(psi_solver_petsc_t * solver, int ntimestep) {

  /* No implementation */
  return -1;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_var_epsilon_solve
 *
 *****************************************************************************/

int psi_solver_petsc_var_epsilon_solve(psi_solver_petsc_t * solver, int nt) {

  /* No implementation */
  return -1;
}

#else

#include "petscdmda.h"
#include "petscksp.h"

/* Here's the internal state. */

struct psi_solver_petsc_block_s {
  DM  da;         /* Domain management */
  Mat a;          /* System matrix */
  Vec x;          /* Unknown (potential) */
  Vec b;          /* Right-hand side */
  KSP ksp;        /* Krylov solver context */
};

/*****************************************************************************
 *
 *  psi_solver_petsc_initialise
 *
 *****************************************************************************/

int psi_solver_petsc_initialise(psi_t * psi, psi_solver_petsc_t * solver) {

  assert(psi);
  assert(solver);

  {
    size_t sz = sizeof(psi_solver_petsc_block_t);
    solver->block = (psi_solver_petsc_block_t *) calloc(1, sz);
    assert(solver->block);
    if (solver->block == NULL) return -1;
  }

  /* In order for the DMDA and the Cartesian MPI communicator
   * to share the same part of the domain decomposition it is
   *  necessary to renumber the process ranks of the default
   *  PETSc communicator. Default PETSc is column major decomposition. */

  {
    cs_t * cs = psi->cs;
    int coords[3] = {0};
    int cartsz[3] = {0};
    int ntotal[3] = {0};
    int nhalo = -1;
    int rank = -1;

    MPI_Comm comm = MPI_COMM_NULL;
    DMBoundaryType periodic = DM_BOUNDARY_PERIODIC;

    cs_cartsz(cs, cartsz);
    cs_cart_coords(cs, coords);
    cs_nhalo(cs, &nhalo);
    cs_ntotal(cs, ntotal);

    /* Set new rank according to PETSc ordering */
    /* Create communicator with new ranks according to PETSc ordering */
    /* Override default PETSc communicator */

    rank = coords[Z]*cartsz[Y]*cartsz[X] + coords[Y]*cartsz[X] + coords[X];
    MPI_Comm_split(PETSC_COMM_WORLD, 1, rank, &comm);
    PETSC_COMM_WORLD = comm;

    /* Create 3D distributed array (always periodic) */

    DMDACreate3d(PETSC_COMM_WORLD, periodic, periodic, periodic,
		 DMDA_STENCIL_BOX, ntotal[X], ntotal[Y], ntotal[Z],
		 cartsz[X], cartsz[Y], cartsz[Z], 1, nhalo,
		 NULL, NULL, NULL, &solver->block->da);

    PetscCall(DMSetVecType(solver->block->da, VECSTANDARD));
    PetscCall(DMSetMatType(solver->block->da, MATMPIAIJ));
    PetscCall(DMSetUp(solver->block->da));
  }

  /* Create global vectors and matrix */

  DMCreateMatrix(solver->block->da, &solver->block->a);
  DMCreateGlobalVector(solver->block->da, &solver->block->x);
  VecDuplicate(solver->block->x, &solver->block->b);

  /* Initialise solver context */

  {
    PetscReal abstol = psi->solver.abstol;
    PetscReal rtol   = psi->solver.reltol;
    PetscInt  maxits = psi->solver.maxits;

    KSPCreate(PETSC_COMM_WORLD, &solver->block->ksp);
    KSPSetOperators(solver->block->ksp, solver->block->a, solver->block->a);
    KSPSetTolerances(solver->block->ksp, rtol, abstol, PETSC_DEFAULT, maxits);
  }

  /* Not required in var-epsilon case, but no harm. */
  psi_solver_petsc_matrix_set(solver);

  KSPSetUp(solver->block->ksp);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_matrix_set
 *
 *****************************************************************************/

int psi_solver_petsc_matrix_set(psi_solver_petsc_t * solver) {

  int xs, ys, zs;
  int xw, yw, zw;
  int xe, ye, ze;
  double epsilon;

  double v[27] = {0};        /* Accomodate largest current stencil */
  MatStencil col[27] = {0};  /* Ditto */

  stencil_t * s = solver->psi->stencil;

  assert(solver);
  assert(solver->psi->solver.nstencil <= 27);

  /* Obtain start and width ... */
  DMDAGetCorners(solver->block->da, &xs, &ys, &zs, &xw, &yw, &zw);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  /* 3D-Laplacian with periodic BCs */
  /* Uniform dielectric constant */

  psi_epsilon(solver->psi, &epsilon);

  for (int k = zs; k < ze; k++) {
    for (int j = ys; j < ye; j++) {
      for (int i = xs; i < xe; i++) {

	MatStencil row = {.i = i, .j = j, .k = k};

	for (int p = 0; p < s->npoints; p++) {
	  col[p].i = i + s->cv[p][X];
	  col[p].j = j + s->cv[p][Y];
	  col[p].k = k + s->cv[p][Z];
	  v[p] = s->wlaplacian[p]*epsilon;
	}
	MatSetValuesStencil(solver->block->a, 1, &row, s->npoints, col, v,
			    INSERT_VALUES);
      }
    }
  }

  /* Matrix assembly & halo swap */
  /* Retain the non-zero structure of the matrix */

  MatAssemblyBegin(solver->block->a, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(solver->block->a, MAT_FINAL_ASSEMBLY);
  MatSetOption(solver->block->a, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);

  /* Set the matrix, and the nullspace */
  KSPSetOperators(solver->block->ksp, solver->block->a, solver->block->a);

  {
    MatNullSpace nullsp;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullsp);
    MatSetNullSpace(solver->block->a, nullsp);
    MatNullSpaceDestroy(&nullsp);
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_solve
 *
 *****************************************************************************/

int psi_solver_petsc_solve(psi_solver_petsc_t * solver, int ntimestep) {

  assert(solver);

  psi_solver_petsc_rhs_set(solver);
  psi_solver_petsc_psi_to_da(solver);

  KSPSetInitialGuessNonzero(solver->block->ksp, PETSC_TRUE);
  KSPSolve(solver->block->ksp, solver->block->b, solver->block->x);

  if (ntimestep % solver->psi->solver.nfreq == 0) {
    /* Report on progress of the solver.
     * Note the default Petsc residual is the preconditioned L2 norm. */
    pe_t * pe = solver->psi->pe;
    PetscInt  its  = 0;
    PetscReal norm = 0.0;
    PetscCall(KSPGetIterationNumber(solver->block->ksp, &its));
    PetscCall(KSPGetResidualNorm(solver->block->ksp, &norm));
    pe_info(pe, "\n");
    pe_info(pe, "Krylov solver\n");
    pe_info(pe, "Norm of residual %g at %d iterations\n", norm, its);
  }

  psi_solver_petsc_da_to_psi(solver);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_rhs_set
 *
 *****************************************************************************/

int psi_solver_petsc_rhs_set(psi_solver_petsc_t * solver) {

  cs_t * cs = NULL;
  int xs, ys, zs;
  int xw, yw, zw;
  int xe, ye, ze;
  int offset[3] = {0};
  double e0[3] = {0};
  double *** rho_3d = {0};

  assert(solver);

  cs = solver->psi->cs;
  cs_nlocal_offset(cs, offset);

  DMDAGetCorners(solver->block->da, &xs, &ys, &zs, &xw, &yw, &zw);
  DMDAVecGetArray(solver->block->da, solver->block->b, &rho_3d);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  for (int k = zs; k < ze; k++) {
    int kc = k - offset[Z] + 1;
    for (int j = ys; j < ye; j++) {
      int jc = j - offset[Y] + 1;
      for (int i = xs; i < xe; i++) {

	int ic = i - offset[X] + 1;
	int index = cs_index(cs, ic, jc, kc);
	double rho_elec = 0.0;
	/* Non-dimensional potential in Poisson eqn requires e/kT */
	double eunit = solver->psi->e;
	double beta  = solver->psi->beta;

	psi_rho_elec(solver->psi, index, &rho_elec);
	rho_3d[k][j][i] = rho_elec*eunit*beta;
      }
    }
  }

  /* Modify right hand side for external electric field */
  /* The system must be periodic, so no need to check. */

  e0[X] = solver->psi->e0[X];
  e0[Y] = solver->psi->e0[Y];
  e0[Z] = solver->psi->e0[Z];

  if (e0[X] || e0[Y] || e0[Z]) {

    int ntotal[3] = {0};
    int mpi_coords[3] = {0};
    int mpi_cartsz[3] = {0};
    double epsilon = 0.0;

    cs_ntotal(cs, ntotal);
    cs_cart_coords(cs, mpi_coords);
    cs_cartsz(cs, mpi_cartsz);

    psi_epsilon(solver->psi, &epsilon);

    if (e0[X] && mpi_coords[X] == 0) {
      for (int k = zs; k < ze; k++) {
	for (int j = ys; j < ye; j++) {
	  rho_3d[k][j][0] += epsilon*e0[X]*ntotal[X];
	}
      }
    }

    if (e0[X] && mpi_coords[X] == mpi_cartsz[X] - 1) {
      for (int k = zs; k < ze; k++) {
	for (int j = ys; j < ye; j++) {
	  rho_3d[k][j][xe-1] -= epsilon*e0[X]*ntotal[X];
	}
      }
    }

    if (e0[Y] && mpi_coords[Y] == 0) {
      for (int k = zs; k < ze; k++) {
	for (int i = xs; i < xe; i++) {
	  rho_3d[k][0][i] += epsilon*e0[Y]*ntotal[Y];
	}
      }
    }

    if (e0[Y] && mpi_coords[Y] == mpi_cartsz[Y] - 1) {
      for (int k = zs; k < ze; k++) {
	for (int i = xs; i < xe; i++) {
	  rho_3d[k][ye-1][i] -= epsilon*e0[Y]*ntotal[Y];
	}
      }
    }

    if (e0[Z] && mpi_coords[Z] == 0) {
      for (int j = ys; j < ye; j++) {
	for (int i = xs; i < xe; i++) {
	  rho_3d[0][j][i] += epsilon*e0[Z]*ntotal[Z];
	}
      }
    }

    if (e0[Z] && mpi_coords[Z] == mpi_cartsz[Z] - 1) {
      for (int j = ys; j < ye; j++) {
	for (int i = xs; i < xe; i++) {
	  rho_3d[ze-1][j][i] -= epsilon*e0[Z]*ntotal[Z];
	}
      }
    }
  }

  DMDAVecRestoreArray(solver->block->da, solver->block->b, &rho_3d);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_psi_to_da
 *
 *  Copy the potential from the psi_t represetation to the solution
 *  vector as an initial guess.
 *
 *****************************************************************************/

int psi_solver_petsc_psi_to_da(psi_solver_petsc_t * solver) {

  cs_t * cs = NULL;
  int xs, ys, zs;
  int xw, yw, zw;
  int xe, ye, ze;
  int offset[3] = {0};
  double *** psi_3d = NULL;

  assert(solver);

  cs = solver->psi->cs;
  cs_nlocal_offset(cs, offset);

  DMDAGetCorners(solver->block->da, &xs, &ys, &zs, &xw, &yw, &zw);
  DMDAVecGetArray(solver->block->da, solver->block->x, &psi_3d);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  for (int k = zs; k < ze; k++) {
    int kc = k - offset[Z] + 1;
    for (int j = ys; j < ye; j++) {
      int jc = j - offset[Y] + 1;
      for (int i = xs; i < xe; i++) {
	int ic = i - offset[X] + 1;
	int index = cs_index(cs, ic, jc, kc);
	psi_3d[k][j][i] = solver->psi->psi->data[index];
      }
    }
  }

  DMDAVecRestoreArray(solver->block->da, solver->block->x, &psi_3d);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_da_to_psi
 *
 *  Copy te Petsc solution back to the psi_t represetation.
 *
 *****************************************************************************/

int psi_solver_petsc_da_to_psi(psi_solver_petsc_t * solver) {

  cs_t * cs = NULL;
  int xs, ys, zs;
  int xw, yw, zw;
  int xe, ye, ze;
  int offset[3] = {0};
  double *** psi_3d = NULL;

  assert(solver);

  cs = solver->psi->cs;
  cs_nlocal_offset(cs, offset);

  DMDAGetCorners(solver->block->da, &xs, &ys, &zs, &xw, &yw, &zw);
  DMDAVecGetArray(solver->block->da, solver->block->x, &psi_3d);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  for (int k = zs; k < ze; k++) {
    int kc = k - offset[Z] + 1;
    for (int j = ys; j < ye; j++) {
      int jc = j - offset[Y] + 1;
      for (int i = xs; i < xe; i++)  {
	int ic = i - offset[X] + 1;
	int index = cs_index(cs, ic, jc, kc);
	solver->psi->psi->data[index] = psi_3d[k][j][i];
      }
    }
  }

  DMDAVecRestoreArray(solver->block->da, solver->block->x, &psi_3d);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_var_epsilon_solve
 *
 *****************************************************************************/

int psi_solver_petsc_var_epsilon_solve(psi_solver_petsc_t * solver, int nt) {

  assert(solver);

  psi_solver_petsc_var_epsilon_matrix_set(solver);
  psi_solver_petsc_var_epsilon_rhs_set(solver);

  psi_solver_petsc_psi_to_da(solver);

  KSPSetInitialGuessNonzero(solver->block->ksp, PETSC_TRUE);
  KSPSolve(solver->block->ksp, solver->block->b, solver->block->x);

  if (nt % solver->psi->solver.nfreq == 0) {
    /* Report on progress of the solver.
     * Note the default Petsc residual is the preconditioned L2 norm. */
    pe_t * pe = solver->psi->pe;
    PetscInt  its  = 0;
    PetscReal norm = 0.0;
    PetscCall(KSPGetIterationNumber(solver->block->ksp, &its));
    PetscCall(KSPGetResidualNorm(solver->block->ksp, &norm));
    pe_info(pe, "\n");
    pe_info(pe, "Krylov solver (with dielectric contrast)\n");
    pe_info(pe, "Norm of residual %g at %d iterations\n", norm, its);
  }

  psi_solver_petsc_da_to_psi(solver);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_var_epsilon_matrix_set
 *
 *****************************************************************************/

int psi_solver_petsc_var_epsilon_matrix_set(psi_solver_petsc_t * solver) {

  cs_t * cs = NULL;
  int xs, ys, zs;
  int xw, yw, zw;
  int xe, ye, ze;
  int offset[3] = {0};

  double v[27] = {0};
  MatStencil col[27] = {0};
  stencil_t * s = solver->psi->stencil;

  assert(solver);

  cs = solver->psi->cs;
  cs_nlocal_offset(cs, offset);

  /* Get details of the distributed array data structure.
     The PETSc directives return global indices, but
     every process works only on its local copy. */

  DMDAGetCorners(solver->block->da, &xs, &ys, &zs, &xw, &yw, &zw);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  /* 3D-operator with periodic BCs */

  for (int k = zs; k < ze; k++) {
    int kc = 1 + k - offset[Z];
    for (int j = ys; j < ye; j++) {
      int jc = 1 + j - offset[Y];
      for (int i = xs; i < xe; i++) {

	int ic = 1 + i - offset[X];
	int index = cs_index(cs, ic, jc, kc);
	double epsilon0 = 0.0;
	double gradeps[3] = {0};

	MatStencil row = {.i = i, .j = j, .k = k};

	solver->epsilon(solver->fe, index, &epsilon0);

	/* Local approx. to grad epsilon ... */
	for (int p = 1; p < s->npoints; p++) {
	  int ic1 = ic + s->cv[p][X];
	  int jc1 = jc + s->cv[p][Y];
	  int kc1 = kc + s->cv[p][Z];
	  int index1 = cs_index(cs, ic1, jc1, kc1);
	  double epsilon1 = 0.0;
	  solver->epsilon(solver->fe, index1, &epsilon1);
	  gradeps[X] += s->wgradients[p]*s->cv[p][X]*epsilon1;
	  gradeps[Y] += s->wgradients[p]*s->cv[p][Y]*epsilon1;
	  gradeps[Z] += s->wgradients[p]*s->cv[p][Z]*epsilon1;
	}

	for (int p = 0; p < s->npoints; p++) {
	  col[p].i = i + s->cv[p][X];
	  col[p].j = j + s->cv[p][Y];
	  col[p].k = k + s->cv[p][Z];

	  /* Laplacian part of operator */
	  v[p] = s->wlaplacian[p]*epsilon0;

	  /* Addtional terms in generalised Poisson equation */
	  v[p] += s->wgradients[p]*s->cv[p][X]*gradeps[X];
	  v[p] += s->wgradients[p]*s->cv[p][Y]*gradeps[Y];
	  v[p] += s->wgradients[p]*s->cv[p][Z]*gradeps[Z];
	}

	MatSetValuesStencil(solver->block->a, 1, &row, s->npoints, col, v,
			    INSERT_VALUES);
      }
    }
  }

  /* Matrix assembly & halo swap */
  /* Retain the non-zero structure of the matrix */

  MatAssemblyBegin(solver->block->a, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(solver->block->a, MAT_FINAL_ASSEMBLY);
  MatSetOption(solver->block->a, MAT_NEW_NONZERO_LOCATIONS, PETSC_FALSE);

  /* Set the matrix, preconditioner and nullspace */
  KSPSetOperators(solver->block->ksp, solver->block->a, solver->block->a);

  {
    MatNullSpace nullsp;
    MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_TRUE, 0, NULL, &nullsp);
    MatSetNullSpace(solver->block->a, nullsp);
    MatNullSpaceDestroy(&nullsp);
  }

  KSPSetFromOptions(solver->block->ksp);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_var_epsilon_rhs_set
 *
 *  The only difference here (cf. uniform epsilon) is in the external
 *  field terms.
 *
 *****************************************************************************/

int psi_solver_petsc_var_epsilon_rhs_set(psi_solver_petsc_t * solver) {

  cs_t * cs = NULL;
  int xs, ys, zs;
  int xw, yw, zw;
  int xe, ye, ze;
  int offset[3] = {0};
  double e0[3] = {0};
  double *** rho_3d = {0};

  assert(solver);

  cs = solver->psi->cs;
  cs_nlocal_offset(cs, offset);

  DMDAGetCorners(solver->block->da, &xs, &ys, &zs, &xw, &yw, &zw);
  DMDAVecGetArray(solver->block->da, solver->block->b, &rho_3d);

  xe = xs + xw;
  ye = ys + yw;
  ze = zs + zw;

  for (int k = zs; k < ze; k++) {
    int kc = k - offset[Z] + 1;
    for (int j = ys; j < ye; j++) {
      int jc = j - offset[Y] + 1;
      for (int i = xs; i < xe; i++) {

	int ic = i - offset[X] + 1;
	int index = cs_index(cs, ic, jc, kc);
	double rho_elec = 0.0;
	/* Non-dimensional potential in Poisson eqn requires e/kT */
	double eunit = solver->psi->e;
	double beta  = solver->psi->beta;

	psi_rho_elec(solver->psi, index, &rho_elec);
	rho_3d[k][j][i] = rho_elec*eunit*beta;
      }
    }
  }

  /* Modify right hand side for external electric field */
  /* The system must be periodic, so no need to check. */

  e0[X] = solver->psi->e0[X];
  e0[Y] = solver->psi->e0[Y];
  e0[Z] = solver->psi->e0[Z];

  if (e0[X] || e0[Y] || e0[Z]) {

    int ntotal[3] = {0};
    int mpi_coords[3] = {0};
    int mpi_cartsz[3] = {0};

    cs_ntotal(cs, ntotal);
    cs_cart_coords(cs, mpi_coords);
    cs_cartsz(cs, mpi_cartsz);

    if (e0[X] && mpi_coords[X] == 0) {
      for (int k = zs; k < ze; k++) {
	int kc = 1 + k - offset[Z];
	for (int j = ys; j < ye; j++) {
	  int jc = 1 + j - offset[Y];
	  int index = cs_index(cs, 1, jc, kc);
	  double epsilon = 0.0;
	  solver->epsilon(solver->fe, index, &epsilon);
	  rho_3d[k][j][0] += epsilon*e0[X]*ntotal[X];
	}
      }
    }

    if (e0[X] && mpi_coords[X] == mpi_cartsz[X] - 1) {
      for (int k = zs; k < ze; k++) {
	int kc = 1 + k - offset[Z];
	for (int j = ys; j < ye; j++) {
	  int jc = 1 + j - offset[Y];
	  int ic = xe    - offset[X];
	  int index = cs_index(cs, ic, jc, kc);
	  double epsilon = 0.0;
	  solver->epsilon(solver->fe, index, &epsilon);
	  rho_3d[k][j][xe-1] -= epsilon*e0[X]*ntotal[X];
	}
      }
    }

    if (e0[Y] && mpi_coords[Y] == 0) {
      for (int k = zs; k < ze; k++) {
	int kc = 1 + k - offset[Z];
	for (int i = xs; i < xe; i++) {
	  int ic = 1 + i - offset[X];
	  int index = cs_index(cs, ic, 1, kc);
	  double epsilon = 0.0;
	  solver->epsilon(solver->fe, index, &epsilon);
	  rho_3d[k][0][i] += epsilon*e0[Y]*ntotal[Y];
	}
      }
    }

    if (e0[Y] && mpi_coords[Y] == mpi_cartsz[Y] - 1) {
      for (int k = zs; k < ze; k++) {
	int kc = 1 + k - offset[Z];
	for (int i = xs; i < xe; i++) {
	  int jc = ye    - offset[Y];
	  int ic = 1 + i - offset[X];
	  int index = cs_index(cs, ic, jc, kc);
	  double epsilon = 0.0;
	  solver->epsilon(solver->fe, index, &epsilon);
	  rho_3d[k][ye-1][i] -= epsilon*e0[Y]*ntotal[Y];
	}
      }
    }

    if (e0[Z] && mpi_coords[Z] == 0) {
      for (int j = ys; j < ye; j++) {
	int jc = 1 + j - offset[Y];
	for (int i = xs; i < xe; i++) {
	  int ic = 1 + i - offset[X];
	  int index = cs_index(cs, ic, jc, 1);
	  double epsilon = 0.0;
	  solver->epsilon(solver->fe, index, &epsilon);
	  rho_3d[0][j][i] += epsilon*e0[Z]*ntotal[Z];
	}
      }
    }

    if (e0[Z] && mpi_coords[Z] == mpi_cartsz[Z] - 1) {
      int kc = ze - offset[Z];
      for (int j = ys; j < ye; j++) {
	int jc = 1 + j - offset[Y];
	for (int i = xs; i < xe; i++) {
	  int ic = 1 + i - offset[X];
	  int index = cs_index(cs, ic, jc, kc);
	  double epsilon = 0.0;
	  solver->epsilon(solver->fe, index, &epsilon);
	  rho_3d[ze-1][j][i] -= epsilon*e0[Z]*ntotal[Z];
	}
      }
    }
  }

  DMDAVecRestoreArray(solver->block->da, solver->block->b, &rho_3d);

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_petsc_finalise
 *
 *****************************************************************************/

int psi_solver_petsc_finalise(psi_solver_petsc_t * solver) {

  assert(solver);

  PetscCall(KSPDestroy(&solver->block->ksp));
  PetscCall(VecDestroy(&solver->block->x));
  PetscCall(VecDestroy(&solver->block->b));
  PetscCall(MatDestroy(&solver->block->a));
  PetscCall(DMDestroy(&solver->block->da));

  free(solver->block);

  return 0;
}

#endif
