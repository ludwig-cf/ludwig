/*****************************************************************************
 *
 *  psi_sor.c
 *
 *  A solution of the Poisson equation for the potential and
 *  charge densities stored in the psi_t object.
 *
 *  The simple Poisson equation looks like
 *
 *    nabla^2 \psi = - rho_elec / epsilon
 *
 *  where psi is the potential, rho_elec is the free charge density, and
 *  epsilon is a permeability.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2023 The University of Edinburgh
 *
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *    Ignacio Pagonabarraga (ipagonabarraga@ub.edu)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

#include "pe.h"
#include "coords.h"
#include "psi_sor.h"
#include "util.h"

/* Function table */

static psi_solver_vt_t vt_ = {
  (psi_solver_free_ft)  psi_solver_sor_free,
  (psi_solver_solve_ft) psi_solver_sor_solve
};

static psi_solver_vt_t vart_ = {
  (psi_solver_free_ft)  psi_solver_sor_free,
  (psi_solver_solve_ft) psi_solver_sor_var_epsilon_solve
};

/*****************************************************************************
 *
 *  psi_solver_sor_create
 *
 *****************************************************************************/

int psi_solver_sor_create(psi_t * psi, psi_solver_sor_t ** sor) {

  int ifail = 0;
  psi_solver_sor_t * solver = NULL;

  assert(psi);
  assert(sor);

  solver = (psi_solver_sor_t *) calloc(1, sizeof(psi_solver_sor_t));
  if (solver == NULL) {
    ifail = -1;
  }
  else {
    /* Set the function table ... */
    solver->super.impl = &vt_;
    solver->psi = psi;
  }

  *sor = solver;

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_sor_free
 *
 *****************************************************************************/

int psi_solver_sor_free(psi_solver_sor_t ** sor) {

  assert(sor);
  assert(*sor);

  free(*sor);
  *sor  = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_sor_solve
 *
 *  Solve Poisson equation with uniform permittivity.
 *
 *  The differencing is a seven point stencil for \nabla^2 \psi. So
 *
 *  epsilon [ psi(i+1,j,k) - 2 psi(i,j,k) + psi(i-1,j,k)
 *          + psi(i,j+1,k) - 2 psi(i,j,k) + psi(i,j-1,k)
 *          + psi(i,j,k+1) - 2 psi(i,j,k) + psi(i,j,k-1) ] = -rho_elec(i,j,k)
 *
 *  We use the asymptotic estimate of the spectral radius for
 *  the Jacobi iteration
 *      radius ~= 1 - (pi^2 / 2N^2)
 *  where N is the linear dimension of the problem. It's important
 *  to get this right to keep the number of iterations as small as
 *  possible.
 *
 *  If this is an initial solve, the initial norm of the residual
 *  may be quite large (e.g., psi(t = 0)  = 0; rhs \neq 0); in this
 *  case a relative tolerance would be appropriate to decide
 *  termination. On subsequent calls, we might expect the initial
 *  residual to be relatively small (psi not charged much since
 *  previous solve), and an absolute tolerance might be appropriate.
 *
 *  The actual residual is checked against both at every 'ncheck'
 *  iterations, and either condition met will result in termination
 *  of the iteration. If neither criterion is met, the iteration will
 *  finish after 'niteration' iterations.
 *
 *  "its" is the time step for statistics purposes.
 *
 *  See, e.g., Press et al. Chapter 19.
 *
 *****************************************************************************/

int psi_solver_sor_solve(psi_solver_sor_t * sor, int its) {

  int niteration = 1000;       /* Maximum number of iterations */
  const int ncheck = 5;        /* Check global residual every n iterations */
  
  int nhalo;
  int nlocal[3];
  int nsites;
  int xs, ys, zs;              /* Memory strides */
  double rho_elec;             /* Right-hand side */
  double residual;             /* Residual at given point */
  double rnorm[2];             /* Initial and current norm of residual */
  double rnorm_local[2];       /* Local values */
  double epsilon;              /* Uniform permittivity */
  double dpsi;
  double omega;                /* Over-relaxation parameter 1 < omega < 2 */
  double radius;               /* Spectral radius of Jacobi iteration */
  double eunit, beta;
  double ltot[3];

  MPI_Comm comm;               /* Cartesian communicator */

  psi_t * psi = sor->psi;
  double * __restrict__ psidata = psi->psi->data;

  cs_ltot(psi->cs, ltot);
  cs_nhalo(psi->cs, &nhalo);
  cs_nsites(psi->cs, &nsites);
  cs_nlocal(psi->cs, nlocal);
  cs_cart_comm(psi->cs, &comm);
  cs_strides(psi->cs, &xs, &ys, &zs);

  assert(nhalo >= 1);

  /* The red/black operation needs to be tested for odd numbers
   * of points in parallel. */

  assert(nlocal[X] % 2 == 0);
  assert(nlocal[Y] % 2 == 0);
  assert(nlocal[Z] % 2 == 0);

  /* Compute initial norm of the residual */

  radius = 1.0 - 0.5*pow(4.0*atan(1.0)/dmax(ltot[X],ltot[Z]), 2);

  psi_epsilon(psi, &epsilon);
  psi_maxits(psi, &niteration);

  psi_beta(psi, &beta);
  psi_unit_charge(psi, &eunit);

  rnorm_local[0] = 0.0;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(psi->cs, ic, jc, kc);

	psi_rho_elec(psi, index, &rho_elec);

	/* Non-dimensional potential in Poisson eqn requires e/kT */
	/* This is just the L2 norm of the right hand side. */

	residual = eunit*beta*rho_elec;
	rnorm_local[0] += residual*residual;
      }
    }
  }

  rnorm_local[0] = sqrt(rnorm_local[0]);

  /* Iterate to solution */

  omega = 1.0;

  for (int n = 0; n < niteration; n++) {

    /* Compute current normal of the residual */

    rnorm_local[1] = 0.0;

    for (int pass = 0; pass < 2; pass++) {

      for (int ic = 1; ic <= nlocal[X]; ic++) {
	for (int jc = 1; jc <= nlocal[Y]; jc++) {
	  int kst = 1 + (ic + jc + pass) % 2;
	  for (int kc = kst; kc <= nlocal[Z]; kc += 2) {

	    int index = cs_index(psi->cs, ic, jc, kc);

	    psi_rho_elec(psi, index, &rho_elec);

	    /* 6-point stencil of Laplacian */

	    dpsi
	      = psidata[addr_rank0(nsites, index + xs)]
	      + psidata[addr_rank0(nsites, index - xs)]
	      + psidata[addr_rank0(nsites, index + ys)]
	      + psidata[addr_rank0(nsites, index - ys)]
	      + psidata[addr_rank0(nsites, index + zs)]
	      + psidata[addr_rank0(nsites, index - zs)]
	      - 6.0*psidata[addr_rank0(nsites, index)];

	    /* Non-dimensional potential in Poisson eqn requires e/kT */

	    residual = epsilon*dpsi + eunit*beta*rho_elec;
	    psidata[addr_rank0(nsites, index)]
	      -= omega*residual / (-6.0*epsilon);
	    rnorm_local[1] += residual*residual;
	  }
	}
      }

      /* Recompute relaxation parameter and next pass */

      if (n == 0 && pass == 0) {
	omega = 1.0 / (1.0 - 0.5*radius*radius);
      }
      else {
	omega = 1.0 / (1.0 - 0.25*radius*radius*omega);
      }
      assert(1.0 < omega && omega < 2.0);

      psi_halo_psi(psi);
      psi_halo_psijump(psi);

    }

    if ((n % ncheck) == 0) {

      /* Compare residual and exit if small enough */
      pe_t * pe = psi->pe;

      rnorm_local[1] = sqrt(rnorm_local[1]);

      MPI_Allreduce(rnorm_local, rnorm, 2, MPI_DOUBLE, MPI_SUM, comm);

      if (rnorm[1] < psi->solver.abstol) {

	if (its % psi->solver.nfreq == 0) {
	  pe_info(pe, "\n");
	  pe_info(pe, "SOR solver converged to absolute tolerance\n");
	  pe_info(pe, "SOR residual %14.7e at %d iterations\n", rnorm[1], n);
	}
	break;
      }

      if (rnorm[1] < psi->solver.reltol*rnorm[0]) {

	if (its % psi->solver.nfreq == 0) {
	  pe_info(pe, "\n");
	  pe_info(pe, "SOR solver converged to relative tolerance\n");
	  pe_info(pe, "SOR residual %14.7e at %d iterations\n", rnorm[1], n);
	}
	break;
      }
    }
 
    if (n == niteration-1) {
      pe_info(psi->pe, "\n");
      pe_info(psi->pe, "SOR solver exceeded %d iterations\n", n+1);
      pe_info(psi->pe, "SOR residual %le (initial) %le (final)\n\n",
	      rnorm[0], rnorm[1]);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_solver_sor_var_epsilon_create
 *
 *****************************************************************************/

int psi_solver_sor_var_epsilon_create(psi_t * psi, var_epsilon_t user,
				      psi_solver_sor_t ** sor) {
  int ifail = 0;
  psi_solver_sor_t * solver = NULL;

  assert(psi);
  assert(sor);

  solver = (psi_solver_sor_t *) calloc(1, sizeof(psi_solver_sor_t));
  if (solver == NULL) {
    ifail = -1;
  }
  else {
    /* Set the function table etc... */
    solver->super.impl = &vart_;
    solver->psi = psi;
    solver->fe = user.fe;
    solver->epsilon = user.epsilon;
  }

  *sor = solver;

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_sor_var_epsilon_solve
 *
 *  This is essentially a copy of the above, but it allows a spatially
 *  varying permittivity epsilon:
 *
 *    div [epsilon(r) grad phi(r) ] = -rho(r)
 *
 *  Only the electro-symmetric free energy is relevant at the moment.
 *
 ****************************************************************************/

int psi_solver_sor_var_epsilon_solve(psi_solver_sor_t * sor, int its) {

  int niteration = 2000;       /* Maximum number of iterations */
  const int ncheck = 1;        /* Check global residual every n iterations */

  int nlocal[3];
  int nsites;
  int xs, ys, zs;              /* Memory strides */

  double rho_elec;             /* Right-hand side */
  double residual;             /* Residual at given point */
  double rnorm[2];             /* Initial and current norm of residual */
  double rnorm_local[2];       /* Local values */

  double depsi;                /* Differenced left-hand side */
  double eps0, eps1;           /* Permittivity values */

  double omega;                /* Over-relaxation parameter 1 < omega < 2 */
  double radius;               /* Spectral radius of Jacobi iteration */

  double ltot[3];
  double eunit, beta;

  MPI_Comm comm;               /* Cartesian communicator */

  psi_t * psi = sor->psi;
  double * __restrict__ psidata = psi->psi->data;

  cs_ltot(psi->cs, ltot);
  cs_nlocal(psi->cs, nlocal);
  cs_nsites(psi->cs, &nsites);
  cs_cart_comm(psi->cs, &comm);
  cs_strides(psi->cs, &xs, &ys, &zs);

  /* The red/black operation needs to be tested for odd numbers
   * of points in parallel. */

  assert(nlocal[X] % 2 == 0);
  assert(nlocal[Y] % 2 == 0);
  assert(nlocal[Z] % 2 == 0);

  /* Compute initial norm of the residual */

  radius = 1.0 - 0.5*pow(4.0*atan(1.0)/dmax(ltot[X],ltot[Z]), 2);

  psi_maxits(psi, &niteration);
  psi_beta(psi, &beta);
  psi_unit_charge(psi, &eunit);


  /* Compute the initial norm of the right hand side. */

  rnorm_local[0] = 0.0;

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(psi->cs, ic, jc, kc);
	psi_rho_elec(psi, index, &rho_elec);

	residual = eunit*beta*rho_elec;
	rnorm_local[0] += residual*residual;
      }
    }
  }

  rnorm_local[0] = sqrt(rnorm_local[0]);

  /* Iterate to solution */

  omega = 1.0;

  for (int n = 0; n < niteration; n++) {

    /* Compute current normal of the residual */

    rnorm_local[1] = 0.0;

    for (int pass = 0; pass < 2; pass++) {

      for (int ic = 1; ic <= nlocal[X]; ic++) {
	for (int jc = 1; jc <= nlocal[Y]; jc++) {
	  int kst = 1 + (ic + jc + pass) % 2;
	  for (int kc = kst; kc <= nlocal[Z]; kc += 2) {

	    int index = cs_index(psi->cs, ic, jc, kc);
	    depsi  = 0.0;

	    psi_rho_elec(psi, index, &rho_elec);
	    sor->epsilon(sor->fe, index, &eps0);

	    /* Laplacian part of operator */

	    depsi += eps0*(-6.0*psidata[addr_rank0(nsites, index)]
			   + psidata[addr_rank0(nsites, index + xs)]
			   + psidata[addr_rank0(nsites, index - xs)]
			   + psidata[addr_rank0(nsites, index + ys)]
			   + psidata[addr_rank0(nsites, index - ys)]
			   + psidata[addr_rank0(nsites, index + zs)]
			   + psidata[addr_rank0(nsites, index - zs)]);

	    /* Additional terms in generalised Poisson equation */

	    sor->epsilon(sor->fe, index + xs, &eps1);
	    depsi += 0.25*eps1*(psidata[addr_rank0(nsites, index + xs)]
			      - psidata[addr_rank0(nsites, index - xs)]);

	    sor->epsilon(sor->fe, index - xs, &eps1);
	    depsi -= 0.25*eps1*(psidata[addr_rank0(nsites, index + xs)]
			      - psidata[addr_rank0(nsites, index - xs)]);

	    sor->epsilon(sor->fe, index + ys, &eps1);
	    depsi += 0.25*eps1*(psidata[addr_rank0(nsites, index + ys)]
			      - psidata[addr_rank0(nsites, index - ys)]);

	    sor->epsilon(sor->fe, index - ys, &eps1);
	    depsi -= 0.25*eps1*(psidata[addr_rank0(nsites, index + ys)]
			      - psidata[addr_rank0(nsites, index - ys)]);

	    sor->epsilon(sor->fe, index + zs, &eps1);
	    depsi += 0.25*eps1*(psidata[addr_rank0(nsites, index + zs)]
			      - psidata[addr_rank0(nsites, index - zs)]);

	    sor->epsilon(sor->fe, index - zs, &eps1);
	    depsi -= 0.25*eps1*(psidata[addr_rank0(nsites, index + zs)]
			      - psidata[addr_rank0(nsites, index - zs)]);

	    /* Non-dimensional potential in Poisson eqn requires e/kT */
	    residual = depsi + eunit*beta*rho_elec;
	    psidata[addr_rank0(nsites,index)] -= omega*residual / (-6.0*eps0);
	    rnorm_local[1] += residual*residual;
	  }
	}
      }

      psi_halo_psi(psi);
      psi_halo_psijump(psi);

    }

    /* Recompute relation parameter */
    /* Note: The default Chebychev acceleration causes a convergence problem */ 
    omega = 1.0 / (1.0 - 0.25*radius*radius*omega);

    if ((n % ncheck) == 0) {

      /* Compare residual and exit if small enough */

      rnorm_local[1] = sqrt(rnorm_local[1]);
      MPI_Allreduce(rnorm_local, rnorm, 2, MPI_DOUBLE, MPI_SUM, comm);

      if (rnorm[1] < psi->solver.abstol) {

	if (its % psi->solver.nfreq == 0) {
	  pe_info(psi->pe, "\n");
	  pe_info(psi->pe, "SOR (heterogeneous) solver converged to "
		  "absolute tolerance\n");
	  pe_info(psi->pe, "SOR residual %14.7e at %d iterations\n",
		  rnorm[1], n);
	}
	break;
      }

      if (rnorm[1] < psi->solver.reltol*rnorm[0]) {

	if (its % psi->solver.nfreq == 0) {
	  pe_info(psi->pe, "\n");
	  pe_info(psi->pe, "SOR (heterogeneous) solver converged to "
		  "relative tolerance\n");
	  pe_info(psi->pe, "SOR residual %14.7e at %d iterations\n",
		  rnorm[1], n);
	}
	break;
      }

      if (n == niteration-1) {
	pe_info(psi->pe, "\n");
	pe_info(psi->pe, "SOR solver (heterogeneous) exceeded %d iterations\n",
		n+1);
	pe_info(psi->pe, "SOR residual %le (initial) %le (final)\n\n",
		rnorm[0], rnorm[1]);
      }
    }
  }

  return 0;
}
