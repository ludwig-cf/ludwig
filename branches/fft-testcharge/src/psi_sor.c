/*****************************************************************************
 *
 *  psi_sor.c
 *
 *  A solution of the Poisson equation for the potenial and
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
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

#include "pe.h"
#include "coords.h"
#include "psi_s.h"
#include "psi_sor.h"

/*****************************************************************************
 *
 *  psi_sor_poisson
 *
 *
 *  First attempt. Uniform permeativity. The differencing is a seven
 *  point stencil for \nabla^2 \psi. So
 *
 *  epsilon [ psi(i+1,j,k) - 2 psi(i,j,k) + psi(i-1,j,k)
 *          + psi(i,j+1,k) - 2 psi(i,j,k) + psi(i,j-1,k)
 *          + psi(i,j,k+1) - 2 psi(i,j,k) + psi(i,j,k-1) ] = -rho_elec(i,j,k)
 *
 *  We use the asymphtotic estimate of the spectral radius for
 *  the Jabcobi iteration
 *      radius ~= 1 - (pi^2 / 2N^2)
 *  where N is the linear dimension of the problem. It's important
 *  to get this right to keep the number of iterations as small as
 *  possible.
 *
 *  If this is an initial solve, the initial norm of the resisdual
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
 *****************************************************************************/

int psi_sor_poisson(psi_t * obj) {

  const int niteration = 1000; /* Maximum number of iterations */
  const int ncheck = 5;        /* Check global residual every n iterations */
  
  int ic, jc, kc, index;
  int nhalo;
  int n;                       /* Relaxation iterations */
  int pass;                    /* Red/black iteration */
  int kst;                     /* Start kc index for red/black iteration */
  int nlocal[3];
  int xs, ys, zs;              /* Memory strides */

  double rho_elec;             /* Right-hand side */
  double residual;             /* Residual at given point */
  double rnorm[2];             /* Initial and current norm of residual */
  double rnorm_local[2];       /* Local values */

  double epsilon;              /* Uniform permeativity */
  double dpsi;

  double omega;                /* Over-relaxation parameter 1 < omega < 2 */
  double radius;               /* Spectral radius of Jacobi iteration */

  double tol_rel;              /* Relative tolerance */
  double tol_abs;              /* Absolute tolerance */

  MPI_Comm comm;               /* Cartesian communicator */

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  comm = cart_comm();

  assert(nhalo >= 1);

  /* The red/black operation needs to be tested for odd numbers
   * of points in parallel. */
  assert(nlocal[X] % 2 == 0);
  assert(nlocal[Y] % 2 == 0);
  assert(nlocal[Z] % 2 == 0);

  zs = 1;
  ys = zs*(nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  /* Compute initial norm of the residual */

  radius = 1.0 - 0.5*pow(4.0*atan(1.0)/L(X), 2);

  psi_epsilon(obj, &epsilon);
  psi_reltol(obj, &tol_rel);
  psi_abstol(obj, &tol_abs);
  rnorm_local[0] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_rho_elec(obj, index, &rho_elec);
	dpsi = obj->psi[index + xs] + obj->psi[index - xs]
	     + obj->psi[index + ys] + obj->psi[index - ys]
	     + obj->psi[index + zs] + obj->psi[index - zs]
	     - 6.0*obj->psi[index];
	rnorm_local[0] += fabs(epsilon*dpsi + rho_elec);
      }
    }
  }

  /* Iterate to solution */

  omega = 1.0;

  for (n = 0; n < niteration; n++) {

    /* Compute current normal of the residual */

    rnorm_local[1] = 0.0;

    for (pass = 0; pass < 2; pass++) {

      for (ic = 1; ic <= nlocal[X]; ic++) {
	for (jc = 1; jc <= nlocal[Y]; jc++) {
	  kst = 1 + (ic + jc + pass) % 2;
	  for (kc = kst; kc <= nlocal[Z]; kc += 2) {

	    index = coords_index(ic, jc, kc);

	    psi_rho_elec(obj, index, &rho_elec);
	    dpsi = obj->psi[index + xs] + obj->psi[index - xs]
	         + obj->psi[index + ys] + obj->psi[index - ys]
	         + obj->psi[index + zs] + obj->psi[index - zs]
	      - 6.0*obj->psi[index];
	    residual = epsilon*dpsi + rho_elec;
	    obj->psi[index] -= omega*residual / (-6.0*epsilon);
	    rnorm_local[1] += fabs(residual);
	  }
	}
      }

      /* Recompute relation parameter and next pass */

      if (n == 0 && pass == 0) {
	omega = 1.0 / (1.0 - 0.5*radius*radius);
      }
      else {
	omega = 1.0 / (1.0 - 0.25*radius*radius*omega);
      }
      assert(1.0 < omega);
      assert(omega < 2.0);

      psi_halo_psi(obj);
    }

    if ((n % ncheck) == 0) {
      /* Compare residual and exit if small enough */
      MPI_Allreduce(rnorm_local, rnorm, 2, MPI_DOUBLE, MPI_SUM, comm);
      if (rnorm[1] < tol_abs || rnorm[1] < tol_rel*rnorm[0]) break;
    }
  }

//  if( pe_rank() == 0) printf("n iterations: %d\n", n);

  return 0;
}

/*****************************************************************************
 *
 *  psi_sor_vare_poisson
 *
 *  This is essentislly a copy of the above, but it allows a spatially
 *  varying permeativity epsilon:
 *
 *    div [epsilon(r) grad phi(r) ] = -rho(r)
 *
 *  The differencing in one dimension is:
 *
 *    d_x (epsilon d_x phi) =~ epsilon(i+1/2,j,k) [ phi(i+1,j,k) - phi(i,j,k) ]
 *                           - epsilon(i-1/2,j,k) [ phi(i,j,k) - phi(i-1,j,k) ]
 *
 *  which collapses to the uniform case above for constant epsilon. This
 *  is straightforwardly extended to three dimensions to give the five
 *  point stencil.
 *
 *  Permeativity is provided via a function of f_vare_t which returns
 *  values at index = (i,j,k); a simple average
 *
 *    epsilon(i+1/2,j,k) = (1/2) [ epsilon(i,j,k) + epsilon(i+1,j,k) ]
 *
 *  is used to get the mid-point values in each coordinate direction.
 *
 ****************************************************************************/

int psi_sor_vare_poisson(psi_t * obj, f_vare_t fepsilon) {

  const int niteration = 1000; /* Maximum number of iterations */
  const int ncheck = 5;        /* Check global residual every n iterations */
  
  int ic, jc, kc, index;
  int nhalo;
  int n;                       /* Relaxation iterations */
  int pass;                    /* Red/black iteration */
  int kst;                     /* Start kc index for red/black iteration */
  int nlocal[3];
  int xs, ys, zs;              /* Memory strides */

  double rho_elec;             /* Right-hand side */
  double residual;             /* Residual at given point */
  double rnorm[2];             /* Initial and current norm of residual */
  double rnorm_local[2];       /* Local values */

  double depsi;                /* Differenced left-hand side */
  double ep0, ep1;             /* Permeativity values */
  double epsh;                 /* Permeativity value half-way */
  double epstot;               /* Net coefficient of the psi(i,j,k) term */

  double omega;                /* Over-relaxation parameter 1 < omega < 2 */
  double radius;               /* Spectral radius of Jacobi iteration */

  double tol_rel;              /* Relative tolerance */
  double tol_abs;              /* Absolute tolerance */

  MPI_Comm comm;               /* Cartesian communicator */

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  comm = cart_comm();

  assert(nhalo >= 1);

  /* The red/black operation needs to be tested for odd numbers
   * of points in parallel. */
  assert(nlocal[X] % 2 == 0);
  assert(nlocal[Y] % 2 == 0);
  assert(nlocal[Z] % 2 == 0);

  zs = 1;
  ys = zs*(nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  /* Compute initial norm of the residual */

  radius = 1.0 - 0.5*pow(4.0*atan(1.0)/L(X), 2);

  psi_reltol(obj, &tol_rel);
  psi_abstol(obj, &tol_abs);
  rnorm_local[0] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_rho_elec(obj, index, &rho_elec);
	fepsilon(index, &ep0);
	depsi = 0.0;

	fepsilon(index + xs, &ep1);
	depsi += 0.5*(ep0 + ep1)*(obj->psi[index + xs] - obj->psi[index]);
	fepsilon(index - xs, &ep1);
	depsi += 0.5*(ep0 + ep1)*(obj->psi[index - xs] - obj->psi[index]);
	fepsilon(index + ys, &ep1);
	depsi += 0.5*(ep0 + ep1)*(obj->psi[index + ys] - obj->psi[index]);
	fepsilon(index - ys, &ep1);
	depsi += 0.5*(ep0 + ep1)*(obj->psi[index - ys] - obj->psi[index]);
	fepsilon(index + zs, &ep1);
	depsi += 0.5*(ep0 + ep1)*(obj->psi[index + zs] - obj->psi[index]);
	fepsilon(index - zs, &ep1);
	depsi += 0.5*(ep0 + ep1)*(obj->psi[index - zs] - obj->psi[index]);

	rnorm_local[0] += fabs(depsi + rho_elec);
      }
    }
  }

  /* Iterate to solution */

  omega = 1.0;

  for (n = 0; n < niteration; n++) {

    /* Compute current normal of the residual */

    rnorm_local[1] = 0.0;

    for (pass = 0; pass < 2; pass++) {

      for (ic = 1; ic <= nlocal[X]; ic++) {
	for (jc = 1; jc <= nlocal[Y]; jc++) {
	  kst = 1 + (ic + jc + pass) % 2;
	  for (kc = kst; kc <= nlocal[Z]; kc += 2) {

	    index = coords_index(ic, jc, kc);

	    psi_rho_elec(obj, index, &rho_elec);
	    fepsilon(index, &ep0);
	    depsi = 0.0;
	    epstot = 0.0;

	    fepsilon(index + xs, &ep1);
	    epsh = 0.5*(ep0 + ep1);
	    epstot += epsh;
	    depsi += epsh*(obj->psi[index + xs] - obj->psi[index]);

	    fepsilon(index - xs, &ep1);
	    epsh = 0.5*(ep0 + ep1);
	    epstot += epsh;
	    depsi += epsh*(obj->psi[index - xs] - obj->psi[index]);

	    fepsilon(index + ys, &ep1);
	    epsh = 0.5*(ep0 + ep1);
	    epstot += epsh;
	    depsi += epsh*(obj->psi[index + ys] - obj->psi[index]);

	    fepsilon(index - ys, &ep1);
	    epsh = 0.5*(ep0 + ep1);
	    epstot += epsh;
	    depsi += epsh*(obj->psi[index - ys] - obj->psi[index]);

	    fepsilon(index + zs, &ep1);
	    epsh = 0.5*(ep0 + ep1);
	    epstot += epsh;
	    depsi += epsh*(obj->psi[index + zs] - obj->psi[index]);

	    fepsilon(index - zs, &ep1);
	    epsh = 0.5*(ep0 + ep1);
	    epstot += epsh;
	    depsi += epsh*(obj->psi[index - zs] - obj->psi[index]);

	    residual = depsi + rho_elec;
	    obj->psi[index] -= omega*residual / (-1.0*epstot);
	    rnorm_local[1] += fabs(residual);
	  }
	}
      }

      /* Recompute relation parameter and next pass */

      if (n == 0 && pass == 0) {
	omega = 1.0 / (1.0 - 0.5*radius*radius);
      }
      else {
	omega = 1.0 / (1.0 - 0.25*radius*radius*omega);
      }
      assert(1.0 < omega);
      assert(omega < 2.0);

      psi_halo_psi(obj);
    }

    if ((n % ncheck) == 0) {
      /* Compare residual and exit if small enough */
      MPI_Allreduce(rnorm_local, rnorm, 2, MPI_DOUBLE, MPI_SUM, comm);
      if (rnorm[1] < tol_abs || rnorm[1] < tol_rel*rnorm[0]) break;
    }
  }


  return 0;
}
