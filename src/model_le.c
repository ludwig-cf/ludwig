/*****************************************************************************
 *
 *  model_le.c
 *
 *  Lees-Edwards transformations for distributions.
 *
 *  Note that the distributions have displacement u*t
 *  not u*(t-1) returned by le_get_displacement().
 *  This is for reasons of backwards compatability.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "control.h"
#include "model.h"
#include "physics.h"
#include "leesedwards.h"

static void le_reproject(void);
static void le_displace_and_interpolate(void);
static void le_displace_and_interpolate_parallel(void);
static void le_reproject_all(void);

/*****************************************************************************
 *
 *  model_le_apply_boundary_conditions
 *
 *  This is the driver to apply the LE conditions to the distributions
 *  (applied to the post-collision distributions). There are two
 *  stages:
 *
 *  1. a reprojection of distributions that will cross a plane in the
 *     upcoming propagation step.
 *  2. a displacement and interpolation of the reprojected distributions
 *     to take account of the sliding displacement as a function of time.
 *
 *  Note we never deal with the halo regions here, as we assume the
 *  upcoming propagation will be immediately preceeded by a distribution
 *  halo update.
 *
 *****************************************************************************/

void model_le_apply_boundary_conditions(void) {

  const int irepro = 0;

  if (le_get_nplane_local() > 0) {

    TIMER_start(TIMER_LE);

    if (irepro == 0) le_reproject();
    if (irepro != 0) le_reproject_all();

    if (cart_size(Y) > 1) {
      le_displace_and_interpolate_parallel();
    }
    else {
      le_displace_and_interpolate();
    }

    TIMER_stop(TIMER_LE);
  }

  return;
}

/*****************************************************************************
 *
 *  le_reproject
 *
 *  This is the reprojection of the post collision distributions to
 *  take account of the velocity jump at the planes.
 *
 *  We compute the moments, and then the change to the moments:
 *
 *     rho  -> rho (unchanged)
 *     g_a  -> g_a +/- rho u^le_a
 *     S_ab -> S_ab +/- rho u_a u^le_b +/- rho u_b u^le_a + rho u^le_a u^le_b
 *
 *  with analogous expressions for order parameter moments.
 * 
 *  The change to the distribution is then computed by a reprojection.
 *  Ghost modes are unchanged.
 * 	    	  
 *****************************************************************************/

static void le_reproject(void) {

  int    ic, jc, kc, index;
  int    nplane, plane, side;
  int    ia, ib;
  int    nlocal[3];
  int    poffset, np, p;
  int    n, ndist;

  double rho, ds[3][3], udotc, sdotq;
  double g[3], du[3];
  double fnew;
  double t;

  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  assert(CVXBLOCK == 1);

  ndist  = distribution_ndist(); 
  nplane = le_get_nplane_local();

  t = 1.0*get_step();
  coords_nlocal(nlocal);

  for (plane = 0; plane < nplane; plane++) {
    for (side = 0; side < 2; side++) {

      du[X] = 0.0;
      du[Y] = 0.0; 
      du[Z] = 0.0;

      if (side == 0) {
	/* Start with plane below Lees-Edwards BC */
	du[Y] = -le_plane_uy(t);
	ic = le_plane_location(plane);
	poffset = xdisp_fwd_cv[0];
      }
      else {
	/* Finally, deal with plane above LEBC */
	du[Y] = +le_plane_uy(t);
	ic = le_plane_location(plane) + 1;
	poffset = xdisp_bwd_cv[0];
      }

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  
	  index = le_site_index(ic, jc, kc);

	  for (n = 0; n < ndist; n++) {

	    /* Compute 0th and 1st moments */

	    rho = distribution_zeroth_moment(index, n);
	    distribution_first_moment(index, n, g);

	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		ds[ia][ib] = (g[ia]*du[ib] + du[ia]*g[ib] + rho*du[ia]*du[ib]);
	      }
	    }

	    /* Now update the distribution */

	    for (np = 0; np < xblocklen_cv[0]; np++) {

	      /* Pick up the correct velocity indices */ 
	      p = poffset + np;

	      udotc = du[Y]*cv[p][Y];
	      sdotq = 0.0;
	      
	      for (ia = 0; ia < 3; ia++) {
		for (ib = 0; ib < 3; ib++) {
		  sdotq += ds[ia][ib]*q_[p][ia][ib];
		}
	      }

	      /* Project all this back to the distribution. */

	      fnew = distribution_f(index, p, n);
	      fnew += wv[p]*(rho*udotc*rcs2 + sdotq*r2rcs4);
	      distribution_f_set(index, p, n, fnew);
	    }
	  }
	  /* next site */
	}
      }
    }
  }

  return;
}


/****************************************************************************
 *
 *  le_reproject_all
 *
 *  An experiemental routine to reproject the ghost currents in
 *  addition to the hydrodynamic modes.
 *
 *  D3Q19
 *
 *  jchi1[Y] = chi1 rho u_y -> chi1 (rho u_y +/- rho u_le)
 *  jchi2[Y] = chi2 rho u_y -> chi2 (rho u_y +/- rho u_le)
 *
 ****************************************************************************/
 
static void le_reproject_all(void) {

  int    ic, jc, kc, index;
  int    nplane, plane, side;
  int    nlocal[3];
  int    p, m, np;
  int    poffset, ndist;

  double mode[NVEL];
  double rho;
  double g[3], du[3];
  double t;

  extern double * f_;

  assert(CVXBLOCK == 1);

  ndist  = distribution_ndist(); 
  nplane = le_get_nplane_local();

  t = 1.0*get_step();
  coords_nlocal(nlocal);

  for (plane = 0; plane < nplane; plane++) {
    for (side = 0; side < 2; side++) {

      du[X] = 0.0;
      du[Y] = 0.0; 
      du[Z] = 0.0;

      if (side == 0) {
	/* Start with plane below Lees-Edwards BC */
	du[Y] = -le_plane_uy(t);
	ic = le_plane_location(plane);
	poffset = xdisp_fwd_cv[0];
      }
      else {
	/* Finally, deal with plane above LEBC */
	du[Y] = +le_plane_uy(t);
	ic = le_plane_location(plane) + 1;
	poffset = xdisp_bwd_cv[0];
      }

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  
	  index = le_site_index(ic, jc, kc);

	  /* Compute modes */

	  for (m = 0; m < NVEL; m++) {
	    mode[m] = 0.0;
	    for (p = 0; p < NVEL; p++) {
	      mode[m] += f_[ndist*NVEL*index + 0 + p]*ma_[m][p];
	    }
	  }

	  /* Transform */

	  rho = mode[0];
	  g[X] = mode[1];
	  g[Y] = mode[2];
	  g[Z] = mode[3];

	  mode[2] = mode[2] + rho*du[Y];
	  mode[5] = mode[5] + g[X]*du[Y];
	  mode[7] = mode[7] + 2.0*g[Y]*du[Y] + rho*du[Y]*du[Y];
	  mode[8] = mode[8] + g[Z]*du[Y];

	  /* All ghosts unaltered */

	  /* Reproject */

	  for (np = 0; np < xblocklen_cv[0]; np++) {
	    p = poffset + np;
	    f_[ndist*NVEL*index + 0 + p] = 0.0;
	    for (m = 0; m < NVEL; m++) {
	      f_[ndist*NVEL*index + 0 + p] += mode[m]*mi_[p][m];
	    }
	  }

	  /* next site */
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  le_displace_and_interpolate
 *
 *  For each side of each plane, work out the relevant displacement
 *  and do the necessary interpolation to get the modified plane-
 *  crossing distributions.
 *
 *****************************************************************************/

void le_displace_and_interpolate(void) {

  int    ic, jc, kc;
  int    index0, index1, i0, i1;
  int    nlocal[3];
  int    n, nplane, plane;
  int    p;
  int    jdy, j1, j2;
  int    ndist;
  int    nprop;
  int    ndata;
  int    nhalo;
  double dy, fr;
  double t;
  double * recv_buff;

  extern double * f_;

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nplane = le_get_nplane_local();

  t = 1.0*get_step();

  /* We need to interpolate into a temporary buffer to make sure we
   * don't overwrite distributions taking part. The size is just
   * determined by the size of the local domain, and the number
   * of plane-crossing distributions. */

  ndist = distribution_ndist();
  nprop = xblocklen_cv[0];
  ndata = ndist*nprop*nlocal[Y]*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  if(recv_buff == NULL) fatal("malloc(recv_buff) failed\n");

  for (plane = 0; plane < nplane; plane++) {
 
    ic  = le_plane_location(plane);

    dy  = le_buffer_displacement(nhalo, t);
    dy  = fmod(dy, L(Y));
    jdy = floor(dy);
    fr = dy - jdy;

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      j2 = 1 + (j1 % nlocal[Y]);

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, j1, kc);
	index1 = le_site_index(ic, j2, kc);
		  
	/* xdisp_fwd_cv[0] identifies cv[p][X] = +1 */

	for (n = 0; n < ndist; n++) {
	  i0 = ndist*NVEL*index0 + n*NVEL + xdisp_fwd_cv[0];
	  i1 = ndist*NVEL*index1 + n*NVEL + xdisp_fwd_cv[0];
	  for (p = 0; p < nprop; p++) {
	    recv_buff[ndata++] = (1.0 - fr)*f_[i0 + p] + fr*f_[i1 + p];
	  }
	}
	/* Next site */
      }
    }

    /* ...and copy back ... */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  i0 = ndist*NVEL*index0 + n*NVEL + xdisp_fwd_cv[0];
	  for (p = 0; p < nprop; p++) {
	    f_[i0 + p] = recv_buff[ndata++];
	  }
	}
	/* Next site */
      }
    }


    /* OTHER DIRECTION */
 
    ic  = le_plane_location(plane) + 1;

    dy  = -le_buffer_displacement(nhalo, t);
    dy  = fmod(dy, L(Y));
    jdy = floor(dy);
    fr = dy - jdy;

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      j2 = 1 + (j1 % nlocal[Y]) ;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, j1, kc);
	index1 = le_site_index(ic, j2, kc);

	for (n = 0; n < ndist; n++) {
	  i0 = ndist*NVEL*index0 + n*NVEL + xdisp_bwd_cv[0];
	  i1 = ndist*NVEL*index1 + n*NVEL + xdisp_bwd_cv[0];
	  for (p = 0; p < nprop; p++) {
	    recv_buff[ndata++] = (1.0 - fr)*f_[i0 + p] + fr*f_[i1 + p];
	  }
	}
	/* Next site */
      }
    }

    /* ...and now overwrite... */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = le_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  i0 = ndist*NVEL*index0 + n*NVEL + xdisp_bwd_cv[0];
	  for (p = 0; p < nprop; p++) {
	    f_[i0 + p] = recv_buff[ndata++];
	  }
	}
      }
    }

    /* Next plane */
  }

  free(recv_buff);

  return;
}

/*****************************************************************************
 *
 *  le_displace_and_interpolate_parallel
 *
 *  Here we need to communicate to be able to do the displacement of
 *  the buffers in the along-plane (Y-) direction.
 *
 *  Locally, we need to find interpolated values of the plane-crossing
 *  distributions for 1 <= jc <= nlocal[Y]. To do a linear interpolation
 *  everywhere, this requires (nlocal[Y] + 1) points displaced in the
 *  appropriate direction.
 *
 *  Likewise, we need to send a total of (nlocal[Y] + 1) points to the
 *  two corresponding recieving processes. Note we never involve the
 *  halo regions here (so a preceeding halo exchange is not required). 
 *
 *****************************************************************************/

static void le_displace_and_interpolate_parallel() {

  int ic, jc, kc;
  int j1, j1mod;
  int jdy;
  int n1, n2;
  int ndata, ndata1, ndata2;
  int nhalo;
  int ind0, ind1, ind2, index, i0;
  int n, nplane, plane;
  int p;
  int nlocal[3];
  int offset[3];
  int nrank_s[3], nrank_r[3];
  int nprop;
  int ndist;

  const int tag1 = 3102;
  const int tag2 = 3103;

  double fr;
  double dy;
  double t;
  double * send_buff;
  double * recv_buff;

  MPI_Comm    comm;
  MPI_Request req[4];
  MPI_Status status[4];

  extern double * f_;

  assert(CVXBLOCK == 1);

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  coords_nlocal_offset(offset);
  nplane = le_get_nplane_local();

  comm = le_communicator();

  t = 1.0*get_step();
  ndist = distribution_ndist();
  nprop = xblocklen_cv[0];

  ndata = ndist*nprop*nlocal[Y]*nlocal[Z];
  send_buff = (double *) malloc(ndata*sizeof(double));
  if (send_buff == NULL) fatal("malloc(send_buff) failed\n");

  ndata = ndist*nprop*(nlocal[Y] + 1)*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  if (recv_buff == NULL) fatal("malloc(recv_buff) failed\n");

  for (plane = 0; plane < nplane; plane++) {

    ic  = le_plane_location(plane);

    dy  = le_buffer_displacement(nhalo, t);
    dy  = fmod(dy, L(Y));
    jdy = floor(dy);
    fr  = dy - jdy;

    /* Starting y coordinate is j1: 1 <= j1 <= N_total.y */

    jc = offset[Y] + 1;
    j1 = 1 + (jc + jdy - 1 + 2*N_total(Y)) % N_total(Y);
    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    j1mod = 1 + (j1 - 1) % nlocal[Y];
    n1 = (nlocal[Y] - j1mod + 1);
    n2 = j1mod;

    ndata1 = n1*nlocal[Z]*ndist*nprop;
    ndata2 = n2*nlocal[Z]*ndist*nprop;

    /* Post the receives */

    MPI_Irecv(recv_buff, ndata1, MPI_DOUBLE, nrank_r[0], tag1, comm, req);
    MPI_Irecv(recv_buff + ndata1, ndata2, MPI_DOUBLE, nrank_r[1], tag2,
	      comm, req + 1);

    /* Load the send buffer. Note that data at j1mod gets sent to both
     * receivers, making up the total of (nlocal[Y] + 1) points */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	/* cv[p][X] = +1 identified by disp_fwd[] */
	index = le_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  i0 = ndist*NVEL*index + n*NVEL + xdisp_fwd_cv[0];
	  for (p = 0; p < nprop; p++) {
	    send_buff[ndata++] = f_[i0 + p];
	  }
	}
	/* Next site */
      }
    }

    ndata = ndata2 - nlocal[Z]*ndist*nprop;

    MPI_Issend(send_buff + ndata, ndata1, MPI_DOUBLE, nrank_s[0], tag1,
	       comm, req + 2);
    MPI_Issend(send_buff,         ndata2, MPI_DOUBLE, nrank_s[1], tag2,
	       comm, req + 3);

    /* Wait for the receives, and sort out the interpolated values */

    MPI_Waitall(2, req, status);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);
	ind0 = ndist*nprop*((jc-1)*nlocal[Z] + (kc-1));

	for (n = 0; n < ndist; n++) {
	  i0   = ndist*NVEL*index + n*NVEL + xdisp_fwd_cv[0];
	  ind1 = ind0 + n*nprop;
	  ind2 = ind0 + ndist*nprop*nlocal[Z] + n*nprop;
	  for (p = 0; p < nprop; p++) {
	    f_[i0 + p] = (1.0-fr)*recv_buff[ind1 + p] + fr*recv_buff[ind2 + p];
	  }
	}
	/* Next site */
      }
    }

    /* Finish the sends */
    MPI_Waitall(2, req + 2, status);



    /* NOW THE OTHER DIRECTION */

    ic  = le_plane_location(plane) + 1;

    dy  = -le_buffer_displacement(nhalo, t);
    dy  = fmod(dy, L(Y));
    jdy = floor(dy);
    fr  = dy - jdy;

    /* Starting y coordinate (global address): range 1 <= j1 <= N_total.y */

    jc = offset[Y] + 1;
    j1 = 1 + (jc + jdy - 1 + 2*N_total(Y)) % N_total(Y);
    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    j1mod = 1 + (j1 - 1) % nlocal[Y];
    n1 = (nlocal[Y] - j1mod + 1);
    n2 = j1mod;

    ndata1 = n1*nlocal[Z]*ndist*nprop;
    ndata2 = n2*nlocal[Z]*ndist*nprop;

    /* Post the receives */

    MPI_Irecv(recv_buff, ndata1, MPI_DOUBLE, nrank_r[0], tag1, comm, req);
    MPI_Irecv(recv_buff + ndata1, ndata2, MPI_DOUBLE, nrank_r[1], tag2,
	      comm, req + 1);

    /* Load the send buffer. Note that data at j1mod gets sent to both
     * receivers, making up the total of (nlocal[Y] + 1) points */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	/* cv[p][X] = -1 identified by disp_bwd[] */
	index = le_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  i0 = ndist*NVEL*index + n*NVEL + xdisp_bwd_cv[0];
	  for (p = 0; p < nprop; p++) {
	    send_buff[ndata++] = f_[i0 + p];
	  }
	}
	/* Next site */
      }
    }

    ndata = ndata2 - nlocal[Z]*ndist*nprop;

    MPI_Issend(send_buff + ndata, ndata1, MPI_DOUBLE, nrank_s[0], tag1,
	       comm, req + 2);
    MPI_Issend(send_buff,         ndata2, MPI_DOUBLE, nrank_s[1], tag2,
	       comm, req + 3);

    /* Wait for the receives, and interpolate from the buffer */

    MPI_Waitall(2, req, status);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = le_site_index(ic, jc, kc);
	ind0 = ndist*nprop*((jc-1)*nlocal[Z] + (kc-1));

	for (n = 0; n < ndist; n++) {
	  i0   = ndist*NVEL*index + n*NVEL + xdisp_bwd_cv[0];
	  ind1 = ind0 + n*nprop;
	  ind2 = ind0 + ndist*nprop*nlocal[Z] + n*nprop;
	  for (p = 0; p < nprop; p++) {
	    f_[i0 + p] = (1.0-fr)*recv_buff[ind1 + p] + fr*recv_buff[ind2 + p];
	  }
	}
	/* Next site */
      }
    }

    /* Mop up the sends */
    MPI_Waitall(2, req + 2, status);
  }

  free(send_buff);
  free(recv_buff);

  return;
}

/*****************************************************************************
 *
 *  model_le_init_shear_profile
 *
 *  Initialise the distributions to be consistent with a steady-state
 *  linear shear profile, consistent with plane velocity.
 *
 *****************************************************************************/

void model_le_init_shear_profile() {

  int ic, jc, kc, index;
  int i, j, p;
  int N[3];
  double rho, u[NDIM], gradu[NDIM][NDIM];
  double eta;

  info("Initialising shear profile\n");

  /* Initialise the density, velocity, gradu; ghost modes are zero */

  rho = get_rho0();
  eta = get_eta_shear();
  coords_nlocal(N);

  for (i = 0; i< NDIM; i++) {
    u[i] = 0.0;
    for (j = 0; j < NDIM; j++) {
      gradu[i][j] = 0.0;
    }
  }

  gradu[X][Y] = le_shear_rate();

  /* Loop trough the sites */

  for (ic = 1; ic <= N[X]; ic++) {
      
    u[Y] = le_get_steady_uy(ic);

    /* We can now project the physical quantities to the distribution */

    for (jc = 1; jc <= N[Y]; jc++) {
      for (kc = 1; kc <= N[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (p = 0; p < NVEL; p++) {
	  double f = 0.0;
	  double cdotu = 0.0;
	  double sdotq = 0.0;

	  for (i = 0; i < NDIM; i++) {
	    cdotu += cv[p][i]*u[i];
	    for (j = 0; j < NDIM; j++) {
	      sdotq += (rho*u[i]*u[j] - eta*gradu[i][j])*q_[p][i][j];
	    }
	  }
	  f = wv[p]*(rho + rcs2*rho*cdotu + 0.5*rcs2*rcs2*sdotq);
	  distribution_f_set(index, p, 0, f);
	}
	/* Next site */
      }
    }
  }

  return;
}
