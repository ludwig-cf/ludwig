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
 *  $Id: model_le.c,v 1.5 2009-08-20 16:39:22 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
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

extern Site * site;

static void le_reproject(void);
static void le_displace_and_interpolate(void);
static void le_displace_and_interpolate_parallel(void);

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

  if (le_get_nplane_local() > 0) {

    TIMER_start(TIMER_LE);

    le_reproject();

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
 *  The change to the distribution is then computed by a reprojection
 *  assuming the ghost modes are zero.
 * 	    	  
 *****************************************************************************/

static void le_reproject(void) {

  int    ic, jc, kc, index;
  int    nplane, plane, side;
  int    ia, ib;
  int    nlocal[3];
  int    poffset, np, p;

  double rho, phi, ds[3][3], dsphi[3][3], udotc, jdotc, sdotq, sphidotq;
  double u[3], jphi[3], du[3], djphi[3];
  double LE_vel;
  double t;

  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */


  assert(CVXBLOCK == 1);

  nplane = le_get_nplane_local();

  t = 1.0*get_step();
  get_N_local(nlocal);

  for (plane = 0; plane < nplane; plane++) {
    for (side = 0; side < 2; side++) {

      /* Start with plane below Lees-Edwards BC */

      if (side == 0) {
	LE_vel =-le_plane_uy(t);
	ic = le_plane_location(plane);
	poffset = xdisp_fwd_cv[0];
      }
      else {
	/* Finally, deal with plane above LEBC */
	LE_vel =+le_plane_uy(t);
	ic = le_plane_location(plane) + 1;
	poffset = xdisp_bwd_cv[0];
      }

      /* First, for plane `below' LE plane, ie, crossing LE plane going up */

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  
	  index = ADDR(ic, jc, kc);

	  /* Compute 0th and 1st moments */

	  rho = site[index].f[0];
	  phi = site[index].g[0];

	  for (ia = 0; ia < 3; ia++) {
	    u[ia] = 0.0;
	    jphi[ia] = 0.0;
	    du[ia] = 0.0;
	    djphi[ia] = 0.0;
	  }

	  for (p = 1; p < NVEL; p++) {
	    rho    += site[index].f[p];
	    phi    += site[index].g[p];
	    for (ia = 0; ia < 3; ia++) {
	      u[ia] += site[index].f[p]*cv[p][ia];
	      jphi[ia] += site[index].g[p]*cv[p][ia];
	    }
	  }

	  /* ... then for the momentum (note that whilst jphi[] represents */
	  /* the moment, u only represents the momentum) */

	  du[Y] = LE_vel; 
	  djphi[Y] = phi*LE_vel;
 
	  for (ia = 0; ia < 3; ia++) {
	    for (ib = 0; ib < 3; ib++) {
	      ds[ia][ib] = rho*(u[ia]*du[ib] + du[ia]*u[ib] + du[ia]*du[ib]);
	      dsphi[ia][ib] =
		du[ia]*jphi[ib] + jphi[ia]*du[ib] + phi*du[ia]*du[ib];
	    }
	  }

	  /* Now update the distribution */

	  for (np = 0; np < xblocklen_cv[0]; np++) {

	    /* Pick up the correct velocity indices */ 
	    p = poffset + np;

	    udotc =    du[Y]*cv[p][Y];
	    jdotc = djphi[Y]*cv[p][Y];
	    
	    sdotq    = 0.0;
	    sphidotq = 0.0;
	      
	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		sdotq += ds[ia][ib]*q_[p][ia][ib];
		sphidotq += dsphi[ia][ib]*q_[p][ia][ib];
	      }
	    }
	      
	    /* Project all this back to the distributions. */

	    site[index].f[p] += wv[p]*(rho*udotc*rcs2 + sdotq*r2rcs4);
	    site[index].g[p] += wv[p]*(    jdotc*rcs2 + sphidotq*r2rcs4);
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
  double rho, u[ND], gradu[ND][ND];
  double eta;

  info("Initialising shear profile\n");

  /* Initialise the density, velocity, gradu; ghost modes are zero */

  rho = get_rho0();
  eta = get_eta_shear();
  get_N_local(N);

  for (i = 0; i< ND; i++) {
    u[i] = 0.0;
    for (j = 0; j < ND; j++) {
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

	index = get_site_index(ic, jc, kc);

	for (p = 0; p < NVEL; p++) {
	  double f = 0.0;
	  double cdotu = 0.0;
	  double sdotq = 0.0;

	  for (i = 0; i < ND; i++) {
	    cdotu += cv[p][i]*u[i];
	    for (j = 0; j < ND; j++) {
	      sdotq += (rho*u[i]*u[j] - eta*gradu[i][j])*q_[p][i][j];
	    }
	  }
	  f = wv[p]*(rho + rcs2*rho*cdotu + 0.5*rcs2*rcs2*sdotq);
	  set_f_at_site(index, p, f);
	}
	/* Next site */
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
  int    index0, index1;
  int    nlocal[3];
  int    n, nplane;
  int    p;
  int    jdy, j1, j2;
  int    ndist;
  int    ndata;
  double dy, fr;
  double t;
  double * recv_buff;

  get_N_local(nlocal);
  nplane = le_get_nplane_local();

  t = 1.0*get_step();

  /* We need to interpolate into a temporary buffer to make sure we
   * don't overwrite distributions taking part. The size is just
   * determined by the size of the local domain, and the number
   * of plane-crossing distributions. */

  ndist = xblocklen_cv[0];
  ndata = 2*ndist*nlocal[Y]*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  if(recv_buff == NULL) fatal("malloc(recv_buff) failed\n");

  for (n = 0; n < nplane; n++) {
 
    ic  = le_plane_location(n);

    dy  = le_buffer_displacement(nhalo_, t);
    dy  = fmod(dy, L(Y));
    jdy = floor(dy);
    fr = dy - jdy;

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      j2 = 1 + (j1 % nlocal[Y]);

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, j1, kc);
	index1 = ADDR(ic, j2, kc);
		  
	/* xdisp_fwd_cv[0] identifies cv[p][X] = +1 */

	for (p = 0; p < ndist; p++) {
	  recv_buff[ndata++] = (1.0 - fr)*site[index0].f[xdisp_fwd_cv[0] + p] 
	    + fr*site[index1].f[xdisp_fwd_cv[0] + p];
	}
	for (p = 0; p < ndist; p++) {
	  recv_buff[ndata++] = (1.0 - fr)*site[index0].g[xdisp_fwd_cv[0] + p] 
	    + fr*site[index1].g[xdisp_fwd_cv[0] + p];
	}
      }
    }

    /* ...and copy back ... */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, jc, kc);
	for (p = 0; p < ndist; p++) {
	  site[index0].f[xdisp_fwd_cv[0] + p] = recv_buff[ndata++]; 
	}
	for (p = 0; p < ndist; p++) {
	  site[index0].g[xdisp_fwd_cv[0] + p] = recv_buff[ndata++]; 
	}
      }
    }


    /* OTHER DIRECTION */
 
    ic  = le_plane_location(n) + 1;

    dy  = -le_buffer_displacement(nhalo_, t);
    dy  = fmod(dy, L(Y));
    jdy = floor(dy);
    fr = dy - jdy;

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      j2 = 1 + (j1 % nlocal[Y]) ;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, j1, kc);
	index1 = ADDR(ic, j2, kc);

	for (p = 0; p < ndist; p++) {
	  recv_buff[ndata++] = (1.0 - fr)*site[index0].f[xdisp_bwd_cv[0] + p]
	    + fr*site[index1].f[xdisp_bwd_cv[0] + p];
	}
	for (p = 0; p < ndist; p++) {
	  recv_buff[ndata++] = (1.0 - fr)*site[index0].g[xdisp_bwd_cv[0] + p]
	    + fr*site[index1].g[xdisp_bwd_cv[0] + p];
	}
      }
    }

    /* ...and now overwrite... */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = ADDR(ic, jc, kc);
	for (p = 0; p < ndist; p++) {
	  site[index0].f[xdisp_bwd_cv[0] + p] = recv_buff[ndata++];
	}
	for (p = 0; p < ndist; p++) {
	  site[index0].g[xdisp_bwd_cv[0] + p] = recv_buff[ndata++];
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
  int ind0, ind1, ind2, index;
  int n, nplane;
  int p;
  int nlocal[3];
  int offset[3];
  int  nrank_s[2], nrank_r[2];

  const int tag1 = 3102;
  const int tag2 = 3103;
  int ndist;

  double fr;
  double dy;
  double t;
  double * send_buff;
  double * recv_buff;

  MPI_Request req[4];
  MPI_Status status[4];
  MPI_Comm comm = le_communicator();

  assert(CVXBLOCK == 1);

  get_N_local(nlocal);
  get_N_offset(offset);
  nplane = le_get_nplane_local();

  t = 1.0*get_step();
  ndist = xblocklen_cv[0];

  ndata = 2*ndist*nlocal[Y]*nlocal[Z];
  send_buff = (double *) malloc(ndata*sizeof(double));
  if (send_buff == NULL) fatal("malloc(send_buff) failed\n");

  ndata = 2*ndist*(nlocal[Y] + 1)*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  if (recv_buff == NULL) fatal("malloc(recv_buff) failed\n");

  for (n = 0; n < nplane; n++) {

    ic  = le_plane_location(n);

    dy  = le_buffer_displacement(nhalo_, t);
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

    ndata1 = n1*nlocal[Z]*2*ndist;
    ndata2 = n2*nlocal[Z]*2*ndist;

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
	index = ADDR(ic, jc, kc);
	for (p = 0; p < ndist; p++) {
	  send_buff[ndata++] = site[index].f[xdisp_fwd_cv[0] + p];
	}
	for (p = 0; p < ndist; p++) {
	  send_buff[ndata++] = site[index].g[xdisp_fwd_cv[0] + p];
	}
      }
    }

    ndata = ndata2 - nlocal[Z]*2*ndist;

    MPI_Issend(send_buff + ndata, ndata1, MPI_DOUBLE, nrank_s[0], tag1,
	       comm, req + 2);
    MPI_Issend(send_buff,         ndata2, MPI_DOUBLE, nrank_s[1], tag2,
	       comm, req + 3);

    /* Wait for the receives, and sort out the interpolated values */

    MPI_Waitall(2, req, status);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = ADDR(ic, jc, kc);
	ind0 = 2*ndist*((jc-1)*nlocal[Z] + (kc-1));
	for (p = 0; p < ndist; p++) {
	  ind1 = ind0 + p;
	  ind2 = ind0 + 2*ndist*nlocal[Z] + p;
	  site[index].f[xdisp_fwd_cv[0] + p] = (1.0 - fr)*recv_buff[ind1]
	    + fr*recv_buff[ind2];
	}
	for (p = 0; p < ndist; p++) {
	  ind1 = ind0 + ndist + p;
	  ind2 = ind0 + 2*ndist*nlocal[Z] + ndist + p;
	  site[index].g[xdisp_fwd_cv[0] + p] = (1.0 - fr)*recv_buff[ind1]
	    + fr*recv_buff[ind2];
	}
      }
    }

    /* Finish the sends */
    MPI_Waitall(2, req + 2, status);



    /* NOW THE OTHER DIRECTION */

    ic  = le_plane_location(n) + 1;

    dy  = -le_buffer_displacement(nhalo_, t);
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

    ndata1 = n1*nlocal[Z]*2*ndist;
    ndata2 = n2*nlocal[Z]*2*ndist;

    /* Post the receives */

    MPI_Irecv(recv_buff, ndata1, MPI_DOUBLE, nrank_r[0], tag1, comm, req);
    MPI_Irecv(recv_buff + ndata1, ndata2, MPI_DOUBLE, nrank_r[1], tag2,
	      comm, req + 1);

    /* Load the send buffer. Note that data at j1mod gets sent to both
     * receivers, making up the total of (nlocal[Y] + 1) points */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	/* cv[p][X] = -1 identified by disp_fwd[] */
	index = ADDR(ic, jc, kc);
	for (p = 0; p < ndist; p++) {
	  send_buff[ndata++] = site[index].f[xdisp_bwd_cv[0] + p];
	}
	for (p = 0; p < ndist; p++) {
	  send_buff[ndata++] = site[index].g[xdisp_bwd_cv[0] + p];
	}
      }
    }

    ndata = ndata2 - nlocal[Z]*2*ndist;

    MPI_Issend(send_buff + ndata, ndata1, MPI_DOUBLE, nrank_s[0], tag1,
	       comm, req + 2);
    MPI_Issend(send_buff,         ndata2, MPI_DOUBLE, nrank_s[1], tag2,
	       comm, req + 3);

    /* Wait for the receives, and interpolate from the buffer */

    MPI_Waitall(2, req, status);

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
	index = ADDR(ic, jc, kc);
	ind0 = 2*ndist*((jc-1)*nlocal[Z] + (kc-1));
	for (p = 0; p < ndist; p++) {
	  ind1 = ind0 + p;
	  ind2 = ind0 + 2*ndist*nlocal[Z] + p;
	  site[index].f[xdisp_bwd_cv[0] + p] = (1.0 - fr)*recv_buff[ind1]
	    + fr*recv_buff[ind2];
	}
	for (p = 0; p < ndist; p++) {
	  ind1 = ind0 + ndist + p;
	  ind2 = ind0 + 2*ndist*nlocal[Z] + ndist + p;
	  site[index].g[xdisp_bwd_cv[0] + p] = (1.0 - fr)*recv_buff[ind1]
	    + fr*recv_buff[ind2];
	}
      }
    }

    /* Mop up the sends */
    MPI_Waitall(2, req + 2, status);
  }

  free(send_buff);
  free(recv_buff);

  return;
}
