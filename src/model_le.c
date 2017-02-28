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
 *  Issue: a 'MODEL_R' implementation of communication is required
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  J.-C. Desplat and Ronojoy Adhikari developed the reprojection method.
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "control.h"
#include "lb_model_s.h"
#include "physics.h"
#include "leesedwards.h"

static int le_reproject(lb_t * lb, lees_edw_t * le);
static int le_reproject_all(lb_t * lb, lees_edw_t * le);
static int le_displace_and_interpolate(lb_t * lb, lees_edw_t * le);
static int le_displace_and_interpolate_parallel(lb_t * lb, lees_edw_t * le);

/*****************************************************************************
 *
 *  lb_le_apply_boundary_conditions
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

__host__ int lb_le_apply_boundary_conditions(lb_t * lb, lees_edw_t * le) {

  const int irepro = 0;
  int mpi_cartsz[3];

  assert(lb);
  assert(le);

  lees_edw_cartsz(le, mpi_cartsz);

  if (lees_edw_nplane_local(le) > 0) {

    TIMER_start(TIMER_LE);

    if (irepro == 0) le_reproject(lb, le);
    if (irepro != 0) le_reproject_all(lb, le);

    if (mpi_cartsz[Y] > 1) {
      le_displace_and_interpolate_parallel(lb, le);
    }
    else {
      le_displace_and_interpolate(lb, le);
    }

    TIMER_stop(TIMER_LE);
  }

  return 0;
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

static int le_reproject(lb_t * lb, lees_edw_t * le) {

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
  physics_t * phys = NULL;

  const double r2rcs4 = 4.5;         /* The constant 1 / 2 c_s^4 */

  assert(lb);
  assert(le);
  assert(CVXBLOCK == 1);

  lb_ndist(lb, &ndist);
  nplane = lees_edw_nplane_local(le);
  physics_ref(&phys);

  t = 1.0*physics_control_timestep(phys);
  lees_edw_nlocal(le, nlocal);

  for (plane = 0; plane < nplane; plane++) {
    for (side = 0; side < 2; side++) {

      du[X] = 0.0;
      du[Y] = 0.0; 
      du[Z] = 0.0;

      if (side == 0) {
	/* Start with plane below Lees-Edwards BC */
	lees_edw_plane_uy_now(le, t, &du[Y]);
	du[Y] *= -1.0;
	ic = lees_edw_plane_location(le, plane);
	poffset = xdisp_fwd_cv[0];
      }
      else {
	/* Finally, deal with plane above LEBC */
	lees_edw_plane_uy_now(le, t, &du[Y]);
	ic = lees_edw_plane_location(le, plane) + 1;
	poffset = xdisp_bwd_cv[0];
      }

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  
	  index = lees_edw_index(le, ic, jc, kc);

	  for (n = 0; n < ndist; n++) {

	    /* Compute 0th and 1st moments */
	    lb_dist_enum_t ndn = (lb_dist_enum_t) n;
	    lb_0th_moment(lb, index, ndn, &rho);
	    lb_1st_moment(lb, index, ndn, g);

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

	      lb_f(lb, index, p, n, &fnew);
	      fnew += wv[p]*(rho*udotc*rcs2 + sdotq*r2rcs4);
	      lb_f_set(lb, index, p, n, fnew);
	    }
	  }
	  /* next site */
	}
      }
    }
  }

  return 0;
}


/****************************************************************************
 *
 *  le_reproject_all
 *
 *  An experiemental routine to reproject the ghost currents in
 *  addition to the hydrodynamic modes.
 *
 *  D3Q19, single fluid
 *
 *  jchi1[Y] = chi1 rho u_y -> chi1 (rho u_y +/- rho u_le)
 *  jchi2[Y] = chi2 rho u_y -> chi2 (rho u_y +/- rho u_le)
 *
 ****************************************************************************/
 
static int le_reproject_all(lb_t * lb, lees_edw_t * le) {

  int    ic, jc, kc, index;
  int    nplane, plane, side;
  int    nlocal[3];
  int    p, m, np;
  int    poffset, ndist;

  double mode[NVEL];
  double rho;
  double g[3], du[3];
  double t;
  physics_t * phys = NULL;

  assert(le);
  assert(lb);
  assert(CVXBLOCK == 1);

  lb_ndist(lb, &ndist);
  nplane = lees_edw_nplane_local(le);
  physics_ref(&phys);

  t = 1.0*physics_control_timestep(phys);
  lees_edw_nlocal(le, nlocal);

  for (plane = 0; plane < nplane; plane++) {
    for (side = 0; side < 2; side++) {

      du[X] = 0.0;
      du[Y] = 0.0; 
      du[Z] = 0.0;

      if (side == 0) {
	/* Start with plane below Lees-Edwards BC */
	lees_edw_plane_uy_now(le, t, &du[Y]);
	du[Y] = -du[Y];
	ic = lees_edw_plane_location(le, plane);
	poffset = xdisp_fwd_cv[0];
      }
      else {
	/* Finally, deal with plane above LEBC */
	lees_edw_plane_uy_now(le, t, &du[Y]);
	ic = lees_edw_plane_location(le, plane) + 1;
	poffset = xdisp_bwd_cv[0];
      }

      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {
	  
	  index = lees_edw_index(le, ic, jc, kc);

	  /* Compute modes */

	  for (m = 0; m < NVEL; m++) {
	    mode[m] = 0.0;
	    for (p = 0; p < NVEL; p++) {
	      mode[m] += lb->f[ndist*NVEL*index + 0 + p]*ma_[m][p];
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
	    lb->f[ndist*NVEL*index + 0 + p] = 0.0;
	    for (m = 0; m < NVEL; m++) {
	      lb->f[ndist*NVEL*index + 0 + p] += mode[m]*mi_[p][m];
	    }
	  }

	  /* next site */
	}
      }
    }
  }

  return 0;
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

int le_displace_and_interpolate(lb_t * lb, lees_edw_t * le) {

  int    ic, jc, kc;
  int    index0, index1;
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
  double ltot[3];
  double * recv_buff;
  physics_t * phys = NULL;

  assert(lb);
  assert(le);

  lees_edw_ltot(le, ltot);
  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);
  nplane = lees_edw_nplane_local(le);
  physics_ref(&phys);

  t = 1.0*physics_control_timestep(phys);

  /* We need to interpolate into a temporary buffer to make sure we
   * don't overwrite distributions taking part. The size is just
   * determined by the size of the local domain, and the number
   * of plane-crossing distributions. */

  lb_ndist(lb, &ndist);
  nprop = xblocklen_cv[0];
  ndata = ndist*nprop*nlocal[Y]*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  if(recv_buff == NULL) pe_fatal(lb->pe, "malloc(recv_buff) failed\n");

  for (plane = 0; plane < nplane; plane++) {
 
    ic  = lees_edw_plane_location(le, plane);

    lees_edw_buffer_displacement(le, nhalo, t, &dy);
    dy  = fmod(dy, ltot[Y]);
    jdy = floor(dy);
    fr = dy - jdy;

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      j2 = 1 + (j1 % nlocal[Y]);

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(le, ic, j1, kc);
	index1 = lees_edw_index(le, ic, j2, kc);
		  
	/* xdisp_fwd_cv[0] identifies cv[p][X] = +1 */

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < nprop; p++) {
	    int pcv = xdisp_fwd_cv[0] + p;
	    recv_buff[ndata++] = (1.0 - fr)*
	      lb->f[LB_ADDR(lb->nsite,ndist,NVEL,index0,n, pcv)]
	      + fr*
	      lb->f[LB_ADDR(lb->nsite,ndist,NVEL,index1,n, pcv)];
	  }
	}
	/* Next site */
      }
    }

    /* ...and copy back ... */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(le, ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < nprop; p++) {
	    int pcv = xdisp_fwd_cv[0] + p;
	    lb->f[LB_ADDR(lb->nsite, ndist, NVEL, index0, n, pcv)] = recv_buff[ndata++];
	  }
	}
	/* Next site */
      }
    }


    /* OTHER DIRECTION */
 
    ic  = lees_edw_plane_location(le, plane) + 1;

    lees_edw_buffer_displacement(le, nhalo, t, &dy);
    dy  = fmod(-dy, ltot[Y]);
    jdy = floor(dy);
    fr = dy - jdy;

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      j2 = 1 + (j1 % nlocal[Y]) ;

      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(le, ic, j1, kc);
	index1 = lees_edw_index(le, ic, j2, kc);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < nprop; p++) {
	    int pcv = xdisp_bwd_cv[0] + p;
	    recv_buff[ndata++] = (1.0 - fr)*
	      lb->f[LB_ADDR(lb->nsite,ndist,NVEL,index0, n, pcv)] + fr*
	      lb->f[LB_ADDR(lb->nsite, ndist, NVEL, index1, n, pcv)];
	  }
	}
	/* Next site */
      }
    }

    /* ...and now overwrite... */

    ndata = 0;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index0 = lees_edw_index(le, ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < nprop; p++) {
	    int pcv = xdisp_bwd_cv[0] + p;
	    lb->f[LB_ADDR(lb->nsite, ndist, NVEL, index0, n, pcv)] = recv_buff[ndata++];
	  }
	}
      }
    }

    /* Next plane */
  }

  free(recv_buff);

  return 0;
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

static int le_displace_and_interpolate_parallel(lb_t * lb, lees_edw_t * le) {

  int ic, jc, kc;
  int j1, j1mod;
  int jdy;
  int n1, n2;
  int ndata, ndata1, ndata2;
  int nhalo;
  int ind0, ind1, ind2, index;
  int n, nplane, plane;
  int p;
  int ntotal[3];
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
  double ltot[3];
  double * send_buff;
  double * recv_buff;

  physics_t * phys = NULL;
  MPI_Comm    comm;
  MPI_Request req[4];
  MPI_Status status[4];

  assert(lb);
  assert(le);
  assert(CVXBLOCK == 1);

  lees_edw_ltot(le, ltot);
  lees_edw_ntotal(le, ntotal);
  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);
  lees_edw_nlocal_offset(le, offset);

  nplane = lees_edw_nplane_local(le);
  lees_edw_comm(le, &comm);

  physics_ref(&phys);

  t = 1.0*physics_control_timestep(phys);
  lb_ndist(lb, &ndist);
  nprop = xblocklen_cv[0];

  ndata = ndist*nprop*nlocal[Y]*nlocal[Z];
  send_buff = (double *) malloc(ndata*sizeof(double));
  if (send_buff == NULL) pe_fatal(lb->pe, "malloc(send_buff) failed\n");

  ndata = ndist*nprop*(nlocal[Y] + 1)*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  if (recv_buff == NULL) pe_fatal(lb->pe, "malloc(recv_buff) failed\n");

  for (plane = 0; plane < nplane; plane++) {

    ic  = lees_edw_plane_location(le, plane);

    lees_edw_buffer_displacement(le, nhalo, t, &dy);
    dy  = fmod(dy, ltot[Y]);
    jdy = floor(dy);
    fr  = dy - jdy;

    /* Starting y coordinate is j1: 1 <= j1 <= ntotal[y] */

    jc = offset[Y] + 1;
    j1 = 1 + (jc + jdy - 1 + 2*ntotal[Y]) % ntotal[Y];
    lees_edw_jstart_to_mpi_ranks(le, j1, nrank_s, nrank_r);

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
	index = lees_edw_index(le, ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < nprop; p++) {
	    send_buff[ndata++] = lb->f[LB_ADDR(lb->nsite, ndist, NVEL, index, n, xdisp_fwd_cv[0] + p)];
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

	index = lees_edw_index(le, ic, jc, kc);
	ind0 = ndist*nprop*((jc-1)*nlocal[Z] + (kc-1));

	for (n = 0; n < ndist; n++) {
	  ind1 = ind0 + n*nprop;
	  ind2 = ind0 + ndist*nprop*nlocal[Z] + n*nprop;

	  for (p = 0; p < nprop; p++) {
	    lb->f[LB_ADDR(lb->nsite, ndist, NVEL, index, n, xdisp_fwd_cv[0] + p)] = (1.0-fr)*recv_buff[ind1 + p] + fr*recv_buff[ind2 + p];
	  }
	}
	/* Next site */
      }
    }

    /* Finish the sends */
    MPI_Waitall(2, req + 2, status);



    /* NOW THE OTHER DIRECTION */

    ic  = lees_edw_plane_location(le, plane) + 1;

    lees_edw_buffer_displacement(le, nhalo, t, &dy);
    dy  = fmod(-dy, ltot[Y]);
    jdy = floor(dy);
    fr  = dy - jdy;

    /* Starting y coordinate (global address): range 1 <= j1 <= ntotal[Y] */

    jc = offset[Y] + 1;
    j1 = 1 + (jc + jdy - 1 + 2*ntotal[Y]) % ntotal[Y];
    lees_edw_jstart_to_mpi_ranks(le, j1, nrank_s, nrank_r);

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
	index = lees_edw_index(le, ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < nprop; p++) {
	    send_buff[ndata++] = lb->f[LB_ADDR(lb->nsite, ndist, NVEL, index, n, xdisp_bwd_cv[0] + p)];
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

	index = lees_edw_index(le, ic, jc, kc);
	ind0 = ndist*nprop*((jc-1)*nlocal[Z] + (kc-1));

	for (n = 0; n < ndist; n++) {
	  ind1 = ind0 + n*nprop;
	  ind2 = ind0 + ndist*nprop*nlocal[Z] + n*nprop;

	  for (p = 0; p < nprop; p++) {
	    lb->f[LB_ADDR(lb->nsite, ndist, NVEL, index, n, xdisp_bwd_cv[0] + p)] = (1.0-fr)*recv_buff[ind1 + p] + fr*recv_buff[ind2 + p];
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

  return 0;
}

/*****************************************************************************
 *
 *  model_le_init_shear_profile
 *
 *  Initialise the distributions to be consistent with a steady-state
 *  linear shear profile, consistent with plane velocity.
 *
 *****************************************************************************/

int lb_le_init_shear_profile(lb_t * lb, lees_edw_t * le) {

  int ic, jc, kc, index;
  int i, j, p;
  int nlocal[3];
  double rho0, u[NDIM], gradu[NDIM][NDIM];
  double eta;
  physics_t * phys = NULL;

  assert(lb);
  assert(le);

  pe_info(lb->pe, "Initialising shear profile\n");

  /* Initialise the density, velocity, gradu; ghost modes are zero */

  physics_ref(&phys);
  physics_rho0(phys, &rho0);
  physics_eta_shear(phys, &eta);

  lees_edw_nlocal(le, nlocal);

  for (i = 0; i< NDIM; i++) {
    u[i] = 0.0;
    for (j = 0; j < NDIM; j++) {
      gradu[i][j] = 0.0;
    }
  }

  lees_edw_shear_rate(le, &gradu[X][Y]);

  /* Loop trough the sites */

  for (ic = 1; ic <= nlocal[X]; ic++) {

    lees_edw_steady_uy(le, ic, &u[Y]);

    /* We can now project the physical quantities to the distribution */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = lees_edw_index(le, ic, jc, kc);

	for (p = 0; p < NVEL; p++) {
	  double f = 0.0;
	  double cdotu = 0.0;
	  double sdotq = 0.0;

	  for (i = 0; i < NDIM; i++) {
	    cdotu += cv[p][i]*u[i];
	    for (j = 0; j < NDIM; j++) {
	      sdotq += (rho0*u[i]*u[j] - eta*gradu[i][j])*q_[p][i][j];
	    }
	  }
	  f = wv[p]*(rho0 + rcs2*rho0*cdotu + 0.5*rcs2*rcs2*sdotq);
	  lb_f_set(lb, index, p, 0, f);
	}
	/* Next site */
      }
    }
  }

  return 0;
}
