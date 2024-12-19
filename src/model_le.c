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
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
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
#include "coords.h"
#include "cs_limits.h"
#include "control.h"
#include "physics.h"
#include "model_le.h"
#include "timer.h"
#include "util.h"

static int le_reproject(lb_t * lb, lees_edw_t * le);
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

__host__ int lb_le_apply_boundary_conditions_old(lb_t * lb, lees_edw_t * le) {

  int mpi_cartsz[3];

  assert(lb);
  assert(le);

  lees_edw_cartsz(le, mpi_cartsz);

  if (lees_edw_nplane_local(le) > 0) {

    TIMER_start(TIMER_LE);

    /* Everything must be done on host at the moment (slowly) ... */
    /* ... and copy back at the end */
    lb_memcpy(lb, tdpMemcpyDeviceToHost);

    le_reproject(lb, le);

    if (mpi_cartsz[Y] > 1) {
      le_displace_and_interpolate_parallel(lb, le);
    }
    else {
      le_displace_and_interpolate(lb, le);
    }

    lb_memcpy(lb, tdpMemcpyHostToDevice);

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
  int    n, ndist;
  int8_t cx = 0;

  double rho, ds[3][3], udotc, sdotq;
  double g[3], du[3];
  double fnew;
  double t;
  physics_t * phys = NULL;

  assert(lb);
  assert(le);

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
	cx = +1;
      }
      else {
	/* Finally, deal with plane above LEBC */
	lees_edw_plane_uy_now(le, t, &du[Y]);
	ic = lees_edw_plane_location(le, plane) + 1;
	cx = -1;
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
	    for (int p = 1; p < lb->model.nvel; p++) {

	      double cs2 = lb->model.cs2;
	      double rcs2 = 1.0/cs2;
	      if (lb->model.cv[p][X] != cx) continue;

	      udotc = du[Y]*lb->model.cv[p][Y];
	      sdotq = 0.0;

	      for (ia = 0; ia < 3; ia++) {
		for (ib = 0; ib < 3; ib++) {
		  double dab = cs2*(ia == ib);
		  double q = (lb->model.cv[p][ia]*lb->model.cv[p][ib] - dab);
		  sdotq += ds[ia][ib]*q;
		}
	      }

	      /* Project all this back to the distribution. */

	      lb_f(lb, index, p, n, &fnew);
	      fnew += lb->model.wv[p]*(rho*udotc*rcs2 + 0.5*sdotq*rcs2*rcs2);
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

  /* Allocate a buffer large enough for all cvp[][X] = +1 */

  nprop = 0;
  for (int p = 1; p < lb->model.nvel; p++) {
    if (lb->model.cv[p][X] == +1) nprop += 1;
  }

  ndata = ndist*nprop*nlocal[Y]*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  assert(recv_buff);
  if (recv_buff == NULL) pe_fatal(lb->pe, "malloc(recv_buff) failed\n");

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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] != +1) continue;
	    recv_buff[ndata++] = (1.0 - fr)*
	      lb->f[LB_ADDR(lb->nsite,ndist,lb->model.nvel,index0,n, p)]
	      + fr*
	      lb->f[LB_ADDR(lb->nsite,ndist,lb->model.nvel,index1,n, p)];
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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] != +1) continue;
	    int la = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, p);
	    lb->f[la] = recv_buff[ndata++];
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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] == -1) {
	      int l0 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, p);
	      int l1 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index1, n, p);
	      recv_buff[ndata++] = (1.0 - fr)*lb->f[l0] + fr*lb->f[l1];
	    }
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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] == -1) {
	      int ijkp = LB_ADDR(lb->nsite, ndist, lb->model.nvel,index0,n,p);
	      lb->f[ijkp] = recv_buff[ndata++];
	    }
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


  nprop = 0;
  for (int p = 1; p < lb->model.nvel; p++) {
    if (lb->model.cv[p][X] == +1) nprop += 1;
  }

  ndata = ndist*nprop*nlocal[Y]*nlocal[Z];
  send_buff = (double *) malloc(ndata*sizeof(double));
  assert(send_buff);
  if (send_buff == NULL) pe_fatal(lb->pe, "malloc(send_buff) failed\n");

  ndata = ndist*nprop*(nlocal[Y] + 1)*nlocal[Z];
  recv_buff = (double *) malloc(ndata*sizeof(double));
  assert(recv_buff);
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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] != +1) continue;
	    int ijkp = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
	    send_buff[ndata++] = lb->f[ijkp];
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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] != +1) continue;
	    int ijk = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
	    lb->f[ijk] = (1.0-fr)*recv_buff[ind1++] + fr*recv_buff[ind2++];
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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] != -1) continue;
	    int ijkp = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
	    send_buff[ndata++] = lb->f[ijkp];
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
	  for (int p = 1; p < lb->model.nvel; p++) {
	    if (lb->model.cv[p][X] != -1) continue;
	    int ijk = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index, n, p);
	    lb->f[ijk] = (1.0-fr)*recv_buff[ind1++] + fr*recv_buff[ind2++];
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

  for (i = 0; i< lb->model.ndim; i++) {
    u[i] = 0.0;
    for (j = 0; j < lb->model.ndim; j++) {
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

	for (p = 0; p < lb->model.nvel; p++) {
	  double f = 0.0;
	  double cdotu = 0.0;
	  double sdotq = 0.0;
	  double cs2 = lb->model.cs2;
	  double rcs2 = 1.0/cs2;

	  for (i = 0; i < lb->model.ndim; i++) {
	    cdotu += lb->model.cv[p][i]*u[i];
	    for (j = 0; j < lb->model.ndim; j++) {
	      double dij = (i == j);
	      double qij = lb->model.cv[p][i]*lb->model.cv[p][j] - cs2*dij;
	      sdotq += (rho0*u[i]*u[j] - eta*gradu[i][j])*qij;
	    }
	  }
	  f = lb->model.wv[p]*(rho0 + rcs2*rho0*cdotu + 0.5*rcs2*rcs2*sdotq);
	  lb_f_set(lb, index, p, 0, f);
	}
	/* Next site */
      }
    }
  }

  return 0;
}





/* FIXME ADDITIONS */

/* Kernel helper structure intended to be passed by value to kernel */

typedef struct lek_s {
  int nlocal[3];        /* 12 */
  int nplane;           /* 16 */
  int nprop;            /* 20 */
  int ndist;            /* 24 */
  int nxdist;           /* 28 */
} le_kernel_helper_t;

/*****************************************************************************
 *
 *  my_ibuf(iside, nlocal[Y], nlocal[Z], nplane, ndist, nprop)
 *
 *****************************************************************************/

__host__ __device__ static int my_ibuf(const le_kernel_helper_t * s,
				       int jc,  int kc, int iplane, int iside,
				       int n, int p) {
  int ib = 0;

  assert(s);
  assert(1 <= jc && jc <= s->nlocal[Y]); /* FIXME recv has one more? */
  assert(1 <= kc && kc <= s->nlocal[Z]);
  assert(0 <= iplane && iplane < s->nplane);
  assert(0 <= iside  && iside <= 1);
  assert(0 <= n && n < s->ndist);
  assert(0 <= p && p < s->nprop);

  ib = p + s->nprop*(n + s->ndist*(iplane + s->nplane*
				   (kc - 1 + s->nlocal[Z]*(jc - 1))));

  //printf("p iplane, ib nxdisp %d %d (%d %d) %d %d\n", p, iplane, jc, kc, ib, s->nxdist);
  assert(0 <= ib && ib < s->nxdist); /* Number of crossing distributions */

  /* All same sides are together */
  ib = iside*s->nxdist + ib;

  return ib;
}


__global__ void lb_data_reproject_kernel(kernel_3d_t k3d,
					 le_kernel_helper_t lekh, lb_t * lb,
					 lees_edw_t * le, double t);
__global__ void lb_data_displace_and_interpolate(kernel_3d_t k3d,
						 le_kernel_helper_t lekh,
						 lb_t * lb,
						 lees_edw_t * le, double t);
__global__ void copy_back(kernel_3d_t k3d, lb_t * lb, lees_edw_t * le);

static le_kernel_helper_t le_kernel_helper(lb_t * lb, lees_edw_t * le);
static int le_displace_and_interpolate_parallel(lb_t *lb, lees_edw_t *le);


/*****************************************************************************
 *
 *  lb_data_displace_and_interpolate
 *
 *  Parallel kernel for interpolation stage.
 *
 *****************************************************************************/

__global__ void lb_data_displace_and_interpolate(kernel_3d_t k3d,
						 le_kernel_helper_t lekh,
						 lb_t * lb,
						 lees_edw_t * le, double t) {
  int kindex = 0;
  int ndist = 0;
  int nlocal[3] = {0};
  int nhalo = 0;
  double ltot[3] = {0};

  lb_ndist(lb, &ndist);
  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);
  lees_edw_ltot(le, ltot);

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int iplane = kernel_3d_ic(&k3d, kindex); /* encodes plane */
    int jc     = kernel_3d_jc(&k3d, kindex);
    int kc     = kernel_3d_kc(&k3d, kindex);

    if (jc <= nlocal[Y] && kc <= nlocal[Z] && iplane < lekh.nplane) {

      int ic = lees_edw_plane_location(le, iplane);
      double dy = 0.0;
      int jdy = 0;
      double fr = 0.0;
      int iside = 0;

      lees_edw_buffer_displacement(le, nhalo, t, &dy);
      dy = fmod(dy, ltot[Y]);
      jdy = floor(dy);
      fr = dy - jdy;

      int j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      int j2 = 1 + (j1 % nlocal[Y]);

      int index0 = lees_edw_index(le, ic, jc, kc);
      //int index1 = lees_edw_index(le, ic, j2, kc);

      for (int n = 0; n < ndist; n++) {
	int ip = 0; /* FIXME  avoid if by lookup table [iside][9] */
	for (int p = 1; p < lb->model.nvel; p++) {
	  if (lb->model.cv[p][X] == +1) {
	    //int l0 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, p);
	    //int l1 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index1, n, p);
	    /* FIXME .... */
	    //int ibuf = ((jc-1)*nlocal[Z] + (kc-1))*ndist*nprop + n*nprop + ip + 2 * plane *nxdist;
	    //lb->sbuff[ibuf] = (1.0 - fr) * lb->f[l0] + fr * lb->f[l1];
	    int ibuf0 = my_ibuf(&lekh, j1, kc, iplane, iside, n, ip);
	    int ibuf1 = my_ibuf(&lekh, j2, kc, iplane, iside, n, ip);
	    double f = (1.0 - fr)*lb->rbuff[ibuf0] + fr*lb->rbuff[ibuf1];
	    lb_f_set(lb, index0, p, n, f);
	    ip++;
	  }
	}
      }

      /* OTHER DIRECTION */
      iside = 1;
      ic = lees_edw_plane_location(le, iplane) + 1;
      lees_edw_buffer_displacement(le, nhalo, t, &dy);
      dy = fmod(-dy, ltot[Y]);
      jdy = floor(dy);
      fr = dy - jdy;

      j1 = 1 + (jc + jdy - 1 + 2*nlocal[Y]) % nlocal[Y];
      j2 = 1 + (j1 % nlocal[Y]);

      index0 = lees_edw_index(le, ic, jc, kc);
      //index1 = lees_edw_index(le, ic, j2, kc);

      for (int n = 0; n < ndist; n++) {
	int ip = 0; assert(ndist == 1); /* FIXME */
	for (int p = 1; p < lb->model.nvel; p++) {
	  if (lb->model.cv[p][X] == -1) {
	    //int l0 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, p);
	    //int l1 = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index1, n, p);
	    //int ibuf = ((jc-1)*nlocal[Z] + (kc-1))*ndist*nprop + n*nprop + ip + (2 * plane + 1) *nxdist;
	    //lb->sbuff[ibuf] = (1.0 - fr) * lb->f[l0] + fr * lb->f[l1];
	    int ibuf0 = my_ibuf(&lekh, j1, kc, iplane, iside, n, ip);
	    int ibuf1 = my_ibuf(&lekh, j2, kc, iplane, iside, n, ip);
	    double f = (1.0 - fr)*lb->rbuff[ibuf0] + fr*lb->rbuff[ibuf1];
	    lb_f_set(lb, index0, p, n, f);
	    ip++;
	  }
	}
      }
      /* next plane */
    }
  }

  return;
}

/*****************************************************************************
 *
 *  We must not overwrite any of the existing distributions until
 *  all the interpolations have been computed.
 *
 *****************************************************************************/

__global__ void copy_back(kernel_3d_t k3d, lb_t * lb, lees_edw_t * le) {

  int kindex = 0;

  int ndist = 0;
  int nlocal[3] = {0};
  int nhalo = 0;
  int nplane = 0;
  int nprop = 0;
  int nxdist = 0;

  lb_ndist(lb, &ndist);
  lees_edw_nlocal(le, nlocal);
  lees_edw_nhalo(le, &nhalo);
  nplane = lees_edw_nplane_local(le);

  nprop = 0;
  for (int p = 1; p < lb->model.nvel; p++) {
    if (lb->model.cv[p][X] == +1) nprop += 1;
  }
  nxdist = ndist*nprop*nlocal[Y]*nlocal[Z];


  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int plane = kernel_3d_ic(&k3d, kindex); /* Encode plane */
    int jc    = kernel_3d_jc(&k3d, kindex);
    int kc    = kernel_3d_kc(&k3d, kindex);

    if (jc <= nlocal[Y] && kc <= nlocal[Z] && plane < nplane) {
      int ic = lees_edw_plane_location(le, plane);
      int index0 = lees_edw_index(le, ic, jc, kc);

      for (int n = 0; n < ndist; n++) {
	int ip = 0;
	for (int p = 1; p < lb->model.nvel; p++) {
	  if (lb->model.cv[p][X] == +1) {
	    /* ibuf = ibuf(jc, kc, n, p, ...) */
	    int ibuf = ((jc-1)*nlocal[Z] + (kc-1))*ndist*nprop + n*nprop + ip + 2 * plane * nxdist;
	    int la = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, p);
	    lb->f[la] = lb->sbuff[ibuf];
	    ip++;
	  }
	}
      }

      /* Other direction */

      ic = lees_edw_plane_location(le, plane) + 1;
      index0 = lees_edw_index(le, ic, jc, kc);

      for (int n = 0; n < ndist; n++) {
	int ip = 0;
	for (int p = 1; p < lb->model.nvel; p++) {
	  if (lb->model.cv[p][X] == -1) {
	    int ibuf = ((jc-1)*nlocal[Z] + (kc-1))*ndist*nprop + n*nprop + ip + (2*plane + 1) * nxdist;
	    int la = LB_ADDR(lb->nsite, ndist, lb->model.nvel, index0, n, p);
	    lb->f[la] = lb->sbuff[ibuf];
	    ip++;
	  }
	}
      }
    }
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

__global__ void lb_data_reproject_kernel(kernel_3d_t k3d,
					 le_kernel_helper_t lekh, lb_t * lb,
					 lees_edw_t * le, double t) {
  assert(lb);
  assert(le);

  int kindex = 0;

  for_simt_parallel(kindex, k3d.kiterations, 1) {

    int ix = kernel_3d_ic(&k3d, kindex); /* encodes plane/side */
    int jc = kernel_3d_jc(&k3d, kindex);
    int kc = kernel_3d_kc(&k3d, kindex);

    if (jc <= lekh.nlocal[Y] && kc <= lekh.nlocal[Z] && ix < 2*lekh.nplane) {

      int iplane = ix / 2; assert(0 <= iplane && iplane < lekh.nplane);
      int iside  = ix % 2; assert(iside == 0 || iside == 1);
      int cx     = 1 - 2*iside; /* below going up, or above going down */

      /* buffer location, propagation direction, velocity jump ... */
      int ic = iside + lees_edw_plane_location(le, iplane);
      int index = lees_edw_index(le, ic, jc, kc);
      double du[3] = {0};

      lees_edw_plane_uy_now(le, t, &du[Y]);
      du[Y] = -1.0*cx*du[Y];

      for (int n = 0; n < lekh.ndist; n++) {

	int ip = 0;
	double rho = 0.0;
	double g[3] = {0};
	double ds[3][3] = {0};

	/* Compute 0th and 1st moments */
	lb_dist_enum_t ndn = (lb_dist_enum_t) n;
	/* EXPAND ? */
	lb_0th_moment(lb, index, ndn, &rho);
	lb_1st_moment(lb, index, ndn, g);

	for (int ia = 0; ia < 3; ia++) {
	  for (int ib = 0; ib < 3; ib++) {
	    ds[ia][ib] = (g[ia]*du[ib] + du[ia]*g[ib] + rho*du[ia]*du[ib]);
	  }
	}

	/* Now update the distribution */
	for (int p = 1; p < lb->model.nvel; p++) {

	  double cs2 = lb->model.cs2;
	  double rcs2 = 1.0/cs2;

	  double udotc = du[Y]*lb->model.cv[p][Y];
	  double sdotq = 0.0;

	  if (lb->model.cv[p][X] != cx) continue;

	  for (int ia = 0; ia < 3; ia++) {
	    for (int ib = 0; ib < 3; ib++) {
	      double dab = cs2 * (ia == ib);
	      double q = (lb->model.cv[p][ia] * lb->model.cv[p][ib] - dab);
	      sdotq += ds[ia][ib] * q;
	    }
	  }

	  /* Project all this back to the distribution. */
	  {
	    int ibuf = my_ibuf(&lekh, jc, kc, iplane, iside, n, ip);
	    double f = 0.0;
	    lb_f(lb, index, p, n, &f);
	    f += lb->model.wv[p] * (rho*udotc*rcs2 + 0.5*sdotq*rcs2*rcs2);
	    /* REPLACE lb_f_set(lb, index, p, n, f); BY */
	    lb->sbuff[ibuf] = f;
	    ++ip;
	  }
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  lb_le_apply_boundary_conditions
 *
 *  Driver for the parallel update.
 *
 *****************************************************************************/

__host__ int lb_le_apply_boundary_conditions(lb_t * lb, lees_edw_t * le) {

  assert(lb);
  assert(le);

  int mpi_cartsz[3] = {0};
  le_kernel_helper_t lekh = le_kernel_helper(lb, le);

  lees_edw_cartsz(le, mpi_cartsz);

  if (lekh.nplane > 0) {
    int ndevice = 0;
    lees_edw_t * le_target;
    double t = -1.0;

    TIMER_start(TIMER_LE);

    tdpAssert( tdpGetDeviceCount(&ndevice) );
    lees_edw_target(le, &le_target);

    /* Require the time t = time step */
    {
      physics_t * phys = NULL;
      physics_ref(&phys);
      t = 1.0*physics_control_timestep(phys);
    }

    /* First, set up the interpolation buffer with the relevant elements of
     * the distribution, i.e., side below the nplane propagating up, and
     * side above plane propagating down. ("Up" and "down" in x-direction) */

    {
      int  nx   = 2*lekh.nplane; /* Two x positions (sides) for each plane */
      dim3 nblk = {0};
      dim3 ntpb = {0};
      cs_limits_t lim = {0, nx - 1, 1, lekh.nlocal[Y], 1, lekh.nlocal[Z]};
      kernel_3d_t k3d = kernel_3d(lb->cs, lim);

      kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

      tdpLaunchKernel(lb_data_reproject_kernel, nblk, ntpb, 0, 0,
		      k3d, lekh, lb->target, le_target, t);
      tdpAssert( tdpPeekAtLastError() );
      tdpAssert( tdpStreamSynchronize(0) );
    }

    /* Swap the send and recv buffer */

    tdpAssert( tdpMemcpy(lb->target->rbuff, lb->target->sbuff,
			 2*lekh.nxdist*sizeof(double),
			 tdpMemcpyDeviceToDevice) );

    /* Displacement and interpolation to replace distributions */
    if (mpi_cartsz[Y] > 1) {
      assert(0); /* PENDING comms for send to recv buffer */
    }
    else {
      int  nx   = lekh.nplane;
      dim3 nblk = {0};
      dim3 ntpb = {0};
      cs_limits_t lim = {0, nx - 1, 1, lekh.nlocal[Y], 1, lekh.nlocal[Z]};
      kernel_3d_t k3d = kernel_3d(lb->cs, lim);

      kernel_3d_launch_param(k3d.kiterations, &nblk, &ntpb);

      /* Two kernels to provide synchronisation: interpolate to temporary
       * and then can copy back values to the distribution itself. */

      tdpLaunchKernel(lb_data_displace_and_interpolate, nblk, ntpb, 0, 0,
		      k3d, lekh, lb->target, le_target, t);

      tdpAssert( tdpPeekAtLastError() );
      tdpAssert( tdpStreamSynchronize(0) );
    }

    TIMER_stop(TIMER_LE);
  }

  return 0;
}

/*****************************************************************************
 *
 *  le_kernel_helper
 *
 *****************************************************************************/

static le_kernel_helper_t le_kernel_helper(lb_t * lb, lees_edw_t * le) {

  le_kernel_helper_t lekh = {0};

  assert(le);

  lees_edw_nlocal(le, lekh.nlocal);
  lekh.nplane = lees_edw_nplane_local(le);

  for (int p = 1; p < lb->model.nvel; p++) {
    if (lb->model.cv[p][X] == +1) lekh.nprop += 1;
  }

  /* FIXME: nxdist to include the number of planes here but ncrossdist
   * does not include the number of planes */
  lekh.ndist  = lb->ndist;
  lekh.nxdist = lekh.ndist*lekh.nprop*lekh.nplane*lekh.nlocal[Y]*lekh.nlocal[Z];

  return lekh;
}
