/*****************************************************************************
 *
 *  leesedwards.c
 *
 *  This sits on top of coords.c and provides a way to deal with
 *  the coordinate transformations required by the Lees Edwards
 *  sliding periodic boundaries.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "leesedwards.h"

enum shear_type {LEES_EDW_LINEAR, LEES_EDW_OSCILLATORY};

struct lees_edw_s {
  pe_t * pe;                /* Parallel environment */
  cs_t * cs;                /* Coordinate system */
  physics_t * phys;         /* Constants */
  int nref;                 /* Reference count */

  /* Global parameters */
  int    nplanes;           /* Total number of planes */
  int    type;              /* Shear type */
  int    period;            /* for oscillatory */
  int    nt0;               /* time0 (input as integer) */
  int nsites;               /* Number of sites incl buffer planes */
  double uy;                /* u[Y] for all planes */
  double dx_min;            /* Position first plane */
  double dx_sep;            /* Plane separation */
  double omega;             /* u_y = u_le cos (omega t) for oscillatory */  
  double time0;             /* time offset */

  /* Local parameters */
  int nlocal;               /* Number of planes local domain */
  int nxbuffer;             /* Size of buffer region in x */
  int index_real_nbuffer;
  int * index_buffer_to_real;
  int * index_real_to_buffer;
  int * buffer_duy;

  MPI_Comm  le_comm;        /* 1-d communicator */
  MPI_Comm  le_plane_comm;  /* 2-d communicator */
};

static int lees_edw_init(lees_edw_t * le, lees_edw_info_t * info);
static int lees_edw_checks(lees_edw_t * le);
static int lees_edw_init_tables(lees_edw_t * le);

/*****************************************************************************
 *
 *  lees_edw_create
 *
 *****************************************************************************/

int lees_edw_create(pe_t * pe, cs_t * cs, lees_edw_info_t * info,
		    lees_edw_t ** ple) {

  lees_edw_t * le = NULL;

  assert(pe);
  assert(cs);

  le = (lees_edw_t *) calloc(1, sizeof(lees_edw_t));
  if (le == NULL) fatal("calloc(lees_edw_t) failed\n");

  assert(pe);
  assert(cs);
  le->pe = pe;
  pe_retain(pe);
  le->cs = cs;
  cs_retain(cs);

  le->nplanes = 0;
  if (info) lees_edw_init(le, info);
  lees_edw_init_tables(le);
  le->nref = 1;

  *ple = le;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_retain
 *
 *****************************************************************************/

int lees_edw_retain(lees_edw_t * le) {

  assert(le);

  le->nref += 1;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_free
 *
 *****************************************************************************/

int lees_edw_free(lees_edw_t * le) {

  assert(le);

  le->nref -= 1;

  if (le->nref <= 0) {
    pe_free(le->pe);
    cs_free(le->cs);
    free(le->index_buffer_to_real);
    free(le->index_real_to_buffer);
    free(le->buffer_duy);
    free(le);
  }

  return 0;
}

/*****************************************************************************
 *
 * lees_edw_nplane_set
 *
 *****************************************************************************/

int lees_edw_nplane_set(lees_edw_t * le, int nplanes) {

  assert(le);

  le->nplanes = nplanes;

  return 0;
}

/*****************************************************************************
 *
 * lees_edw_plane_uy_set
 *
 *****************************************************************************/

int lees_edw_plane_uy_set(lees_edw_t * le, double uy) {

  assert(le);

  le->uy = uy;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_oscillatory_set
 *
 *****************************************************************************/

int lees_edw_oscillatory_set(lees_edw_t * le, int period) {

  assert(le);

  le->type = LEES_EDW_OSCILLATORY;
  le->period = period;
  le->omega = 2.0*4.0*atan(1.0)/le->period;

  return 0;
} 

/*****************************************************************************
 *
 *  lees_edw_toffset_set
 *
 *****************************************************************************/

int lees_edw_toffset_set(lees_edw_t * le, int nt0) {

  assert(le);

  le->nt0 = nt0;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_init
 *
 *  We assume there are a given number of equally-spaced planes
 *  all with the same speed.
 *
 *  Pending (KS):
 *   - dx should be integers, i.e, ntotal[X] % ntotal must be zero
 *   - look at displacement issue
 *
 *****************************************************************************/

static int lees_edw_init(lees_edw_t * le, lees_edw_info_t * info) {

  int ntotal[3];

  assert(le);
  assert(info);

  le->nplanes = info->nplanes;
  le->uy = info->uy;
  le->type = info->type;
  le->period = info->period;
  le->nt0 = info->nt0;

  cs_ntotal(le->cs, ntotal);
  physics_ref(&le->phys);

  if (le->nplanes > 0) {

    if (ntotal[X] % le->nplanes) {
      pe_info(le->pe, "System size x-direction: %d\n", ntotal[X]);
      pe_info(le->pe, "Number of LE planes requested: %d\n", le->nplanes);
      pe_fatal(le->pe, "Number of planes must divide system size\n");
    }

    le->dx_sep = 1.0*ntotal[X] / le->nplanes;
    le->dx_min = 0.5*le->dx_sep;
    le->time0 = 1.0*le->nt0;
  }

  lees_edw_checks(le);

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_nplane_total
 *
 *****************************************************************************/

int lees_edw_nplane_total(lees_edw_t * le) {

  assert(le);

  return le->nplanes;
}

/*****************************************************************************
 *
 *  lees_edw_nplane_local
 *
 *****************************************************************************/

int lees_edw_nplane_local(lees_edw_t * le) {

  assert(le);

  return le->nlocal;
}

/*****************************************************************************
 *
 *  lees_edw_plane_uy
 *
 *****************************************************************************/

int lees_edw_plane_uy(lees_edw_t * le, double * uy) {

  assert(le);

  *uy = le->uy;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_nxbuffer
 *
 *****************************************************************************/

int lees_edw_nxbuffer(lees_edw_t * le, int * nxb) {

  assert(le);

  *nxb = le->nxbuffer;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_info
 *
 *  A human-readable summary of Lees Edwards information.
 *
 *****************************************************************************/

int lees_edw_info(lees_edw_t * le) {

  int np;
  double gammadot;

  assert(le);

  if (le->nplanes > 0) {

    pe_info(le->pe, "\nLees-Edwards boundary conditions are active:\n");

    lees_edw_shear_rate(le, &gammadot);

    for (np = 0; np < le->nplanes; np++) {
      pe_info(le->pe, "LE plane %d is at x = %d with speed %f\n", np+1,
	   (int)(le->dx_min + np*le->dx_sep), le->uy);
    }

    if (le->type == LEES_EDW_LINEAR) {
      pe_info(le->pe, "Overall shear rate = %f\n", gammadot);
    }
    else {
      pe_info(le->pe, "Oscillation period: %d time steps\n", le->period);
      pe_info(le->pe, "Maximum shear rate = %f\n", gammadot);
    }

    pe_info(le->pe, "\n");
    pe_info(le->pe, "Lees-Edwards time offset (time steps): %8d\n", le->nt0);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_init_tables
 *
 *  Initialise the buffer look up tables.
 *
 *****************************************************************************/

static int lees_edw_init_tables(lees_edw_t * le) {

  int ib, ic, ip, n, nb, nh, np;
  int nhalo;
  int nlocal[3];
  int rdims[3];
  int cartsz[3];
  MPI_Comm cartcomm;

  assert(le);

  cs_nhalo(le->cs, &nhalo);
  cs_nlocal(le->cs, nlocal);
  cs_cartsz(le->cs, cartsz);
  cs_cart_comm(le->cs, &cartcomm);

  le->nlocal = le->nplanes / cartsz[X];

  /* Look up table for buffer -> real index */

  /* For each 'x' location in the buffer region, work out the corresponding
   * x index in the real system:
   *   - for each boundary there are 2*nhalo buffer planes
   *   - the locations extend nhalo points either side of the boundary.
   */

  le->nxbuffer = 2*nhalo*le->nlocal;

  if (le->nxbuffer > 0) {
    le->index_buffer_to_real = (int *) calloc(le->nxbuffer, sizeof(int));
    if (le->index_buffer_to_real == NULL) fatal("calloc(le) failed\n");
  }

  ib = 0;
  for (n = 0; n < le->nlocal; n++) {
    ic = lees_edw_plane_location(le, n) - (nhalo - 1);
    for (nh = 0; nh < 2*nhalo; nh++) {
      assert(ib < 2*nhalo*le->nlocal);
      le->index_buffer_to_real[ib] = ic + nh;
      ib++;
    }
  }

  /* Look up table for real -> buffer index */

  /* For each x location in the real system, work out the index of
   * the appropriate x-location in the buffer region. This is more
   * complex, as it depends on whether you are looking across an
   * LE boundary, and if so, in which direction.
   * ie., we need a look up table = function(x, +/- dx).
   * Note that this table exists when no planes are present, ie.,
   * there is no transformation, ie., f(x, dx) = x + dx for all dx.
   */

  n = (nlocal[X] + 2*nhalo)*(2*nhalo + 1);
  le->index_real_nbuffer = n;

  le->index_real_to_buffer = (int *) calloc(n, sizeof(int));
  if (le->index_real_to_buffer == NULL) fatal("calloc(le) failed\n");

  /* Set table in abscence of planes. */
  /* Note the elements of the table at the extreme edges of the local
   * system point outside the system. Accesses must take care. */

   for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
     for (nh = -nhalo; nh <= nhalo; nh++) {
       n = (ic + nhalo - 1)*(2*nhalo+1) + (nh + nhalo);
       assert(n >= 0 && n < (nlocal[X] + 2*nhalo)*(2*nhalo + 1));
       le->index_real_to_buffer[n] = ic + nh;
     }
   }

   /* For each position in the buffer, add appropriate
    * corrections in the table. */

   nb = nlocal[X] + nhalo + 1;

   for (ib = 0; ib < le->nxbuffer; ib++) {
     np = ib / (2*nhalo);
     ip = lees_edw_plane_location(le, np);

     /* This bit of logic chooses the first nhalo points of the
      * buffer region for each plane as the 'downward' looking part */

     if ((ib - np*2*nhalo) < nhalo) {

       /* Looking across the plane in the -ve x-direction */

       for (ic = ip + 1; ic <= ip + nhalo; ic++) {
	 for (nh = -nhalo; nh <= -1; nh++) {
	   if (ic + nh == le->index_buffer_to_real[ib]) {
	     n = (ic + nhalo - 1)*(2*nhalo+1) + (nh + nhalo);
	     assert(n >= 0 && n < (nlocal[X] + 2*nhalo)*(2*nhalo + 1));
	     le->index_real_to_buffer[n] = nb+ib;
	   }
	 }
       }
     }
     else {
       /* looking across the plane in the +ve x-direction */

       for (ic = ip - (nhalo - 1); ic <= ip; ic++) {
	 for (nh = 1; nh <= nhalo; nh++) {
	   if (ic + nh == le->index_buffer_to_real[ib]) {
	     n = (ic + nhalo - 1)*(2*nhalo+1) + (nh + nhalo);
	     assert(n >= 0 && n < (nlocal[X] + 2*nhalo)*(2*nhalo + 1));
	     le->index_real_to_buffer[n] = nb+ib;	   
	   }
	 }
       }
     }
     /* Next buffer point */
   }

   /* Buffer velocity jumps. When looking from the real system across
    * a boundary into a given buffer, what is the associated velocity
    * jump? This is +1 for 'looking up' and -1 for 'looking down'.*/

   if (le->nxbuffer > 0) {
     le->buffer_duy = (int *) calloc(le->nxbuffer, sizeof(double));
     if (le->buffer_duy == NULL) fatal("calloc(buffer_duy) failed\n");
   }

  ib = 0;
  for (n = 0; n < le->nlocal; n++) {
    for (nh = 0; nh < nhalo; nh++) {
      assert(ib < le->nxbuffer);
      le->buffer_duy[ib] = -1;
      ib++;
    }
    for (nh = 0; nh < nhalo; nh++) {
      assert(ib < le->nxbuffer);
      le->buffer_duy[ib] = +1;
      ib++;
    }
  }

  /* Set up a 1-dimensional communicator for transfer of data
   * along the y-direction. */

  rdims[X] = 0;
  rdims[Y] = 1;
  rdims[Z] = 0;
  MPI_Cart_sub(cartcomm, rdims, &le->le_comm);

  /* Plane communicator in yz, or x = const. */

  rdims[X] = 0;
  rdims[Y] = 1;
  rdims[Z] = 1;

  MPI_Cart_sub(cartcomm, rdims, &le->le_plane_comm);

  return 0;
}

/****************************************************************************
 *
 *  lees_edw_checks
 *
 *  We check that there are no planes within range of the processor or
 *  periodic halo regions.
 *
 ****************************************************************************/
 
static int lees_edw_checks(lees_edw_t * le) {

  int nhalo;
  int nlocal[3];
  int n;
  int ifail_local = 0;
  int ifail_global;
  int cartsz[3];
  MPI_Comm cartcomm;

  assert(le);

  cs_nhalo(le->cs, &nhalo);
  cs_nlocal(le->cs, nlocal);
  cs_cartsz(le->cs, cartsz);
  cs_cart_comm(le->cs, &cartcomm);

  /* From the local viewpoint, there must be no planes at either
   * x = 1 or x = nlocal[X] (or indeed, within nhalo points of
   * a processor or periodic boundary). */

  for (n = 0; n < le->nlocal; n++) {
    if (lees_edw_plane_location(le, n) <= nhalo) ifail_local = 1;
    if (lees_edw_plane_location(le, n) > nlocal[X] - nhalo) ifail_local = 1;
  }

  MPI_Allreduce(&ifail_local, &ifail_global, 1, MPI_INT, MPI_LOR, cartcomm);

  if (ifail_global) {
    pe_fatal(le->pe, "Wall at domain boundary\n");
  }

  /* As nplane_local = ntotal/cartsz[X] (integer division) we must have
   * ntotal % cartsz[X] = 0 */

  if ((le->nplanes % cartsz[X]) != 0) {
    pe_info(le->pe, "\n");
    pe_info(le->pe, "Must have a uniform number of planes per process\n");
    pe_info(le->pe, "Eg., use one plane per process.\n");
    pe_fatal(le->pe, "Please check and try again.\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_nsites
 *
 *  The equivalent of cs_nsites() adding the necessary buffer
 *  space required for LE quantities.
 *
 *****************************************************************************/

int lees_edw_nsites(lees_edw_t * le, int * nsites) {

  int nhalo;
  int nlocal[3];

  assert(le);
  cs_nhalo(le->cs, &nhalo);
  cs_nlocal(le->cs, nlocal);

  *nsites = (nlocal[X] + 2*nhalo + le->nxbuffer)
    *(nlocal[Y] + 2*nhalo)
    *(nlocal[Z] + 2*nhalo);

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_steady_uy
 *
 *  Return the velocity expected for steady shear profile at
 *  position x (dependent on x-direction only). Takes a local index.
 *
 *  Should do something sensible for oscillatory shear.
 *
 *****************************************************************************/

int lees_edw_steady_uy(lees_edw_t * le, int ic, double * uy) {

  int offset[3];
  int nplane;
  double gammadot;
  double xglobal;

  assert(le);
  assert(le->type == LEES_EDW_LINEAR);

  cs_nlocal_offset(le->cs, offset);
  lees_edw_shear_rate(le, &gammadot);

  /* The shear profile is linear, so the local velocity is just a
   * function of position, modulo the number of planes encountered
   * since the origin. The planes are half way between sites giving
   * the - 0.5. */

  xglobal = offset[X] + (double) ic - 0.5;
  nplane = (int) ((le->dx_min + xglobal)/le->dx_sep);

  *uy = xglobal*gammadot - le->uy*nplane;
 
  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_get_block_uy
 *
 *  Return the velocity of the LE 'block' at ic relative to the
 *  centre of the system.
 *
 *  This is useful to output y velocities corrected for the planes.
 *  We always consider the central LE block to be stationary, i.e.,
 *  'unroll' from the centre.
 *
 *****************************************************************************/

int lees_edw_block_uy(lees_edw_t * le, int ic, double * uy) {

  int offset[3];
  int n;
  double xh;
  double lmin[3];
  double ltot[3];

  assert(le);
  assert(le->type == LEES_EDW_LINEAR);

  cs_lmin(le->cs, lmin);
  cs_ltot(le->cs, ltot);
  cs_nlocal_offset(le->cs, offset);

  /* So, just count the number of blocks from the centre L_x/2
   * and mutliply by the plane speed. */

  xh = offset[X] + (double) ic - lmin[X] - 0.5*ltot[X];
  if (xh > 0.0) {
    n = (0.5 + xh/le->dx_sep);
  }
  else {
    n = (-0.5 + xh/le->dx_sep);
  }

  *uy = le->uy*n;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_plane_uy_now
 *
 *  Return the current plane velocity for time t.
 *
 *****************************************************************************/

int lees_edw_plane_uy_now(lees_edw_t * le, double t, double * uy) {

  double tle;

  assert(le);

  tle = t - le->time0;
  assert(tle >= 0.0);

  *uy = le->uy;
  if (le->type == LEES_EDW_OSCILLATORY) *uy *= cos(le->omega*tle);

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_plane_location
 *
 *  Return location (local x-coordinte - 0.5) of local plane np.
 *  It is erroneous to call this if no planes.
 *
 *****************************************************************************/

int lees_edw_plane_location(lees_edw_t * le, int np) {

  int offset[3];
  int nplane_offset;
  int cartcoords[3];
  int ix;

  assert(le);
  assert(np >= 0 && np < le->nlocal);

  cs_cart_coords(le->cs, cartcoords);
  cs_nlocal_offset(le->cs, offset);
  nplane_offset = cartcoords[X]*le->nlocal;

  ix = le->dx_min + (np + nplane_offset)*le->dx_sep - offset[X];

  return ix;
}

/*****************************************************************************
 *
 *  lees_edw_index_real_to_buffer
 *
 *  For x index 'ic' and step size 'idisplace', return the x index of the
 *  translated buffer.
 *
 *****************************************************************************/

int lees_edw_index_real_to_buffer(lees_edw_t * le,  int ic,  int idisplace) {

  int ib;
  int nhalo;

  assert(le);
  assert(le->index_real_to_buffer);

  cs_nhalo(le->cs, &nhalo);
  assert(idisplace >= -nhalo && idisplace <= +nhalo);

  ib = (ic + nhalo - 1)*(2*nhalo + 1) + idisplace + nhalo;

  assert(ib >= 0 && ib < le->index_real_nbuffer);

  return le->index_real_to_buffer[ib];
}

/*****************************************************************************
 *
 *  lees_edw_index_buffer_to_real
 *
 *  For x index in the buffer region, return the corresponding
 *  x index in the real system.
 *
 *****************************************************************************/

int lees_edw_index_buffer_to_real(lees_edw_t * le, int ib) {

  assert(le);
  assert(le->index_buffer_to_real);
  assert(ib >=0 && ib < le->nxbuffer);

  return le->index_buffer_to_real[ib];
}

/*****************************************************************************
 *
 *  lees_edw_buffer_displacement
 *
 *  Return the current displacement
 *
 *    dy = u_y t                     in the linear case
 *    dy = (u_y/omega) sin(omega t)  in the oscillatory case
 *
 *  for the buffer planewith x location ib.
 *
 *****************************************************************************/

int lees_edw_buffer_displacement(lees_edw_t * le, int ib, double t, double * dy) {

  double tle;

  assert(le);
  assert(ib >= 0 && ib < le->nxbuffer);

  tle = t - le->time0;
  assert(tle >= 0.0);

  *dy = 0.0;

  if (le->type == LEES_EDW_LINEAR) *dy = tle*le->uy*le->buffer_duy[ib];
  if (le->type == LEES_EDW_OSCILLATORY) {
    *dy = le->uy*sin(le->omega*tle)/le->omega;
  }

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_communicator
 *
 *  Return the handle to the Lees Edwards communicator.
 *
 *****************************************************************************/

int lees_edw_comm(lees_edw_t * le, MPI_Comm * comm) {

  assert(le);

  *comm = le->le_comm;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_plane_comm
 *
 *****************************************************************************/

int lees_edw_plane_comm(lees_edw_t * le, MPI_Comm * comm) {

  assert(le);

  *comm = le->le_plane_comm;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_jstart_to_mpi_ranks
 *
 *  For global period position j1, work out which ranks are to
 *  receive messages, and which are to send from the current
 *  process in order to grab translated information.
 *
 *****************************************************************************/

int lees_edw_jstart_to_mpi_ranks(lees_edw_t * le, const int j1, int send[3],
				 int recv[3]) {

  int nlocal[3];
  int cartcoords[3];
  int pe_carty1, pe_carty2, pe_carty3;

  assert(le);

  cs_cart_coords(le->cs, cartcoords);
  cs_nlocal(le->cs, nlocal);

  /* Receive from ... */

  pe_carty1 = (j1 - 1) / nlocal[Y];
  pe_carty2 = pe_carty1 + 1;
  pe_carty3 = pe_carty1 + 2;

  MPI_Cart_rank(le->le_comm, &pe_carty1, recv);
  MPI_Cart_rank(le->le_comm, &pe_carty2, recv + 1);
  MPI_Cart_rank(le->le_comm, &pe_carty3, recv + 2);

  /* Send to ... */

  pe_carty1 = cartcoords[Y] - (((j1 - 1)/nlocal[Y]) - cartcoords[Y]);
  pe_carty2 = pe_carty1 - 1;
  pe_carty3 = pe_carty1 - 2;

  MPI_Cart_rank(le->le_comm, &pe_carty1, send);
  MPI_Cart_rank(le->le_comm, &pe_carty2, send + 1);
  MPI_Cart_rank(le->le_comm, &pe_carty3, send + 2);

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_shear_rate
 *
 *  Return the maximum steady shear rate.
 *
 *****************************************************************************/

int lees_edw_shear_rate(lees_edw_t * le, double * gammadot) {

  double ltot[3];

  assert(le);
  cs_ltot(le->cs, ltot);

  *gammadot = le->uy*le->nplanes/ltot[X];

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *  This differs from coords_index() in that it allows the buffer
 *  region to be used.
 *
 *****************************************************************************/

int lees_edw_index(lees_edw_t * le, int ic, int jc, int kc) {

  int nhalo;
  int nlocal[3];
  int index;

  assert(le);

  cs_nhalo(le->cs, &nhalo);
  cs_nlocal(le->cs, nlocal);

  assert(ic >= 1-nhalo);
  assert(jc >= 1-nhalo);
  assert(kc >= 1-nhalo);
  assert(ic <= nlocal[X] + nhalo + le->nxbuffer);
  assert(jc <= nlocal[Y] + nhalo);
  assert(kc <= nlocal[Z] + nhalo);

  index = (nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo)*(nhalo + ic - 1)
    +                           (nlocal[Z] + 2*nhalo)*(nhalo + jc - 1)
    +                                                  nhalo + kc - 1;

  return index;
}

/*****************************************************************************
 *
 *  lees_edw_nlocal
 *
 *****************************************************************************/

int lees_edw_nlocal(lees_edw_t * le, int nlocal[3]) {

  assert(le);

  return cs_nlocal(le->cs, nlocal);
}

/*****************************************************************************
 *
 *  lees_edw_strides
 *
 *****************************************************************************/

int lees_edw_strides(lees_edw_t * le, int * xs, int * ys, int * zs) {

  assert(le);

  return cs_strides(le->cs, xs, ys, zs);
}

/*****************************************************************************
 *
 *  lees_edw_nhalo
 *
 *****************************************************************************/

int lees_edw_nhalo(lees_edw_t * le, int * nhalo) {

  assert(le);

  return cs_nhalo(le->cs, nhalo);
}

/*****************************************************************************
 *
 *  lees_edw_ltot
 *
 *****************************************************************************/

int lees_edw_ltot(lees_edw_t * le, double ltot[3]) {

  assert(le);

  return cs_ltot(le->cs, ltot);
}

/*****************************************************************************
 *
 *  lees_edw_cartsz
 *
 *****************************************************************************/

int lees_edw_cartsz(lees_edw_t * le, int cartsz[3]) {

  assert(le);

  return cs_cartsz(le->cs, cartsz);
}

/*****************************************************************************
 *
 *  lees_edw_ntotal
 *
 *****************************************************************************/

int lees_edw_ntotal(lees_edw_t * le, int ntotal[3]) {

  assert(le);

  return cs_ntotal(le->cs, ntotal);
}

/*****************************************************************************
 *
 *  lees_edw_nlocal_offset
 *
 *****************************************************************************/

int lees_edw_nlocal_offset(lees_edw_t * le, int noffset[3]) {

  assert(le);

  return cs_nlocal_offset(le->cs, noffset);
}

/*****************************************************************************
 *
 *  lees_edw_cart_coords
 *
 *****************************************************************************/

int lees_edw_cart_coords(lees_edw_t * le, int cartcoord[3]) {

  assert(le);

  return cs_cart_coords(le->cs, cartcoord);
}

/****************************************************************************
 *
 *  lees_edw_plane_dy
 *
 *  Based on time = t-1
 *
 ****************************************************************************/

__host__ int lees_edw_plane_dy(lees_edw_t * le, double * dy) {

  double t;

  assert(le);
  assert(le->phys);
  assert(dy);

  physics_control_time(le->phys, &t);
  *dy = t*le->uy;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_buffer_dy
 *
 *  Displacement related to buffer point ib.
 *  t0 is offset = -1.0 when called from field.c for regression purposes.
 *
 *****************************************************************************/

__host__ int lees_edw_buffer_dy(lees_edw_t * le, int ib, double t0,
				double * dy) {

  double t;

  assert(le);
  assert(le->phys);

  physics_control_time(le->phys, &t);
  lees_edw_buffer_displacement(le, ib, t+t0, dy);

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_buffer_du
 *
 *  Velocity jump associated with buffer point ib (with sign).
 *
 *****************************************************************************/

__host__ int lees_edw_buffer_du(lees_edw_t * le, int ib, double ule[3]) {

  assert(le);

  if (le->type == LEES_EDW_LINEAR) {
    ule[X] = 0.0;
    ule[Y] = le->uy*le->buffer_duy[ib];
    ule[Z] = 0.0;
  }
  else {
    assert(0); /* Check delta u as function of (ib,t) */
  }

  return 0;
}
