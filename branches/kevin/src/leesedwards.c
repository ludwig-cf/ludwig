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
 *  (c) 2010-2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "leesedwards.h"

enum shear_type {LE_LINEAR, LE_OSCILLATORY};

struct le_s {
  coords_t * coords;        /* Reference to coordinate system */
  int nref;                 /* Reference count */

  /* Global parameters */
  int    nplanes;           /* Total number of planes */
  int    type;              /* Shear type */
  int    period;            /* for oscillatory */
  int    nt0;               /* time0 (input as integer) */
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

static le_t * le_stat = NULL;

static int le_checks(le_t * le);
static int le_init_tables(le_t * le);

/*****************************************************************************
 *
 *  le_create
 *
 *****************************************************************************/

int le_create(coords_t * cs, le_t ** ple) {

  le_t * le = NULL;

  le = (le_t *) calloc(1, sizeof(le_t));
  if (le == NULL) fatal("calloc(le_t) failed\n");

  assert(cs);
  le->coords = cs;
  coords_retain(cs);

  le->nplanes = 0;
  le->type = LE_LINEAR;
  le->nref = 1;

  *ple = le;

  return 0;
}

/*****************************************************************************
 *
 *  le_free
 *
 *****************************************************************************/

int le_free(le_t ** ple) {

  le_t * le = NULL;

  assert(ple);
  le = *ple;

  coords_free(&le->coords);
  free(le->index_buffer_to_real);
  free(le->index_real_to_buffer);
  free(le->buffer_duy);
  free(le);

  *ple = NULL;

  return 0;
}

/*****************************************************************************
 *
 * le_nplane_set
 *
 *****************************************************************************/

int le_nplane_set(le_t * le, int nplanes) {

  assert(le);

  le->nplanes = nplanes;

  return 0;
}

/*****************************************************************************
 *
 * le_uy_set
 *
 *****************************************************************************/

int le_uy_set(le_t * le, double uy) {

  assert(le);

  le->uy = uy;

  return 0;
}

/*****************************************************************************
 *
 *  le_oscillatory_set
 *
 *****************************************************************************/

int le_oscillatory_set(le_t * le, int period) {

  assert(le);

  le->type = LE_OSCILLATORY;
  le->period = period;
  le->omega = 2.0*4.0*atan(1.0)/le->period;

  return 0;
} 

/*****************************************************************************
 *
 *  le_toffset_set
 *
 *****************************************************************************/

int le_toffset_set(le_t * le, int nt0) {

  assert(le);

  le->nt0 = nt0;

  return 0;
}

/*****************************************************************************
 *
 *  le_commit
 *
 *  We assume there are a given number of equally-spaced planes
 *  all with the same speed.
 *
 *  Pending (KS):
 *   - dx should be integers, i.e, ntotal[X] % ntotal must be zero
 *   - look at displacement issue
 *
 *****************************************************************************/

int le_commit(le_t * le) {

  int ntotal[3];

  assert(le);

  coords_ntotal(le->coords, ntotal);

  if (le->nplanes > 0) {

    if (ntotal[X] % le->nplanes) {
      info("System size x-direction: %d\n", ntotal[X]);
      info("Number of LE planes requested: %d\n", le->nplanes);
      fatal("Number of planes must divide system size\n");
    }

    le->dx_sep = 1.0*ntotal[X] / le->nplanes;
    le->dx_min = 0.5*le->dx_sep;
    le->time0 = 1.0*le->nt0;
  }

  le_stat = le;

  le_checks(le);
  le_init_tables(le);

  return 0;
}

/*****************************************************************************
 *
 *  le_nplane_total
 *
 *****************************************************************************/

int le_nplane_total(le_t * le, int * npt) {

  assert(le);

  *npt = le->nplanes;

  return 0;
}

/*****************************************************************************
 *
 *  le_nplane_local
 *
 *****************************************************************************/

int le_nplane_local(le_t * le, int * npl) {

  assert(le);

  *npl = le->nlocal;

  return 0;
}

/*****************************************************************************
 *
 *  le_uy
 *
 *****************************************************************************/

int le_uy(le_t * le, double * uy) {

  assert(le);

  *uy = le->uy;

  return 0;
}

/*****************************************************************************
 *
 *  le_nxbuffer
 *
 *****************************************************************************/

int le_nxbuffer(le_t * le, int * nxb) {

  assert(le);

  *nxb = le->nxbuffer;

  return 0;
}

/*****************************************************************************
 *
 *  le_info
 *
 *  A human-readable summary of Lees Edwards information.
 *
 *****************************************************************************/

int le_info(le_t * le) {

  int np;

  assert(le);

  if (le->nplanes > 0) {

    info("\nLees-Edwards boundary conditions are active:\n");

    for (np = 0; np < le->nplanes; np++) {
      info("LE plane %d is at x = %d with speed %f\n", np+1,
	   (int)(le->dx_min + np*le->dx_sep), le->uy);
    }

    if (le->type == LE_LINEAR) {
      info("Overall shear rate = %f\n", le_shear_rate());
    }
    else {
      info("Oscillation period: %d time steps\n", le->period);
      info("Maximum shear rate = %f\n", le_shear_rate());
    }

    info("\n");
    info("Lees-Edwards time offset (time steps): %8d\n", le->nt0);
  }

  return 0;
}

/*****************************************************************************
 *
 *  le_init_tables
 *
 *  Initialise the buffer look up tables.
 *
 *****************************************************************************/

static int le_init_tables(le_t * le) {

  int ib, ic, ip, n, nb, nh, np;
  int nhalo;
  int nlocal[3];
  int rdims[3];
  int cartsz[3];
  MPI_Comm cartcomm;

  assert(le);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_cartsz(le->coords, cartsz);
  coords_cart_comm(le->coords, &cartcomm);

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
    ic = le_plane_location(n) - (nhalo - 1);
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
     ip = le_plane_location(np);

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
 *  le_checks
 *
 *  We check that there are no planes within range of the processor or
 *  periodic halo regions.
 *
 ****************************************************************************/
 
static int le_checks(le_t * le) {

  int nhalo;
  int nlocal[3];
  int n;
  int ifail_local = 0;
  int ifail_global;
  int cartsz[3];
  MPI_Comm cartcomm;

  assert(le);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_cartsz(le->coords, cartsz);
  coords_cart_comm(le->coords, &cartcomm);

  /* From the local viewpoint, there must be no planes at either
   * x = 1 or x = nlocal[X] (or indeed, within nhalo points of
   * a processor or periodic boundary). */

  for (n = 0; n < le->nlocal; n++) {
    if (le_plane_location(n) <= nhalo) ifail_local = 1;
    if (le_plane_location(n) > nlocal[X] - nhalo) ifail_local = 1;
  }

  MPI_Allreduce(&ifail_local, &ifail_global, 1, MPI_INT, MPI_LOR, cartcomm);

  if (ifail_global) {
    fatal("LE_init(): wall at domain boundary\n");
  }

  /* As nplane_local = ntotal/cartsz[X] (integer division) we must have
   * ntotal % cartsz[X] = 0 */

  if ((le->nplanes % cartsz[X]) != 0) {
    info("\n");
    info("Must have a uniform number of planes per process in X direction\n");
    info("Eg., use one plane per process.\n");
    fatal("Please check and try again.\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  le_nsites
 *
 *  The equivalent of coords_nsites() adding the necessary buffer
 *  space required for LE quantities.
 *
 *****************************************************************************/

int le_nsites(void) {

  int nhalo;
  int nlocal[3];
  int nsites;

  assert(le_stat);
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nsites = (nlocal[X] + 2*nhalo + le_get_nxbuffer())*(nlocal[Y] + 2*nhalo)
    *(nlocal[Z] + 2*nhalo);

  return nsites;
}

/*****************************************************************************
 *
 *  le_get_steady_uy
 *
 *  Return the velocity expected for steady shear profile at
 *  position x (dependent on x-direction only). Takes a local index.
 *
 *  Should do something sensible for oscillatory shear.
 *
 *****************************************************************************/

double le_get_steady_uy(int ic) {

  int offset[3];
  int nplane;
  double xglobal, uy;

  assert(le_stat);
  assert(le_stat->type == LE_LINEAR);
  coords_nlocal_offset(offset);

  /* The shear profile is linear, so the local velocity is just a
   * function of position, modulo the number of planes encountered
   * since the origin. The planes are half way between sites giving
   * the - 0.5. */

  xglobal = offset[X] + (double) ic - 0.5;
  nplane = (int) ((le_stat->dx_min + xglobal)/le_stat->dx_sep);

  uy = xglobal*le_shear_rate() - le_plane_uy_max()*nplane;
 
  return uy;
}

/*****************************************************************************
 *
 *  le_get_block_uy
 *
 *  Return the velocity of the LE 'block' at ic relative to the
 *  centre of the system.
 *
 *  This is useful to output y velocities corrected for the planes.
 *  We always consider the central LE block to be stationary, i.e.,
 *  'unroll' from the centre.
 *
 *****************************************************************************/

double le_get_block_uy(int ic) {

  int offset[3];
  int n;
  double xh, uy;
  double lmin[3];
  double ltot[3];

  assert(le_stat);
  assert(le_stat->type == LE_LINEAR);

  coords_lmin(le_stat->coords, lmin);
  coords_ltot(le_stat->coords, ltot);
  coords_nlocal_offset(offset);

  /* So, just count the number of blocks from the centre L_x/2
   * and mutliply by the plane speed. */

  xh = offset[X] + (double) ic - lmin[X] - 0.5*ltot[X];
  if (xh > 0.0) {
    n = (0.5 + xh/le_stat->dx_sep);
  }
  else {
    n = (-0.5 + xh/le_stat->dx_sep);
  }
  uy = le_plane_uy_max()*n;

  return uy;
}

/*****************************************************************************
 *
 *  le_get_nplane_local
 *
 *  Return the local number of planes.
 *
 *****************************************************************************/

int le_get_nplane_local() {
  assert(le_stat);
  return le_stat->nlocal;
}

/*****************************************************************************
 *
 *  le_get_nplane_total
 *
 *  Return the total number of planes in the system.
 *
 *****************************************************************************/

int le_get_nplane_total() {
  assert(le_stat);
  return le_stat->nplanes;
}

/*****************************************************************************
 *
 *  le_plane_uy
 *
 *  Return the current plane velocity for time t.
 *
 *****************************************************************************/

double le_plane_uy(double t) {

  double uy;
  double tle;

  assert(le_stat);

  tle = t - le_stat->time0;
  assert(tle >= 0.0);

  uy = le_stat->uy;
  if (le_stat->type == LE_OSCILLATORY) uy *= cos(le_stat->omega*tle);

  return uy;
}

/*****************************************************************************
 *
 *  le_plane_uy_max
 *
 *  Return the maximum plane velocity.
 *
 *****************************************************************************/

double le_plane_uy_max() {
  assert(le_stat);
  return le_stat->uy;
}

/*****************************************************************************
 *
 *  le_plane_location
 *
 *  Return location (local x-coordinte - 0.5) of local plane n.
 *  It is erroneous to call this if no planes.
 *
 *****************************************************************************/

int le_plane_location(const int n) {

  int offset[3];
  int nplane_offset;
  int ix;
  int cartcoords[3];

  assert(le_stat);
  assert(n >= 0 && n < le_stat->nlocal);

  coords_cart_coords(le_stat->coords, cartcoords);
  coords_nlocal_offset(offset);
  nplane_offset = cartcoords[X]*le_stat->nlocal;

  ix = le_stat->dx_min + (n + nplane_offset)*le_stat->dx_sep - offset[X];

  return ix;
}

/*****************************************************************************
 *
 *  le_get_nxbuffer
 *
 *  Return the size (x-direction) of the buffer required to hold
 *  cross-boundary interpolated quantities.
 *
 *****************************************************************************/

int le_get_nxbuffer() {
  assert(le_stat);
  return le_stat->nxbuffer;
}

/*****************************************************************************
 *
 *  le_index_real_to_buffer
 *
 *  For x index and step size di, return the x index of the translated
 *  buffer.
 *
 *****************************************************************************/

int le_index_real_to_buffer(const int ic, const int di) {

  int ib;
  int nhalo;

  assert(le_stat);

  nhalo = coords_nhalo();
  assert(di >= -nhalo && di <= +nhalo);

  ib = (ic + nhalo - 1)*(2*nhalo + 1) + di + nhalo;

  assert(ib >= 0 && ib < le_stat->index_real_nbuffer);

  return le_stat->index_real_to_buffer[ib];
}

/*****************************************************************************
 *
 *  le_index_buffer_to_real
 *
 *  For x index in the buffer region, return the corresponding
 *  x index in the real system.
 *
 *****************************************************************************/

int le_index_buffer_to_real(int ib) {

  assert(le_stat);
  assert(ib >=0 && ib < le_get_nxbuffer());

  return le_stat->index_buffer_to_real[ib];
}

/*****************************************************************************
 *
 *  le_buffer_displacement
 *
 *  Return the current displacement
 *
 *    dy = u_y t                     in the linear case
 *    dy = (u_y/omega) sin(omega t)  in the oscillatory case
 *
 *  for the buffer planewith x location ib.
 *
 *****************************************************************************/

double le_buffer_displacement(int ib, double t) {

  double dy = 0.0;
  double tle;

  assert(le_stat);
  assert(ib >= 0 && ib < le_stat->nxbuffer);

  tle = t - le_stat->time0;
  assert(tle >= 0.0);

  if (le_stat->type == LE_LINEAR) dy = tle*le_stat->uy*le_stat->buffer_duy[ib];
  if (le_stat->type == LE_OSCILLATORY) {
    dy = le_stat->uy*sin(le_stat->omega*tle)/le_stat->omega;
  }

  return dy;
}

/*****************************************************************************
 *
 *  le_communicator
 *
 *  Return the handle to the Lees Edwards communicator.
 *
 *****************************************************************************/

MPI_Comm le_communicator(void) {

  assert(le_stat);
  return le_stat->le_comm;
}

int le_comm(le_t * le, MPI_Comm * comm) {

  assert(le);

  *comm = le->le_comm;

  return 0;
}

/*****************************************************************************
 *
 *  le_plane_comm
 *
 *****************************************************************************/

MPI_Comm le_plane_comm(void) {

  assert(le_stat);

  return le_stat->le_plane_comm;
}

int le_xplane_comm(le_t * le, MPI_Comm * comm) {

  assert(le);

  *comm = le->le_plane_comm;

  return 0;
}

/*****************************************************************************
 *
 *  le_jstart_to_ranks
 *
 *  For global period position j1, work out which ranks are to
 *  receive messages, and which are to send from the current
 *  process in order to grab translated information.
 *
 *****************************************************************************/

int le_jstart_to_ranks(const int j1, int send[3], int recv[3]) {

  assert(le_stat);
  le_jstart_to_mpi_ranks(le_stat, j1, send, recv);

  return 0;
}

int le_jstart_to_mpi_ranks(le_t * le, const int j1, int send[3], int recv[3]) {

  int nlocal[3];
  int cartcoords[3];
  int pe_carty1, pe_carty2, pe_carty3;

  assert(le);
  coords_cart_coords(le->coords, cartcoords);
  coords_nlocal(nlocal);

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
 *  le_shear_rate
 *
 *  Return the maximum steady shear rate.
 *
 *****************************************************************************/

double le_shear_rate() {

  double ltot[3];

  assert(le_stat);
  coords_ltot(le_stat->coords, ltot);

  return (le_plane_uy_max()*le_stat->nplanes/ltot[X]);
}

/*****************************************************************************
 *
 *  le_site_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *  This differs from get_site_index only in construction (not in result).
 *
 *  Where performance is important, prefer macro version via NDEBUG.
 *
 *****************************************************************************/

int le_site_index(const int ic, const int jc, const int kc) {

  int nhalo;
  int nlocal[3];
  int index;

  assert(le_stat);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  assert(ic >= 1-nhalo);
  assert(jc >= 1-nhalo);
  assert(kc >= 1-nhalo);
  assert(ic <= nlocal[X] + nhalo + le_stat->nxbuffer);
  assert(jc <= nlocal[Y] + nhalo);
  assert(kc <= nlocal[Z] + nhalo);

  index = (nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo)*(nhalo + ic - 1)
    +                           (nlocal[Z] + 2*nhalo)*(nhalo + jc - 1)
    +                                                  nhalo + kc - 1;

  return index;
}
