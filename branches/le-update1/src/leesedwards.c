/*****************************************************************************
 *
 *  leesedwards.c
 *
 *  This sits on top of coords.c and provides a way to deal with
 *  the coordinate transformations required by the Lees Edwards
 *  sliding periodic boundaries.
 *
 *  $Id: leesedwards.c,v 1.12.4.5 2009-03-25 18:11:35 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "control.h"
#include "leesedwards.h"

enum shear_type {LINEAR, OSCILLATORY};

static void le_checks(void);
static void le_init_tables(void);

static struct le_parameters {
  /* Global parameters */
  double uy_plane;          /* u[Y] for all planes */
  double dx_min;            /* Position first plane */
  double dx_sep;            /* Plane separation */
  double omega;             /* u_y = u_le cos (omega t) for oscillatory */  

  /* Local parameters */
  MPI_Comm  le_comm;
  int *     index_buffer_to_real;
  int       index_real_nbuffer;
  int *     index_real_to_buffer;
  int *     buffer_duy;
} le_params_;

static int nplane_total_ = 0;     /* Total number of planes */
static int le_type_ = LINEAR;
static int initialised_ = 0;

/*****************************************************************************
 *
 *  le_init
 *
 *  We assume there are a given number of equally-spaced planes
 *  all with the same speed.
 *
 *  Pending (KS):
 *   - dx should be integers, i.e, N_total(X) % ntotal must be zero
 *   - look at displacement issue
 *
 *****************************************************************************/

void le_init() {

  int n;
  int ntotal;

  n = RUN_get_int_parameter("N_LE_plane", &ntotal);
  if (n != 0) nplane_total_ = ntotal;

  n = RUN_get_double_parameter("LE_plane_vel", &le_params_.uy_plane);
  initialised_ = 1;

  if (le_get_nplane_total() != 0) {

    info("\nLees-Edwards boundary conditions are active:\n");

    if (N_total(X) % ntotal) {
      info("System size x-direction: %d\n", N_total(X));
      info("Number of LE planes requested: %d\n", ntotal);
      fatal("Number of planes must divide system size\n");
    }

    le_params_.dx_sep = L(X) / ntotal;
    le_params_.dx_min = 0.5*le_params_.dx_sep;

    for (n = 0; n < ntotal; n++) {
      info("LE plane %d is at x = %d with speed %f\n", n+1,
	   (int)(le_params_.dx_min + n*le_params_.dx_sep),
	   le_get_plane_uy());
    }

    info("Overall shear rate = %f\n", le_shear_rate());
  }

  le_checks();
  le_init_tables();

  return;
}

/*****************************************************************************
 *
 *  le_finish
 *
 *****************************************************************************/

void le_finish() {

  if (initialised_ == 0) fatal("Calling le_finish() without init\n");

  free(le_params_.index_buffer_to_real);
  free(le_params_.index_real_to_buffer);
  free(le_params_.buffer_duy);

  nplane_total_ = 0;
  le_type_ = LINEAR;
  initialised_ = 0;

  return;
}

/*****************************************************************************
 *
 *  le_init_tables
 *
 *  Initialise the buffer look up tables.
 *
 *****************************************************************************/

static void le_init_tables() {

  int ib, ic, ip, n, nb, nh, np;
  int nlocal[3];
  int nplane;
  int rdims[3];

  get_N_local(nlocal);
  nplane = le_get_nplane_local();

  /* Look up table for buffer -> real index */

  /* For each 'x' location in the buffer region, work out the corresponding
   * x index in the real system:
   *   - for each boundary there are 2*nhalo_ buffer planes
   *   - the locations extend nhalo_ points either side of the boundary.
   */

  n = le_get_nxbuffer();

  if (n > 0) {
    le_params_.index_buffer_to_real = (int *) malloc(n*sizeof(int));
    if (le_params_.index_buffer_to_real == NULL) fatal("malloc(le) failed\n");
  }

  ib = 0;
  for (n = 0; n < nplane; n++) {
    ic = le_plane_location(n) - (nhalo_ - 1);
    for (nh = 0; nh < 2*nhalo_; nh++) {
      assert(ib < 2*nhalo_*nplane);
      le_params_.index_buffer_to_real[ib] = ic + nh;
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

  n = (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1);
  le_params_.index_real_nbuffer = n;

  le_params_.index_real_to_buffer = (int *) malloc(n*sizeof(int));
  if (le_params_.index_real_to_buffer == NULL) fatal("malloc(le) failed\n");

  /* Set table in abscence of planes. */
  /* Note the elements of the table at the extreme edges of the local
   * system point outside the system. Accesses must take care. */

   for (ic = 1 - nhalo_; ic <= nlocal[X] + nhalo_; ic++) {
     for (nh = -nhalo_; nh <= nhalo_; nh++) {
       n = (ic + nhalo_ - 1)*(2*nhalo_+1) + (nh + nhalo_);
       assert(n >= 0 && n < (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1));
       le_params_.index_real_to_buffer[n] = ic + nh;
     }
   }

   /* For each position in the buffer, add appropriate
    * corrections in the table. */

   nb = nlocal[X] + nhalo_ + 1;

   for (ib = 0; ib < le_get_nxbuffer(); ib++) {
     np = ib / (2*nhalo_);
     ip = le_plane_location(np);

     /* This bit of logical chooses the first nhalo_ points of the
      * buffer region for each plane as the 'downward' looking part */

     if ((ib - np*2*nhalo_) < nhalo_) {

       /* Looking across the plane in the -ve x-direction */

       for (ic = ip + 1; ic <= ip + nhalo_; ic++) {
	 for (nh = -nhalo_; nh <= -1; nh++) {
	   if (ic + nh == le_params_.index_buffer_to_real[ib]) {
	     n = (ic + nhalo_ - 1)*(2*nhalo_+1) + (nh + nhalo_);
	     assert(n >= 0 && n < (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1));
	     le_params_.index_real_to_buffer[n] = nb+ib;
	   }
	 }
       }
     }
     else {
       /* looking across the plane in the +ve x-direction */

       for (ic = ip - (nhalo_ - 1); ic <= ip; ic++) {
	 for (nh = 1; nh <= nhalo_; nh++) {
	   if (ic + nh == le_params_.index_buffer_to_real[ib]) {
	     n = (ic + nhalo_ - 1)*(2*nhalo_+1) + (nh + nhalo_);
	     assert(n >= 0 && n < (nlocal[X] + 2*nhalo_)*(2*nhalo_ + 1));
	     le_params_.index_real_to_buffer[n] = nb+ib;	   
	   }
	 }
       }
     }
     /* Next buffer point */
   }

   
   /* Buffer velocity jumps. When looking from the real system across
    * a boundary into a given buffer, what is the associated velocity
    * jump? This is +1 for 'looking up' and -1 for 'looking down'.*/

   n = le_get_nxbuffer();
   if (n > 0) {
     le_params_.buffer_duy = (int *) malloc(n*sizeof(double));
     if (le_params_.buffer_duy == NULL) fatal("malloc(buffer_duy) failed\n");
   }

  ib = 0;
  for (n = 0; n < nplane; n++) {
    for (nh = 0; nh < nhalo_; nh++) {
      assert(ib < le_get_nxbuffer());
      le_params_.buffer_duy[ib] = -1;
      ib++;
    }
    for (nh = 0; nh < nhalo_; nh++) {
      assert(ib < le_get_nxbuffer());
      le_params_.buffer_duy[ib] = +1;
      ib++;
    }
  }

  /* Set up a 1-dimensional communicator for transfer of data
   * along the y-direction. */

  rdims[X] = 0;
  rdims[Y] = 1;
  rdims[Z] = 0;
  MPI_Cart_sub(cart_comm(), rdims, &(le_params_.le_comm));

  MPI_Comm_rank(le_params_.le_comm, &n);

  return;
}

/****************************************************************************
 *
 *  le_checks
 *
 *  We check that there are no planes within range of the processor or
 *  periodic halo regions.
 *
 ****************************************************************************/
 
static void le_checks(void) {

  int     nlocal[3];
  int     nplane, n;
  int ifail_local = 0;
  int ifail_global;

  get_N_local(nlocal);

  /* From the local viewpoint, there must be no planes at either
   * x = 1 or x = nlocal[X] (or indeed, within nhalo_ points of
   * a processor or periodic boundary). */

  nplane = le_get_nplane_local();

  for (n = 0; n < nplane; n++) {
    if (le_plane_location(n) <= nhalo_) ifail_local = 1;
    if (le_plane_location(n) > nlocal[X] - nhalo_) ifail_local = 1;
  }

  MPI_Allreduce(&ifail_local, &ifail_global, 1, MPI_INT, MPI_LOR, cart_comm());

  if (ifail_global) {
    fatal("LE_init(): wall at domain boundary\n");
  }

}

/*****************************************************************************
 *
 *  le_get_steady_uy
 *
 *  Return the velocity expected for steady shear profile at
 *  position x (dependent on x-direction only). Takes a local index.
 *
 *****************************************************************************/

double le_get_steady_uy(int ic) {

  int offset[3];
  int nplane;
  double xglobal, uy;

  assert(initialised_);
  get_N_offset(offset);

  /* The shear profile is linear, so the local velocity is just a
   * function of position, modulo the number of planes encountered
   * since the origin. The planes are half way between sites giving
   * the - 0.5. */

  xglobal = offset[X] + (double) ic - 0.5;
  nplane = (int) ((le_params_.dx_min + xglobal)/le_params_.dx_sep);

  uy = xglobal*le_shear_rate() - le_get_plane_uy()*nplane;
 
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

  assert(initialised_);
  get_N_offset(offset);

  /* So, just count the number of blocks from the centre L(X)/2
   * and mutliply by the plane speed. */

  xh = offset[X] + (double) ic - Lmin(X) - 0.5*L(X);
  if (xh > 0.0) {
    n = (0.5 + xh/le_params_.dx_sep);
  }
  else {
    n = (-0.5 + xh/le_params_.dx_sep);
  }
  uy = le_get_plane_uy()*n;

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

  return nplane_total_/cart_size(X);
}

/*****************************************************************************
 *
 *  le_get_nplane_total
 *
 *  Return the total number of planes in the system.
 *
 *****************************************************************************/

int le_get_nplane_total() {

  return nplane_total_;
}

/*****************************************************************************
 *
 *  le_get_plane_uy
 *
 *****************************************************************************/

double le_get_plane_uy() {

  double uy;

  if (le_type_ == LINEAR) uy = le_params_.uy_plane;
  if (le_type_ == OSCILLATORY) {
    /* The -1 is for backwards compatability... */
    double t = get_step() - 1.0;
    uy = le_params_.uy_plane*cos(le_params_.omega*t);
  }

  return le_params_.uy_plane;
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

  assert(initialised_);
  assert(n >= 0 && n < le_get_nplane_local());

  get_N_offset(offset);
  nplane_offset = cart_coords(X)*le_get_nplane_local();

  ix = le_params_.dx_min + (n + nplane_offset)*le_params_.dx_sep - offset[X];

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

  return (2*nhalo_*le_get_nplane_local());
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

  assert(initialised_);
  assert(di >= -nhalo_ && di <= +nhalo_);

  ib = (ic + nhalo_ - 1)*(2*nhalo_ + 1) + di + nhalo_;

  assert(ib >= 0 && ib < le_params_.index_real_nbuffer);

  return le_params_.index_real_to_buffer[ib];
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

  assert(initialised_);
  assert(ib >=0 && ib < le_get_nxbuffer());

  return le_params_.index_buffer_to_real[ib];
}

/*****************************************************************************
 *
 *  le_buffer_displacement
 *
 *  Return the current displacement dy = du_y t for the buffer plane
 *  with x location ib.
 *
 *****************************************************************************/

double le_buffer_displacement(int ib) {

  double dy = 0.0;

  /* The minus one is to ensure the regression test doesn't fail. The
   * displacement oringally updated between the phi and f_i
   * transformations */
  double dt = get_step() - 1.0;

  assert(initialised_);
  assert(ib >= 0 && ib < le_get_nxbuffer());

  if (le_type_ == LINEAR) dy = dt*le_get_plane_uy()*le_params_.buffer_duy[ib];
  if (le_type_ == OSCILLATORY) {
    dy = le_params_.uy_plane*sin(le_params_.omega*dt)/le_params_.omega;
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

MPI_Comm le_communicator() {

  assert(initialised_);
  return le_params_.le_comm;
}

/*****************************************************************************
 *
 *  le_displacement_ranks
 *
 *  For a given  displacement, work out which two ranks in the
 *  one-diemnsional LE communicator are required for communication.
 *
 *****************************************************************************/

void le_displacement_ranks(const double dy, int recv[2], int send[2]) {

  int nlocal[3];
  int noffset[3];
  int j1, jdy;
  int pe_carty1, pe_carty2;

  assert(initialised_);
  get_N_local(nlocal);
  get_N_offset(noffset);

  jdy = floor(fmod(dy, L(Y)));
  j1 = 1 + (noffset[Y] + 1 - nhalo_ - jdy - 2 + 2*N_total(Y)) % N_total(Y);

  /* Receive from ... */

  pe_carty1 = j1 / nlocal[Y];
  pe_carty2 = pe_carty1 + 1;

  MPI_Cart_rank(le_params_.le_comm, &pe_carty1, recv);
  MPI_Cart_rank(le_params_.le_comm, &pe_carty2, recv + 1);

  /* Send to ... */

  pe_carty1 = cart_coords(Y) - ((j1/nlocal[Y]) - cart_coords(Y));
  pe_carty2 = pe_carty1 - 1;

  MPI_Cart_rank(le_params_.le_comm, &pe_carty1, send);
  MPI_Cart_rank(le_params_.le_comm, &pe_carty2, send + 1);

  return;
}

/*****************************************************************************
 *
 *  le_shear_rate
 *
 *  Return the steady shear rate.
 *
 *****************************************************************************/

double le_shear_rate() {

  return (le_get_plane_uy()*nplane_total_/L(X));
}

/*****************************************************************************
 *
 *  le_set_oscillatory
 *
 *****************************************************************************/

void le_set_oscillatory(double period) {

  le_type_ = OSCILLATORY;
  le_params_.omega = 2.0*4.0*atan(1.0)/period;

  return;
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

  int nlocal[3];
  int index;

  get_N_local(nlocal);

  assert(ic >= 1-nhalo_);
  assert(jc >= 1-nhalo_);
  assert(kc >= 1-nhalo_);
  assert(ic <= nlocal[X] + nhalo_ + le_get_nxbuffer());
  assert(jc <= nlocal[Y] + nhalo_);
  assert(kc <= nlocal[Z] + nhalo_);

  index = (nlocal[Y] + 2*nhalo_)*(nlocal[Z] + 2*nhalo_)*(nhalo_ + ic - 1)
    +                            (nlocal[Z] + 2*nhalo_)*(nhalo_ + jc - 1)
    +                                                    nhalo_ + kc - 1;

  return index;
}
