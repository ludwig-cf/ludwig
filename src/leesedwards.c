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
#include "util.h"

typedef struct lees_edw_param_s lees_edw_param_t;

enum shear_type {LEES_EDW_LINEAR, LEES_EDW_OSCILLATORY};

struct lees_edw_s {
  pe_t * pe;                /* Parallel environment */
  cs_t * cs;                /* Coordinate system */
  physics_t * phys;         /* Constants, time step */

  lees_edw_param_t * param; /* Parameters */

  int nref;                 /* Reference count */
  int * icbuff_to_real;     /* look up table */
  int * icreal_to_buff;     /* look up table */
  int * buffer_duy;         /* look up table +/- uy as function of ib */

  MPI_Comm  le_comm;        /* 1-d communicator */
  MPI_Comm  le_plane_comm;  /* 2-d communicator */

  lees_edw_t * target;      /* Device memory */
};

struct lees_edw_param_s {
  /* Local parameters */
  int nplanelocal;          /* Number of planes local domain */
  int nxbuffer;             /* Size of buffer region in x */
  int index_real_nbuffer;
  /* For cs */
  int nhalo;
  int str[3];
  int nlocal[3];
  /* Global parameters */
  int nplanetotal;          /* Total number of planes */
  int type;                 /* Shear type */
  int period;               /* for oscillatory */
  int nt0;                  /* time0 (input as integer) */
  int nsites;               /* Number of sites incl buffer planes */
  double uy;                /* u[Y] for all planes */
  double dx_min;            /* Position first plane */
  double dx_sep;            /* Plane separation */
  double omega;             /* u_y = u_le cos (omega t) for oscillatory */  
  double time0;             /* time offset */
};

static int lees_edw_init(lees_edw_t * le, lees_edw_info_t * info);
static int lees_edw_checks(lees_edw_t * le);
static int lees_edw_init_tables(lees_edw_t * le);

static __constant__ lees_edw_param_t static_param;

int lees_edw_icbuff_to_real(lees_edw_t * le, int ib);
int lees_edw_buffer_duy(lees_edw_t * le, int ib);

/*****************************************************************************
 *
 *  lees_edw_create
 *
 *****************************************************************************/

__host__ int lees_edw_create(pe_t * pe, cs_t * cs, lees_edw_info_t * info,
			     lees_edw_t ** ple) {

  int ndevice;
  lees_edw_t * le = NULL;

  assert(pe);
  assert(cs);

  le = (lees_edw_t *) calloc(1, sizeof(lees_edw_t));
  if (le == NULL) pe_fatal(pe, "calloc(lees_edw_t) failed\n");

  le->param = (lees_edw_param_t *) calloc(1, sizeof(lees_edw_param_t));
  if (le->param == NULL) pe_fatal(pe, "calloc(lees_edw_param_t) failed\n");

  le->pe = pe;
  pe_retain(pe);
  le->cs = cs;
  cs_retain(cs);

  le->param->nplanetotal = 0;
  if (info) lees_edw_init(le, info);
  lees_edw_init_tables(le);
  le->nref = 1;

  targetGetDeviceCount(&ndevice);

  if (ndevice == 1) {
    le->target = le;
  }
  else {
    lees_edw_param_t * tmp;
    targetCalloc((void **) &le->target, sizeof(lees_edw_t));
    targetConstAddress((void **) &tmp, static_param);
    copyToTarget(&le->target->param, (const void *) &tmp,
		 sizeof(lees_edw_param_t *));
    lees_edw_commit(le);
  }

  *ple = le;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_retain
 *
 *****************************************************************************/

__host__ int lees_edw_retain(lees_edw_t * le) {

  assert(le);

  le->nref += 1;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_free
 *
 *****************************************************************************/

__host__ int lees_edw_free(lees_edw_t * le) {

  assert(le);

  le->nref -= 1;

  if (le->nref <= 0) {

    if (le->target != le) targetFree(le->target);

    pe_free(le->pe);
    cs_free(le->cs);
    free(le->icbuff_to_real);
    free(le->icreal_to_buff);
    free(le->buffer_duy);
    free(le->param);
    free(le);
  }

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_commit
 *
 *****************************************************************************/

__host__ int lees_edw_commit(lees_edw_t * le) {

  assert(le);

  copyConstToTarget(&static_param, le->param, sizeof(lees_edw_param_t));

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_target
 *
 *****************************************************************************/

__host__ int lees_edw_target(lees_edw_t * le, lees_edw_t ** target) {

  assert(le);
  assert(target);

  *target = le->target;

  return 0;
}

/*****************************************************************************
 *
 * lees_edw_nplane_set
 *
 *****************************************************************************/

int lees_edw_nplane_set(lees_edw_t * le, int nplanetotal) {

  assert(le);

  le->param->nplanetotal = nplanetotal;

  return 0;
}

/*****************************************************************************
 *
 * lees_edw_plane_uy_set
 *
 *****************************************************************************/

int lees_edw_plane_uy_set(lees_edw_t * le, double uy) {

  assert(le);

  le->param->uy = uy;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_oscillatory_set
 *
 *****************************************************************************/

int lees_edw_oscillatory_set(lees_edw_t * le, int period) {

  assert(le);

  le->param->type = LEES_EDW_OSCILLATORY;
  le->param->period = period;
  le->param->omega = 2.0*4.0*atan(1.0)/le->param->period;

  return 0;
} 

/*****************************************************************************
 *
 *  lees_edw_toffset_set
 *
 *****************************************************************************/

int lees_edw_toffset_set(lees_edw_t * le, int nt0) {

  assert(le);

  le->param->nt0 = nt0;

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

  le->param->nplanetotal = info->nplanes;
  le->param->uy = info->uy;
  le->param->type = info->type;
  le->param->period = info->period;
  le->param->nt0 = info->nt0;

  cs_ntotal(le->cs, ntotal);
  physics_ref(&le->phys);

  if (le->param->nplanetotal > 0) {

    if (ntotal[X] % le->param->nplanetotal) {
      pe_info(le->pe, "System size x-direction: %d\n", ntotal[X]);
      pe_info(le->pe, "Number of LE planes requested: %d\n", info->nplanes);
      pe_fatal(le->pe, "Number of planes must divide system size\n");
    }

    le->param->dx_sep = 1.0*ntotal[X] / le->param->nplanetotal;
    le->param->dx_min = 0.5*le->param->dx_sep;
    le->param->time0 = 1.0*le->param->nt0;
  }

  lees_edw_checks(le);

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_nplane_total
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_nplane_total(lees_edw_t * le) {

  assert(le);

  return le->param->nplanetotal;
}

/*****************************************************************************
 *
 *  lees_edw_nplane_local
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_nplane_local(lees_edw_t * le) {

  assert(le);

  return le->param->nplanelocal;
}

/*****************************************************************************
 *
 *  lees_edw_plane_uy
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_plane_uy(lees_edw_t * le, double * uy) {

  assert(le);

  *uy = le->param->uy;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_nxbuffer
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_nxbuffer(lees_edw_t * le, int * nxb) {

  assert(le);

  *nxb = le->param->nxbuffer;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_info
 *
 *  A human-readable summary of Lees Edwards information.
 *
 *****************************************************************************/

__host__ int lees_edw_info(lees_edw_t * le) {

  int np;
  double gammadot;

  assert(le);

  if (le->param->nplanetotal > 0) {

    pe_info(le->pe, "\nLees-Edwards boundary conditions are active:\n");

    lees_edw_shear_rate(le, &gammadot);

    for (np = 0; np < le->param->nplanetotal; np++) {
      pe_info(le->pe, "LE plane %d is at x = %d with speed %f\n", np+1,
	   (int)(le->param->dx_min + np*le->param->dx_sep), le->param->uy);
    }

    if (le->param->type == LEES_EDW_LINEAR) {
      pe_info(le->pe, "Overall shear rate = %f\n", gammadot);
    }
    else {
      pe_info(le->pe, "Oscillation period: %d time steps\n", le->param->period);
      pe_info(le->pe, "Maximum shear rate = %f\n", gammadot);
    }

    pe_info(le->pe, "\n");
    pe_info(le->pe, "Lees-Edwards time offset (time steps): %8d\n", le->param->nt0);
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
  int rdims[3];
  int cartsz[3];
  MPI_Comm cartcomm;

  assert(le);

  cs_nhalo(le->cs, &nhalo);
  cs_nlocal(le->cs, le->param->nlocal);
  cs_cartsz(le->cs, cartsz);
  cs_cart_comm(le->cs, &cartcomm);
  cs_strides(le->cs, le->param->str+X, le->param->str+Y, le->param->str+Z);

  le->param->nhalo = nhalo;
  le->param->nplanelocal = le->param->nplanetotal / cartsz[X];

  /* Look up table for buffer -> real index */

  /* For each 'x' location in the buffer region, work out the corresponding
   * x index in the real system:
   *   - for each boundary there are 2*nhalo buffer planes
   *   - the locations extend nhalo points either side of the boundary.
   */

  le->param->nxbuffer = 2*nhalo*le->param->nplanelocal;

  if (le->param->nxbuffer > 0) {
    le->icbuff_to_real = (int *) calloc(le->param->nxbuffer, sizeof(int));
    if (le->icbuff_to_real == NULL) pe_fatal(le->pe, "calloc(le) failed\n");
  }

  ib = 0;
  for (n = 0; n < le->param->nplanelocal; n++) {
    ic = lees_edw_plane_location(le, n) - (nhalo - 1);
    for (nh = 0; nh < 2*nhalo; nh++) {
      assert(ib < 2*nhalo*le->param->nplanelocal);
      le->icbuff_to_real[ib] = ic + nh;
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

  n = (le->param->nlocal[X] + 2*nhalo)*(2*nhalo + 1);
  le->param->index_real_nbuffer = n;

  le->icreal_to_buff = (int *) calloc(n, sizeof(int));
  if (le->icreal_to_buff == NULL) pe_fatal(le->pe, "calloc(le) failed\n");

  /* Set table in abscence of planes. */
  /* Note the elements of the table at the extreme edges of the local
   * system point outside the system. Accesses must take care. */

   for (ic = 1 - nhalo; ic <= le->param->nlocal[X] + nhalo; ic++) {
     for (nh = -nhalo; nh <= nhalo; nh++) {
       n = (ic + nhalo - 1)*(2*nhalo+1) + (nh + nhalo);
       assert(n >= 0 && n < (le->param->nlocal[X] + 2*nhalo)*(2*nhalo + 1));
       le->icreal_to_buff[n] = ic + nh;
     }
   }

   /* For each position in the buffer, add appropriate
    * corrections in the table. */

   nb = le->param->nlocal[X] + nhalo + 1;

   for (ib = 0; ib < le->param->nxbuffer; ib++) {
     np = ib / (2*nhalo);
     ip = lees_edw_plane_location(le, np);

     /* This bit of logic chooses the first nhalo points of the
      * buffer region for each plane as the 'downward' looking part */

     if ((ib - np*2*nhalo) < nhalo) {

       /* Looking across the plane in the -ve x-direction */

       for (ic = ip + 1; ic <= ip + nhalo; ic++) {
	 for (nh = -nhalo; nh <= -1; nh++) {
	   if (ic + nh == le->icbuff_to_real[ib]) {
	     n = (ic + nhalo - 1)*(2*nhalo+1) + (nh + nhalo);
	     assert(n >= 0 && n < (le->param->nlocal[X] + 2*nhalo)*(2*nhalo + 1));
	     le->icreal_to_buff[n] = nb+ib;
	   }
	 }
       }
     }
     else {
       /* looking across the plane in the +ve x-direction */

       for (ic = ip - (nhalo - 1); ic <= ip; ic++) {
	 for (nh = 1; nh <= nhalo; nh++) {
	   if (ic + nh == le->icbuff_to_real[ib]) {
	     n = (ic + nhalo - 1)*(2*nhalo+1) + (nh + nhalo);
	     assert(n >= 0 && n < (le->param->nlocal[X] + 2*nhalo)*(2*nhalo + 1));
	     le->icreal_to_buff[n] = nb+ib;	   
	   }
	 }
       }
     }
     /* Next buffer point */
   }

   /* Buffer velocity jumps. When looking from the real system across
    * a boundary into a given buffer, what is the associated velocity
    * jump? This is +1 for 'looking up' and -1 for 'looking down'.*/

   if (le->param->nxbuffer > 0) {
     le->buffer_duy = (int *) calloc(le->param->nxbuffer, sizeof(double));
     if (le->buffer_duy == NULL) pe_fatal(le->pe,"calloc(buffer_duy) failed\n");
   }

  ib = 0;
  for (n = 0; n < le->param->nplanelocal; n++) {
    for (nh = 0; nh < nhalo; nh++) {
      assert(ib < le->param->nxbuffer);
      le->buffer_duy[ib] = -1;
      ib++;
    }
    for (nh = 0; nh < nhalo; nh++) {
      assert(ib < le->param->nxbuffer);
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

  int n;
  int ic;
  int ifail_local = 0;
  int ifail_global;
  int cartsz[3];
  MPI_Comm cartcomm;

  assert(le);

  cs_cartsz(le->cs, cartsz);
  cs_cart_comm(le->cs, &cartcomm);

  /* From the local viewpoint, there must be no planes at either
   * x = 1 or x = nlocal[X] (or indeed, within nhalo points of
   * a processor or periodic boundary). */

  for (n = 0; n < le->param->nplanelocal; n++) {
    ic = lees_edw_plane_location(le, n);
    if (ic <= le->param->nhalo) ifail_local = 1;
    if (ic  > le->param->nlocal[X] - le->param->nhalo) ifail_local = 1;
  }

  MPI_Allreduce(&ifail_local, &ifail_global, 1, MPI_INT, MPI_LOR, cartcomm);

  if (ifail_global) {
    pe_fatal(le->pe, "Wall at domain boundary\n");
  }

  /* As nplane_local = ntotal/cartsz[X] (integer division) we must have
   * ntotal % cartsz[X] = 0 */

  if ((le->param->nplanetotal % cartsz[X]) != 0) {
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

__host__ __device__
int lees_edw_nsites(lees_edw_t * le, int * nsites) {

  assert(le);

  *nsites = (le->param->nlocal[X] + 2*le->param->nhalo + le->param->nxbuffer)
    *(le->param->nlocal[Y] + 2*le->param->nhalo)
    *(le->param->nlocal[Z] + 2*le->param->nhalo);

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

__host__ __device__
int lees_edw_steady_uy(lees_edw_t * le, int ic, double * uy) {

  int offset[3];
  int nplane;
  double gammadot;
  double xglobal;

  assert(le);
  assert(le->param->type == LEES_EDW_LINEAR);

  cs_nlocal_offset(le->cs, offset);
  lees_edw_shear_rate(le, &gammadot);

  /* The shear profile is linear, so the local velocity is just a
   * function of position, modulo the number of planes encountered
   * since the origin. The planes are half way between sites giving
   * the - 0.5. */

  xglobal = offset[X] + (double) ic - 0.5;
  nplane = (int) ((le->param->dx_min + xglobal)/le->param->dx_sep);

  *uy = xglobal*gammadot - le->param->uy*nplane;
 
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

__host__ __device__
int lees_edw_block_uy(lees_edw_t * le, int ic, double * uy) {

  int offset[3];
  int n;
  double xh;
  double lmin[3];
  double ltot[3];

  assert(le);
  assert(le->param->type == LEES_EDW_LINEAR);

  cs_lmin(le->cs, lmin);
  cs_ltot(le->cs, ltot);
  cs_nlocal_offset(le->cs, offset);

  /* So, just count the number of blocks from the centre L_x/2
   * and mutliply by the plane speed. */

  xh = offset[X] + (double) ic - lmin[X] - 0.5*ltot[X];
  if (xh > 0.0) {
    n = (0.5 + xh/le->param->dx_sep);
  }
  else {
    n = (-0.5 + xh/le->param->dx_sep);
  }

  *uy = le->param->uy*n;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_plane_uy_now
 *
 *  Return the current plane velocity for time t.
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_plane_uy_now(lees_edw_t * le, double t, double * uy) {

  double tle;

  assert(le);

  tle = t - le->param->time0;
  assert(tle >= 0.0);

  *uy = le->param->uy;
  if (le->param->type == LEES_EDW_OSCILLATORY) *uy *= cos(le->param->omega*tle);

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

__host__ __device__
int lees_edw_plane_location(lees_edw_t * le, int np) {

  int offset[3];
  int nplane_offset;
  int cartcoords[3];
  int ix;

  assert(le);
  assert(np >= 0 && np < le->param->nplanelocal);

  cs_cart_coords(le->cs, cartcoords);
  cs_nlocal_offset(le->cs, offset);
  nplane_offset = cartcoords[X]*le->param->nplanelocal;

  ix = le->param->dx_min + (np + nplane_offset)*le->param->dx_sep - offset[X];

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

__host__ __device__
int lees_edw_index_real_to_buffer(lees_edw_t * le,  int ic,  int idisplace) {

  int ib;

  assert(le);
  assert(le->icreal_to_buff);

  assert(idisplace >= -le->param->nhalo && idisplace <= +le->param->nhalo);

  ib = (ic + le->param->nhalo - 1)*(2*le->param->nhalo + 1) + idisplace + le->param->nhalo;

  assert(ib >= 0 && ib < le->param->index_real_nbuffer);

  assert(le->icreal_to_buff[ib] == lees_edw_ic_to_buff(le, ic, idisplace));
  return le->icreal_to_buff[ib];
}

/*****************************************************************************
 *
 *  lees_edw_index_buffer_to_real
 *
 *  For x index in the buffer region, return the corresponding
 *  x index in the real system.
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_index_buffer_to_real(lees_edw_t * le, int ib) {

  assert(le);
  assert(le->icbuff_to_real);
  assert(ib >=0 && ib < le->param->nxbuffer);

  assert(le->icbuff_to_real[ib] == lees_edw_icbuff_to_real(le, ib));
  return le->icbuff_to_real[ib];
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

__host__ __device__
int lees_edw_buffer_displacement(lees_edw_t * le, int ib, double t, double * dy) {

  double tle;

  assert(le);
  assert(ib >= 0 && ib < le->param->nxbuffer);

  tle = t - le->param->time0;
  assert(tle >= 0.0);

  *dy = 0.0;

  if (le->param->type == LEES_EDW_LINEAR) {
    *dy = tle*le->param->uy*le->buffer_duy[ib];
    assert(le->buffer_duy[ib] == lees_edw_buffer_duy(le, ib));
  }

  if (le->param->type == LEES_EDW_OSCILLATORY) {
    *dy = le->param->uy*sin(le->param->omega*tle)/le->param->omega;
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

__host__ int lees_edw_comm(lees_edw_t * le, MPI_Comm * comm) {

  assert(le);

  *comm = le->le_comm;

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_plane_comm
 *
 *****************************************************************************/

__host__ int lees_edw_plane_comm(lees_edw_t * le, MPI_Comm * comm) {

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

__host__
int lees_edw_jstart_to_mpi_ranks(lees_edw_t * le, const int j1, int send[3],
				 int recv[3]) {

  int cartcoords[3];
  int pe_carty1, pe_carty2, pe_carty3;

  assert(le);

  cs_cart_coords(le->cs, cartcoords);

  /* Receive from ... */

  pe_carty1 = (j1 - 1) / le->param->nlocal[Y];
  pe_carty2 = pe_carty1 + 1;
  pe_carty3 = pe_carty1 + 2;

  MPI_Cart_rank(le->le_comm, &pe_carty1, recv);
  MPI_Cart_rank(le->le_comm, &pe_carty2, recv + 1);
  MPI_Cart_rank(le->le_comm, &pe_carty3, recv + 2);

  /* Send to ... */

  pe_carty1 = cartcoords[Y] - (((j1 - 1)/le->param->nlocal[Y]) - cartcoords[Y]);
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

__host__ __device__
int lees_edw_shear_rate(lees_edw_t * le, double * gammadot) {

  double ltot[3];

  assert(le);
  cs_ltot(le->cs, ltot);

  *gammadot = le->param->uy*le->param->nplanetotal/ltot[X];

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_index
 *
 *  Compute the one-dimensional index from coordinates ic, jc, kc.
 *  This differs from cs_index() in that it allows the buffer
 *  region to be used.
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_index(lees_edw_t * le, int ic, int jc, int kc) {

  assert(le);

  assert(ic >= 1-le->param->nhalo);
  assert(jc >= 1-le->param->nhalo);
  assert(kc >= 1-le->param->nhalo);
  assert(ic <= le->param->nlocal[X] + le->param->nhalo + le->param->nxbuffer);
  assert(jc <= le->param->nlocal[Y] + le->param->nhalo);
  assert(kc <= le->param->nlocal[Z] + le->param->nhalo);

  return (le->param->str[X]*(le->param->nhalo + ic - 1) +
	  le->param->str[Y]*(le->param->nhalo + jc - 1) +
	  le->param->str[Z]*(le->param->nhalo + kc - 1));
}

/*****************************************************************************
 *
 *  lees_edw_index_v
 *
 *****************************************************************************/

__host__ __device__ void lees_edw_index_v(lees_edw_t * le, int ic[NSIMDVL],
					  int jc[NSIMDVL], int kc[NSIMDVL],
					  int index[NSIMDVL]) {
  int iv;
  assert(le);

  for (iv = 0; iv < NSIMDVL; iv++) {
    index[iv] = lees_edw_index(le, ic[iv], jc[iv], kc[iv]);
  }

  return;
} 

/*****************************************************************************
 *
 *  lees_edw_nlocal
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_nlocal(lees_edw_t * le, int nlocal[3]) {

  assert(le);

  return cs_nlocal(le->cs, nlocal);
}

/*****************************************************************************
 *
 *  lees_edw_strides
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_strides(lees_edw_t * le, int * xs, int * ys, int * zs) {

  assert(le);

  return cs_strides(le->cs, xs, ys, zs);
}

/*****************************************************************************
 *
 *  lees_edw_nhalo
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_nhalo(lees_edw_t * le, int * nhalo) {

  assert(le);

  return cs_nhalo(le->cs, nhalo);
}

/*****************************************************************************
 *
 *  lees_edw_ltot
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_ltot(lees_edw_t * le, double ltot[3]) {

  assert(le);

  return cs_ltot(le->cs, ltot);
}

/*****************************************************************************
 *
 *  lees_edw_cartsz
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_cartsz(lees_edw_t * le, int cartsz[3]) {

  assert(le);

  return cs_cartsz(le->cs, cartsz);
}

/*****************************************************************************
 *
 *  lees_edw_ntotal
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_ntotal(lees_edw_t * le, int ntotal[3]) {

  assert(le);

  return cs_ntotal(le->cs, ntotal);
}

/*****************************************************************************
 *
 *  lees_edw_nlocal_offset
 *
 *****************************************************************************/

__host__ __device__
int lees_edw_nlocal_offset(lees_edw_t * le, int noffset[3]) {

  assert(le);

  return cs_nlocal_offset(le->cs, noffset);
}

/*****************************************************************************
 *
 *  lees_edw_cart_coords
 *
 *****************************************************************************/

__host__ __device__
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
  *dy = t*le->param->uy;

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
  assert(ib >= 0 && ib < le->param->nxbuffer);

  if (le->param->type == LEES_EDW_LINEAR) {
    ule[X] = 0.0;
    ule[Y] = le->param->uy*le->buffer_duy[ib];
    assert(le->buffer_duy[ib] == lees_edw_buffer_duy(le, ib));
    ule[Z] = 0.0;
  }
  else {
    assert(0); /* Check delta u as function of (ib,t) */
  }

  return 0;
}

/*****************************************************************************
 *
 *  lees_edw_icbuff_to_real
 *
 *****************************************************************************/

__host__ int lees_edw_icbuff_to_real(lees_edw_t * le, int ib) {

  int ic;
  int p;

  assert(le);
  assert(ib >= 0 && ib < le->param->nxbuffer);

  p = ib / (2*le->param->nhalo);

  ic = lees_edw_plane_location(le, p) - (le->param->nhalo - 1);
  ic = ic + ib % (2*le->param->nhalo);

  return ic;
}

/******************************************************************************
 *
 *  lees_edw_ic_to_buff
 *
 ******************************************************************************/

__host__ __device__
int lees_edw_ic_to_buff(lees_edw_t * le, int ic, int di) {

  int ib;
  int p, ip;
  int nh;

  assert(le);
  assert(di <= le->param->nhalo);
  assert(di >= -le->param->nhalo);

  ib = ic + di;

  if (le->param->nplanelocal > 0) {

    p = ic / (le->param->nlocal[X]/le->param->nplanelocal);
    p = imax(0, imin(p, le->param->nplanelocal - 1));

    nh = le->param->nhalo;
    ip = lees_edw_plane_location(le, p) - (nh - 1);

    if (di > 0 && (ic >= ip && ic < ip + nh) && (ic + di >= ip + nh)) {
      ib = le->param->nlocal[X] + (1 + 2*p)*nh + (ic - ip + 1) + di;
      return ib;
    }

    ip = lees_edw_plane_location(le, p) + 1;

    if (di < 0 && (ic >= ip && ic < ip + nh) && (ic + di < ip)) {
      ib = le->param->nlocal[X] + (2 + 2*p)*nh + (ic - ip + 1) + di;
      return ib;
    }
  }

  return ib;
}

/******************************************************************************
 *
 *  lees_edw_buffer_duy
 *
 ******************************************************************************/

__host__ __device__ int lees_edw_buffer_duy(lees_edw_t * le, int ib) {

  int pm1;

  assert(le);
  assert(ib >= 0 && ib < le->param->nxbuffer);

  pm1 = +1;
  if (ib % (2*le->param->nhalo) < le->param->nhalo) pm1 = -1;

  return pm1;
}
