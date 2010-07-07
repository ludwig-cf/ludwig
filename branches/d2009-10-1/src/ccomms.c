/*****************************************************************************
 *
 *  ccomms.c
 *
 *  Colloid communications.
 *
 *  This deals with exchanges of colloid information at the periodic
 *  and processor boundaries.
 *
 *  MPI (or serial, with some overhead).
 *
 *  $Id: ccomms.c,v 1.12.2.7 2010-07-07 11:16:07 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "util.h"
#include "ccomms.h"

static MPI_Datatype _mpi_halo;
static MPI_Datatype _mpi_sum1;
static MPI_Datatype _mpi_sum2;
static MPI_Datatype _mpi_sum7;
static MPI_Datatype _mpi_sum8;

/* Colloid message structures
 *
 * Halo:
 *   - exchange particle state information for "halo swaps"
 * Sum1...:
 *   - for summation of ...
 */

typedef struct colloid_halo_message      Colloid_halo_message;
typedef struct colloid_sum_message_type1 Colloid_sum_message_one;
typedef struct colloid_sum_message_type2 Colloid_sum_message_two;
typedef struct colloid_sum_message_type7 Colloid_sum_message_sev;
typedef struct colloid_sum_message_type8 Colloid_sum_message_eig;


struct colloid_halo_message {
  int      index;
  double   a0;
  double   ah;
  double   r[3];
  double   v[3];
  double   omega[3];
  double   direction[3];
  double   dp;
  double   b1;
  double   b2;
  double   cosine_ca;
  double   random[6];
  double   s[3];
  double   dr[3];
  double   cwetting;
  double   hwetting;
};

struct colloid_sum_message_type1 {
  int    index;
  double cbar[3];
  double rxcbar[3];
  double sumw;
  double force[3];
  double torque[3];
  double deltam;
  double deltaphi;
};

struct colloid_sum_message_type2 {
  int    index;
  double f0[3];
  double t0[3];
  double zeta[21];
  double sump;
};

struct colloid_sum_message_type7 {
  int index;
  double f0[3];
};

struct colloid_sum_message_type8 {
  int     index;
  double  fc0[3];
  double  tc0[3];
};


static void CCOM_load_halo_buffer(Colloid *, int, const double rp[3]);
static void CCOM_unload_halo_buffer(Colloid *, int);
static void CCOM_load_sum_message_buffer(Colloid *, int, int);
static void CCOM_unload_sum_message_buffer(Colloid *, int, int);
static void CCOM_exchange_halo_sum(int, int, int, int);

static double colloid_min_ah(void);

static void CMPI_accept_new(int);
static int  CMPI_exchange_halo(int, int, int);
static void CMPI_exchange_sum_type1(int, int, int);
static void CMPI_exchange_sum_type2(int, int, int);
static void CMPI_exchange_sum_type7(int, int, int);
static void CMPI_exchange_sum_type8(int, int, int);
static void CMPI_anull_buffers(void);
static void CMPI_init_messages(void);

static int                       _halo_message_nmax;
static int                       _halo_message_size[5];

static Colloid_halo_message    * _halo_send;
static Colloid_halo_message    * _halo_recv;
static Colloid_sum_message_one * _halo_send_one;
static Colloid_sum_message_one * _halo_recv_one;
static Colloid_sum_message_two * _halo_send_two;
static Colloid_sum_message_two * _halo_recv_two;
static Colloid_sum_message_sev * _halo_send_sev;
static Colloid_sum_message_sev * _halo_recv_sev;
static Colloid_sum_message_eig * _halo_send_eig;
static Colloid_sum_message_eig * _halo_recv_eig;



/*****************************************************************************
 *
 *  CCOM_init_halos
 *
 *  Allocate buffer space for the particle transfers. Capacity
 *  is computed on the basis of particles with ah 
 *  100% solid volume fraction (which should always be enough).
 *
 *****************************************************************************/

void CCOM_init_halos() {

  double ah, vcell;
  int ncell;

  /* Work out the volume of each halo region and take the
   * largest. Then work out how many particles will fit
   * in the volume. */

  vcell = L(X)*L(Y)*L(Z);
  vcell /= (Ncell(X)*cart_size(X)*Ncell(Y)*cart_size(Y)*Ncell(Z)*cart_size(Z));
  ncell = imax(Ncell(X), Ncell(Y));
  ncell = imax(Ncell(Z), ncell);
  ncell += 2; /* Add halo cells */

  ah = colloid_min_ah();

  _halo_message_nmax = ncell*ncell*vcell/(ah*ah*ah);

  info("\nColloid message initiailisation...\n");
  info("Taking volume of halo region = %f x %d and ah = %f\n",
       vcell, ncell*ncell, ah);
  info("Allocating space for %d particles in halo buffers\n",
       _halo_message_nmax);

  _halo_message_size[CHALO_TYPE1] = sizeof(Colloid_sum_message_one);
  _halo_message_size[CHALO_TYPE2] = sizeof(Colloid_sum_message_two);
  _halo_message_size[CHALO_TYPE7] = sizeof(Colloid_sum_message_sev);
  _halo_message_size[CHALO_TYPE8] = sizeof(Colloid_sum_message_eig);

  info("Requesting %d bytes for halo messages\n",
       2*_halo_message_nmax*sizeof(Colloid_halo_message));
  info("Requesting %d bytes for type one messages\n",
       2*_halo_message_nmax*_halo_message_size[CHALO_TYPE1]);
  info("Requesting %d bytes for type two messages\n",
       2*_halo_message_nmax*_halo_message_size[CHALO_TYPE2]);
  info("Requesting %d bytes for type eight messages\n",
       2*_halo_message_nmax*_halo_message_size[CHALO_TYPE8]);

  _halo_send     = (Colloid_halo_message *)
    calloc(_halo_message_nmax, sizeof(Colloid_halo_message));
  _halo_recv     = (Colloid_halo_message *)
    calloc(_halo_message_nmax, sizeof(Colloid_halo_message));
  _halo_send_one = (Colloid_sum_message_one *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_one));
  _halo_recv_one = (Colloid_sum_message_one *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_one));
  _halo_send_two = (Colloid_sum_message_two *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_two));
  _halo_recv_two = (Colloid_sum_message_two *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_two));
  _halo_send_sev = (Colloid_sum_message_sev *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_sev));
  _halo_recv_sev = (Colloid_sum_message_sev *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_sev));
  _halo_send_eig = (Colloid_sum_message_eig *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_eig));
  _halo_recv_eig = (Colloid_sum_message_eig *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_eig));


  if (_halo_send      == NULL) fatal("_halo_send failed");
  if (_halo_recv      == NULL) fatal("_halo_recv failed");
  if (_halo_send_one  == NULL) fatal("_halo_send_one failed");
  if (_halo_recv_one  == NULL) fatal("_halo_recv_one failed");
  if (_halo_send_two  == NULL) fatal("_halo_send_two failed");
  if (_halo_recv_two  == NULL) fatal("_halo_recv_two failed");
  if (_halo_send_sev  == NULL) fatal("_halo_send_sev failed");
  if (_halo_recv_sev  == NULL) fatal("_halo_recv_sev failed");
  if (_halo_send_eig  == NULL) fatal("_halo_send_eig failed");
  if (_halo_recv_eig  == NULL) fatal("_halo_recv_eig failed");


  CMPI_init_messages();

  return;
}



/*****************************************************************************
 *
 *  CCOM_halo_particles
 *
 *  This drives the exchange of particle information between the
 *  halo regions.
 *
 *  In each coordinate direction in turn, and then in the backward
 *  and forward directions, particles are loaded into the
 *  send buffer and exchanged between processes.
 *
 *  Issues
 *
 *****************************************************************************/


void CCOM_halo_particles() {

  int         nforw, nback, nrecv;
  int         ic, jc, kc;

  Colloid *   p_colloid;
  double     rperiod[3];

  /* Non-periodic system requires no halo exchanges */
  if (is_periodic(X) == 0) return;

  /* This is for testing purposes: it sets all the indexes in the
   * communication buffers to -1 in order to pick up attempted
   * access to stale information. It does little harm to keep. */

  CMPI_anull_buffers();


  /* Periodic boundary conditions are slightly clunky at the moment,
   * so could be improved */

  rperiod[X] = 0.0;
  rperiod[Y] = 0.0;
  rperiod[Z] = 0.0;

  /* x-direction */

  nforw = 0;
  nback = 0;

  /* Backward x direction. */

  for (jc = 1; jc <= Ncell(Y); jc++) {
    for (kc = 1; kc <= Ncell(Z); kc++) {

      ic = 1;
      p_colloid = colloids_cell_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(X) == 0) rperiod[X] = L(X)*(1.0 - DBL_EPSILON);
	CCOM_load_halo_buffer(p_colloid, nback, rperiod);
	rperiod[X] = 0.0;
	++nback;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }
    }
  }

  /* Forward x direction. */

  for (jc = 1; jc <= Ncell(Y); jc++) {
    for (kc = 1; kc <= Ncell(Z); kc++) {

      ic = Ncell(X);
      p_colloid = colloids_cell_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(X) == cart_size(X)-1) rperiod[X] = -L(X);
	CCOM_load_halo_buffer(p_colloid, nback + nforw, rperiod);
	rperiod[X] = 0.0;
	++nforw;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }
    }
  }

  nrecv = CMPI_exchange_halo(X, nforw, nback);
  CMPI_accept_new(nrecv);

  if (is_periodic(Y) == 0) return;

  /* Backward y direction. */

  nforw = 0;
  nback = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (kc = 1; kc <= Ncell(Z)    ; kc++) {

      jc = 1;
      p_colloid = colloids_cell_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Y) == 0) rperiod[Y] = L(Y)*(1.0 - DBL_EPSILON);
	CCOM_load_halo_buffer(p_colloid, nback, rperiod);
	rperiod[Y] = 0.0;
	++nback;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }
    }
  }

  /* Forward y direction. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (kc = 1; kc <= Ncell(Z)    ; kc++) {

      jc = Ncell(Y);
      p_colloid = colloids_cell_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Y) == cart_size(Y)-1) rperiod[Y] = -L(Y);
	CCOM_load_halo_buffer(p_colloid, nback + nforw, rperiod);
	rperiod[Y] = 0.0;
	++nforw;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }

    }
  }

  nrecv = CMPI_exchange_halo(Y, nforw, nback);
  CMPI_accept_new(nrecv);


  if (is_periodic(Z) == 0) return;

  /* Backward z-direction. */

  nforw = 0;
  nback = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {

      kc = 1;
      p_colloid = colloids_cell_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Z) == 0) rperiod[Z] = L(Z)*(1.0 - DBL_EPSILON);
	CCOM_load_halo_buffer(p_colloid, nback, rperiod);
	rperiod[Z] = 0.0;
	++nback;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }
    }
  }

  /* Forward z direction. */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {

      kc = Ncell(Z);
      p_colloid = colloids_cell_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Z) == cart_size(Z)-1) rperiod[Z] = -L(Z);
	CCOM_load_halo_buffer(p_colloid, nback + nforw, rperiod);
	rperiod[Z] = 0.0;
	++nforw;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }
    }
  }

  nrecv = CMPI_exchange_halo(Z, nforw, nback);
  CMPI_accept_new(nrecv);

  return;
}


/*****************************************************************************
 *
 *  CMPI_accept_new
 *
 *  This routine examines the current nrecv particles in the
 *  receive buffer, and creates a copy of the particle in the
 *  halo region if one doesn't already exist.
 *
 *****************************************************************************/

void CMPI_accept_new(int nrecv) {

  Colloid * p_colloid;
  Colloid * p_existing;
  int       cell[3];
  int       exists;
  int       n;

  for (n = 0; n < nrecv; n++) {

    /* Load the new particle */
    p_colloid = colloid_allocate();
    CCOM_unload_halo_buffer(p_colloid, n);

    /* See where it wants to go in the cell list */

    colloids_cell_coords(p_colloid->s.r, cell);

    /* Have a look at the existing particles in the list location:
     * if it's already present, make a copy of the state; if it's new,
     * the incoming particle does get added to the cell list. */

    exists = 0;

    /* At this stage newc may actually be out of the domain, owing to
     * round-off error in position of particles whose coords have
     * changed crossing the periodic boundaries. Trap this here?
     * The whole mechanism needs cleaning up. */

    p_existing = colloids_cell_list(cell[X], cell[Y], cell[Z]);

    while (p_existing) {

      if (p_colloid->s.index == p_existing->s.index) {

	p_existing->s = p_colloid->s;
	exists = 1;
      }

      p_existing = p_existing->next;
    }

    if (exists) {
      /* Just drop the incoming copy */
      colloid_free(p_colloid);
    }
    else {
      /* Add the incoming colloid */
      colloids_cell_insert_colloid(p_colloid);
    }
  }

  return;
}


/*****************************************************************************
 *
 *  CCOM_halo_sum()
 *
 *  Perform sums over all colloid links.
 *
 *
 ****************************************************************************/

void CCOM_halo_sum(const int type) {

  int         ic, jc, kc;
  int         n, nback, nforw;
  Colloid *   p_colloid;

  /* Send back and forward in the x-direction. First, back */

  if (is_periodic(X) == 0) return;

  nforw = 0;
  nback = 0;

  for (ic = 0; ic <= 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_load_sum_message_buffer(p_colloid, nback, type);
	  ++nback;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* Send forward in the x-direction */

  for (ic = Ncell(X); ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_load_sum_message_buffer(p_colloid, nback + nforw, type);
	  ++nforw;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }


  /* Exchange halo */
  CCOM_exchange_halo_sum(X, type, nback, nforw);
  
  /* Add the incoming subtotals to the corresponding particle in
   * the local sub-domain */

  /* Receive from the backward x-direction */

  n = 0;

  for (ic = 0; ic <= 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_unload_sum_message_buffer(p_colloid, nforw + n, type);
	  ++n;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* Receive from the forward x-direction */

  n = 0;

  for (ic = Ncell(X); ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_unload_sum_message_buffer(p_colloid, n, type);
	  ++n;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }


  /* y-direction */

  if (is_periodic(Y) == 0) return;

  nforw = 0;
  nback = 0;

  /* Send back in the y-direction */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_load_sum_message_buffer(p_colloid, nback, type);
	  ++nback;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* Send forward in the y-direction */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = Ncell(Y); jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_load_sum_message_buffer(p_colloid, nback + nforw, type);
	  ++nforw;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }


  /* Exchange halo */
  CCOM_exchange_halo_sum(Y, type, nback, nforw);

  /* Extract the incoming partial sums from the receive buffer and
   * add to the corresponding particle in the local domain */


  n = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_unload_sum_message_buffer(p_colloid, nforw + n, type);
	  ++n;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  n = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = Ncell(Y); jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_unload_sum_message_buffer(p_colloid, n, type);
	  ++n;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }



  /* z-direction */

  if (is_periodic(Z) == 0) return;

  nforw = 0;
  nback = 0;

  /* Send back in the z-direction */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_load_sum_message_buffer(p_colloid, nback, type);
	  ++nback;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* Send forward in the z-direction */

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = Ncell(Z); kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_load_sum_message_buffer(p_colloid, nback + nforw, type);
	  ++nforw;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* Exchange halo in z-direction */
  CCOM_exchange_halo_sum(Z, type, nback, nforw);

  /* Receive from the forward z-direction */

  n = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = 0; kc <= 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_unload_sum_message_buffer(p_colloid, nforw + n, type);
	  ++n;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  /* Receive from the backward z-direction */

  n = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {
      for (kc = Ncell(Z); kc <= Ncell(Z) + 1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  CCOM_unload_sum_message_buffer(p_colloid, n, type);
	  ++n;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  CCOM_exchange_halo_sum
 *
 *  Exchange the particles in the halo message buffers.
 *
 *****************************************************************************/

void CCOM_exchange_halo_sum(int dimension, int type, int nback, int nforw) {

  switch (type) {
  case CHALO_TYPE1:
    if (cart_size(dimension) > 1)
      CMPI_exchange_sum_type1(dimension, nforw, nback);
    else {
      memcpy(_halo_recv_one, _halo_send_one,
	     (nback + nforw)*_halo_message_size[type]);
    }
    break;
  case CHALO_TYPE2:
    if (cart_size(dimension) > 1)
      CMPI_exchange_sum_type2(dimension, nforw, nback);
    else {
      memcpy(_halo_recv_two, _halo_send_two,
	     (nback + nforw)*_halo_message_size[type]);
    }
    break;
  case CHALO_TYPE7:
    if (cart_size(dimension) > 1)
      CMPI_exchange_sum_type7(dimension, nforw, nback);
    else {
      memcpy(_halo_recv_sev, _halo_send_sev,
	     (nback + nforw)*_halo_message_size[type]);
    }
    break;
  case CHALO_TYPE8:
    if (cart_size(dimension) > 1){

      CMPI_exchange_sum_type8(dimension, nforw, nback);

    }else {

      memcpy(_halo_recv_eig, _halo_send_eig,
	     (nback + nforw)*_halo_message_size[type]);

    }
    break;

  default:
    fatal("internal error: Incorrect message type %d\n", type);
  }


  return;
}


/*****************************************************************************
 *
 *  CCOM_load_halo_buffer
 *
 *  Insert a particle into the halo message structure. Apply the
 *  periodic boundary conditions to the copy before it leaves.
 *
 *****************************************************************************/

void CCOM_load_halo_buffer(Colloid * p_colloid, int n,
			   const double rperiod[3]) {

  if (n >= _halo_message_nmax) fatal("_halo_send buffer too small (%d)\n", n);

  _halo_send[n].index     = p_colloid->s.index;
  _halo_send[n].a0        = p_colloid->s.a0;
  _halo_send[n].ah        = p_colloid->s.ah;
  _halo_send[n].r[X]       = p_colloid->s.r[X] + rperiod[X];
  _halo_send[n].r[Y]       = p_colloid->s.r[Y] + rperiod[Y];
  _halo_send[n].r[Z]       = p_colloid->s.r[Z] + rperiod[Z];
  _halo_send[n].v[X]       = p_colloid->s.v[X];
  _halo_send[n].v[Y]       = p_colloid->s.v[Y];
  _halo_send[n].v[Z]       = p_colloid->s.v[Z];
  _halo_send[n].omega[X]   = p_colloid->s.w[X];
  _halo_send[n].omega[Y]   = p_colloid->s.w[Y];
  _halo_send[n].omega[Z]   = p_colloid->s.w[Z];
  _halo_send[n].direction[X]     = p_colloid->s.m[X];
  _halo_send[n].direction[Y]     = p_colloid->s.m[Y];
  _halo_send[n].direction[Z]     = p_colloid->s.m[Z];
  _halo_send[n].b1        = p_colloid->s.b1;
  _halo_send[n].b2        = p_colloid->s.b2;

  _halo_send[n].s[X] = p_colloid->s.s[X];
  _halo_send[n].s[Y] = p_colloid->s.s[Y];
  _halo_send[n].s[Z] = p_colloid->s.s[Z];

  _halo_send[n].dr[X] = p_colloid->s.dr[X];
  _halo_send[n].dr[Y] = p_colloid->s.dr[Y];
  _halo_send[n].dr[Z] = p_colloid->s.dr[Z];

  _halo_send[n].cwetting = p_colloid->s.c;
  _halo_send[n].hwetting = p_colloid->s.h;

  _halo_send[n].random[0] = p_colloid->random[0];
  _halo_send[n].random[1] = p_colloid->random[1];
  _halo_send[n].random[2] = p_colloid->random[2];
  _halo_send[n].random[3] = p_colloid->random[3];
  _halo_send[n].random[4] = p_colloid->random[4];
  _halo_send[n].random[5] = p_colloid->random[5];

  return;
}


/*****************************************************************************
 *
 *  CCOM_unload_halo_buffer
 *
 *  Unpack a received particle from the halo message structure.
 *
 *****************************************************************************/

void CCOM_unload_halo_buffer(Colloid * p_colloid, int nrecv) {

  int ia;

  /* Check for stale particle data in the buffer */
  if (_halo_recv[nrecv].index < 0) {
    fatal("Unloaded stale particle (order %d)\n", nrecv);
  }

  p_colloid->s.index     = _halo_recv[nrecv].index;
  p_colloid->s.a0        = _halo_recv[nrecv].a0;
  p_colloid->s.ah        = _halo_recv[nrecv].ah;

  for (ia = 0; ia < 3; ia++) {
    p_colloid->s.r[ia]       = _halo_recv[nrecv].r[ia];
    p_colloid->s.v[ia]       = _halo_recv[nrecv].v[ia];
    p_colloid->s.w[ia]   = _halo_recv[nrecv].omega[ia];
  }

  p_colloid->s.m[X]     = _halo_recv[nrecv].direction[X];
  p_colloid->s.m[Y]     = _halo_recv[nrecv].direction[Y];
  p_colloid->s.m[Z]     = _halo_recv[nrecv].direction[Z];
  p_colloid->s.b1        = _halo_recv[nrecv].b1;
  p_colloid->s.b2        = _halo_recv[nrecv].b2;

  p_colloid->random[0] = _halo_recv[nrecv].random[0];
  p_colloid->random[1] = _halo_recv[nrecv].random[1];
  p_colloid->random[2] = _halo_recv[nrecv].random[2];
  p_colloid->random[3] = _halo_recv[nrecv].random[3];
  p_colloid->random[4] = _halo_recv[nrecv].random[4];
  p_colloid->random[5] = _halo_recv[nrecv].random[5];

  p_colloid->s.s[X] = _halo_recv[nrecv].s[X];
  p_colloid->s.s[Y] = _halo_recv[nrecv].s[Y];
  p_colloid->s.s[Z] = _halo_recv[nrecv].s[Z];

  p_colloid->s.dr[X] = _halo_recv[nrecv].dr[X];
  p_colloid->s.dr[Y] = _halo_recv[nrecv].dr[Y];
  p_colloid->s.dr[Z] = _halo_recv[nrecv].dr[Z];

  p_colloid->s.c = _halo_recv[nrecv].cwetting;
  p_colloid->s.h = _halo_recv[nrecv].hwetting;

  p_colloid->s.deltaphi= 0.0;

  /* Additionally, must set all accumulated quantities to zero. */
  p_colloid->s.rebuild = 1;

  for (ia = 0; ia < 3; ia++) {
    p_colloid->f0[ia] = 0.0;
    p_colloid->t0[ia] = 0.0;
    p_colloid->cbar[ia] = 0.0;
    p_colloid->rxcbar[ia] = 0.0;
    p_colloid->force[ia] = 0.0;
    p_colloid->torque[ia] = 0.0;
    p_colloid->fc0[ia] = 0.0;
    p_colloid->tc0[ia] = 0.0;
  }

  p_colloid->sump    = 0.0;
  p_colloid->sumw    = 0.0;
  p_colloid->deltam  = 0.0;

  p_colloid->lnk     = NULL;
  p_colloid->next    = NULL;

  return;
}


/*****************************************************************************
 *
 *  CCOM_load_sum_message_buffer
 *
 *  Load the appropriate information into the message buffer.
 *
 *****************************************************************************/

void CCOM_load_sum_message_buffer(Colloid * p_colloid, int n, int type) {

  int ia, iz;

  if (n >= _halo_message_nmax) fatal("_halo_send_one too small (%d)\n", n);

  switch (type) {
  case CHALO_TYPE1:
    _halo_send_one[n].index    = p_colloid->s.index;

    for (ia = 0; ia < 3; ia++) {
      _halo_send_one[n].cbar[ia]   = p_colloid->cbar[ia];
      _halo_send_one[n].rxcbar[ia] = p_colloid->rxcbar[ia];
      _halo_send_one[n].force[ia]  = p_colloid->force[ia];
      _halo_send_one[n].torque[ia] = p_colloid->torque[ia];
    }
    _halo_send_one[n].sumw     = p_colloid->sumw;
    _halo_send_one[n].deltam   = p_colloid->deltam;
    _halo_send_one[n].deltaphi = p_colloid->s.deltaphi;
    break;
  case CHALO_TYPE2:
    _halo_send_two[n].index = p_colloid->s.index;

    for (ia = 0; ia < 3; ia++) {
      _halo_send_two[n].f0[ia]  = p_colloid->f0[ia];
      _halo_send_two[n].t0[ia]  = p_colloid->t0[ia];
    }
    _halo_send_two[n].sump  = p_colloid->sump;
    for (iz = 0; iz < 21; iz++) {
      _halo_send_two[n].zeta[iz] = p_colloid->zeta[iz];
    }
    break;
  case CHALO_TYPE7:
    _halo_send_sev[n].index = p_colloid->s.index;

    for (ia = 0; ia < 3; ia++) {
      _halo_send_sev[n].f0[ia] = p_colloid->f0[ia];
    }
    break;
  case CHALO_TYPE8:
    _halo_send_eig[n].index = p_colloid->s.index;

    for (ia = 0; ia < 3; ia++) {
      _halo_send_eig[n].fc0[ia]  = p_colloid->fc0[ia];
      _halo_send_eig[n].tc0[ia]  = p_colloid->tc0[ia];
    }
    break;

  default:
    fatal("Incorrect message type\n");
  }

  return;
}


/*****************************************************************************
 *
 *  CCOM_unload_sum_message_buffer
 *
 *  Unpack colloid information from the message buffer and perform
 *  the sum by adding to the corresponding local particle.
 *
 *****************************************************************************/

void CCOM_unload_sum_message_buffer(Colloid * p_colloid, int n, int type) {

  int ia;
  int iz;

  switch (type) {
  case CHALO_TYPE1:
    if (p_colloid->s.index != _halo_recv_one[n].index) {
      verbose("Type one does not match (order %d) [expected %d got %d]\n",
	      n, p_colloid->s.index, _halo_recv_one[n].index);
      fatal("Stopping");
    }

    for (ia = 0; ia < 3; ia++) {
      p_colloid->cbar[ia]   += _halo_recv_one[n].cbar[ia];
      p_colloid->rxcbar[ia] += _halo_recv_one[n].rxcbar[ia];
      p_colloid->force[ia]  += _halo_recv_one[n].force[ia];
      p_colloid->torque[ia] += _halo_recv_one[n].torque[ia];
    }
    p_colloid->sumw     += _halo_recv_one[n].sumw;
    p_colloid->deltam   += _halo_recv_one[n].deltam;
    p_colloid->s.deltaphi += _halo_recv_one[n].deltaphi;
    break;

  case CHALO_TYPE2:
    if (p_colloid->s.index != _halo_recv_two[n].index) {
      verbose("Type two does not match (order %d) (expected %d got %d)\n",
	      n, p_colloid->s.index, _halo_recv_two[n].index);
      fatal("");
    }

    for (ia = 0; ia < 3; ia++) {
      p_colloid->f0[ia] += _halo_recv_two[n].f0[ia];
      p_colloid->t0[ia] += _halo_recv_two[n].t0[ia];
    }
    p_colloid->sump += _halo_recv_two[n].sump;
    for (iz = 0; iz < 21; iz++) {
      p_colloid->zeta[iz] += _halo_recv_two[n].zeta[iz];
    }
    break;

  case CHALO_TYPE7:
    if (p_colloid->s.index != _halo_recv_sev[n].index) {
      verbose("Type seven does not match (order %d) (expected %d got %d)\n",
	      n, p_colloid->s.index, _halo_recv_sev[n].index);
      fatal("");
    }

    for (ia = 0; ia < 3; ia++) {
      p_colloid->f0[ia] += _halo_recv_sev[n].f0[ia];
    }
    break;

  case CHALO_TYPE8:
    if (p_colloid->s.index != _halo_recv_eig[n].index) {
      verbose("Type eight does not match (order %d) (expected %d got %d)\n",
	      n, p_colloid->s.index, _halo_recv_eig[n].index);
      fatal("");
    }

    for (ia = 0; ia < 3; ia++) {
      p_colloid->fc0[ia] += _halo_recv_eig[n].fc0[ia];
      p_colloid->tc0[ia] += _halo_recv_eig[n].tc0[ia];
    }
    break;

  default:
    fatal("Internal error: invalid message type\n");
  }

  return;
}



/*****************************************************************************
 *
 *  CMPI_init_messages
 *
 *  Initisalise the MPI message types for colloid data.
 *
 *****************************************************************************/

void CMPI_init_messages() {

  MPI_Type_contiguous(sizeof(Colloid_halo_message),    MPI_BYTE, &_mpi_halo);
  MPI_Type_contiguous(sizeof(Colloid_sum_message_one), MPI_BYTE, &_mpi_sum1);
  MPI_Type_contiguous(sizeof(Colloid_sum_message_two), MPI_BYTE, &_mpi_sum2);
  MPI_Type_contiguous(sizeof(Colloid_sum_message_sev), MPI_BYTE, &_mpi_sum7);
  MPI_Type_contiguous(sizeof(Colloid_sum_message_eig), MPI_BYTE, &_mpi_sum8);
  MPI_Type_commit(&_mpi_halo);
  MPI_Type_commit(&_mpi_sum1);
  MPI_Type_commit(&_mpi_sum2);
  MPI_Type_commit(&_mpi_sum7);
  MPI_Type_commit(&_mpi_sum8);

  return;
}


/*****************************************************************************
 *
 *  CMPI_exchange_halo
 *
 *  Exchange particle information between halo regions. The number of
 *  particles is not known in advance, so must be communicated first.
 *
 *  Could be a job for MPI single-sided?
 *
 *  The total number of particles received by this process is returned.
 *
 *****************************************************************************/

int CMPI_exchange_halo(int dimension, int nforw, int nback) {

  int nrecv = 0;

  if (cart_size(dimension) == 1) {
    nrecv = nforw + nback;
    memcpy(_halo_recv, _halo_send, nrecv*sizeof(Colloid_halo_message));
  }
  else {

    int         nrecv_forw, nrecv_back;
    const int   tag = 1000;
    MPI_Request request[4];
    MPI_Status  status[4];

    /* First, have to send the number particles expexted */

    MPI_Issend(&nforw, 1, MPI_INT, cart_neighb(FORWARD,dimension), tag,
	       cart_comm(), &request[0]);
    MPI_Issend(&nback, 1, MPI_INT, cart_neighb(BACKWARD,dimension), tag,
	       cart_comm(), &request[1]);

    MPI_Irecv(&nrecv_forw, 1, MPI_INT, cart_neighb(FORWARD,dimension), tag,
	      cart_comm(), &request[2]);
    MPI_Irecv(&nrecv_back, 1, MPI_INT, cart_neighb(BACKWARD,dimension), tag,
	      cart_comm(), &request[3]);

    MPI_Waitall(4, request, status);

    /* OK, now transfer the data */

    if (nforw)
      MPI_Issend(_halo_send + nback, nforw, _mpi_halo,
		 cart_neighb(FORWARD,dimension), tag, cart_comm(), &request[0]);
    if (nback)
      MPI_Issend(_halo_send, nback, _mpi_halo,
		 cart_neighb(BACKWARD,dimension), tag, cart_comm(), &request[1]);
    if (nrecv_forw)
      MPI_Irecv(_halo_recv, nrecv_forw, _mpi_halo,
		cart_neighb(FORWARD,dimension), tag,cart_comm(), &request[2]);
    if (nrecv_back)
      MPI_Irecv(_halo_recv + nrecv_forw, nrecv_back, _mpi_halo,
		cart_neighb(BACKWARD,dimension), tag, cart_comm(), &request[3]);

    MPI_Waitall(4, request, status);

    nrecv = nrecv_forw + nrecv_back;

  }

  return nrecv;
}


/*****************************************************************************
 *
 *  CMPI_exchange_sum_type1
 *
 *  Exchange data for partial colloid sums. Both send and receive
 *  sides of the communication shlould at this point already
 *  agree on the number of particles in th halo region (or
 *  there'll be trouble).
 *
 *****************************************************************************/

void CMPI_exchange_sum_type1(int dimension, int nforw, int nback) {

  const int   tagf = 1001, tagb = 1002;
  MPI_Request requests[4];
  MPI_Status  status[4];
  int         nr = 0;

  if (nback) {
    MPI_Issend(_halo_send_one, nback, _mpi_sum1,
	       cart_neighb(BACKWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_one + nforw, nback, _mpi_sum1,
	       cart_neighb(BACKWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
  }

  if (nforw) {
    MPI_Issend(_halo_send_one + nback, nforw, _mpi_sum1,
	       cart_neighb(FORWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_one, nforw, _mpi_sum1,
	       cart_neighb(FORWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
  }

  MPI_Waitall(nr, requests, status);

  return;
}


/*****************************************************************************
 *
 *  CMPI_exchange_sum_type2
 *
 *  This is exactly the same as the above, except for type two
 *  messages.
 *
 *****************************************************************************/

void CMPI_exchange_sum_type2(int dimension, int nforw, int nback) {

  const int   tagf = 1001, tagb = 1002;
  MPI_Request requests[4];
  MPI_Status  status[4];
  int         nr = 0;

  if (nback) {
    MPI_Issend(_halo_send_two, nback, _mpi_sum2,
	       cart_neighb(BACKWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_two + nforw, nback, _mpi_sum2,
	       cart_neighb(BACKWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
  }

  if (nforw) {
    MPI_Issend(_halo_send_two + nback, nforw, _mpi_sum2,
	       cart_neighb(FORWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_two, nforw, _mpi_sum2,
	       cart_neighb(FORWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
  }

  MPI_Waitall(nr, requests, status);

  return;
}

/*****************************************************************************
 *
 *  CMPI_exchange_sum_type7
 *
 *  This is exactly the same as the above, except for type seven
 *  messages.
 *
 *****************************************************************************/

void CMPI_exchange_sum_type7(int dimension, int nforw, int nback) {

  const int   tagf = 1001, tagb = 1002;
  MPI_Request requests[4];
  MPI_Status  status[4];
  int         nr = 0;

  if (nback) {
    MPI_Issend(_halo_send_sev, nback, _mpi_sum7,
	       cart_neighb(BACKWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_sev + nforw, nback, _mpi_sum7,
	       cart_neighb(BACKWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
  }

  if (nforw) {
    MPI_Issend(_halo_send_sev + nback, nforw, _mpi_sum7,
	       cart_neighb(FORWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_sev, nforw, _mpi_sum7,
	       cart_neighb(FORWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
  }

  MPI_Waitall(nr, requests, status);

  return;
}

/*****************************************************************************
 *
 *  CMPI_exchange_sum_type8
 *
 *  This is exactly the same as the above, except for type eight
 *  messages.
 *
 *****************************************************************************/

void CMPI_exchange_sum_type8(int dimension, int nforw, int nback) {

  const int   tagf = 1001, tagb = 1002;
  MPI_Request requests[4];
  MPI_Status  status[4];
  int         nr = 0;

  if (nback) {
    MPI_Issend(_halo_send_eig, nback, _mpi_sum8,
	       cart_neighb(BACKWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_eig + nforw, nback, _mpi_sum8,
	       cart_neighb(BACKWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
  }

  if (nforw) {
    MPI_Issend(_halo_send_eig + nback, nforw, _mpi_sum8,
	       cart_neighb(FORWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_eig, nforw, _mpi_sum8,
	       cart_neighb(FORWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
  }

  MPI_Waitall(nr, requests, status);

  return;
}


/*****************************************************************************
 *
 *  CMPI_anull_buffers
 *
 *  Set the index of associated with all particle messages to -1.
 *
 *****************************************************************************/


void CMPI_anull_buffers() {

  int n;

  for (n = 0; n < _halo_message_nmax; n++) {
    _halo_send[n].index = -1;
    _halo_recv[n].index = -1;
    _halo_send_one[n].index = -1;
    _halo_recv_one[n].index = -1;
    _halo_send_two[n].index = -1;
    _halo_recv_two[n].index = -1;
    _halo_send_eig[n].index = -1;
    _halo_recv_eig[n].index = -1;

  }

  return;
}

/*****************************************************************************
 *
 *  colloid_min_ah
 *
 *  Return the smallest ah present along particles.
 *
 *****************************************************************************/

double colloid_min_ah(void) {

  int ic, jc, kc;
  double ahminlocal = FLT_MAX;
  double ahmin;
  Colloid * p_colloid;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  ahminlocal = dmin(ahminlocal, p_colloid->s.ah);
	  p_colloid = p_colloid->next;
	}

      }
    }
  }

  MPI_Allreduce(&ahminlocal, &ahmin, 1, MPI_DOUBLE, MPI_MIN, cart_comm());

  return ahmin;
}
