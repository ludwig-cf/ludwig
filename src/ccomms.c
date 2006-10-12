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
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "ccomms.h"

#ifdef _MPI_

static MPI_Datatype _mpi_halo;
static MPI_Datatype _mpi_sum1;
static MPI_Datatype _mpi_sum2;
static MPI_Datatype _mpi_sum6;

#endif

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
typedef struct colloid_sum_message_type6 Colloid_sum_message_six;

struct colloid_halo_message {
  int      index;
  double   a0;
  double   ah;
  FVector  r;
  FVector  v;
  FVector  omega;
  FVector  dir;
  double   dp;
  double   cosine_ca;
  double   random[6];
};

struct colloid_sum_message_type1 {
  int     index;
  FVector cbar;
  FVector rxcbar;
  double   sumw;
  FVector fex;
  FVector tex;
  double   deltam;
  double   deltaphi;
};

struct colloid_sum_message_type2 {
  int     index;
  FVector f0;
  FVector t0;
  double   zeta[21];
};

struct colloid_sum_message_type6 {
  int index;
  int n1_nodes;
  int n2_nodes;
};

static void CCOM_load_halo_buffer(Colloid *, int, FVector);
static void CCOM_unload_halo_buffer(Colloid *, int);
static void CCOM_load_sum_message_buffer(Colloid *, int, int);
static void CCOM_unload_sum_message_buffer(Colloid *, int, int);
static void CCOM_exchange_halo_sum(int, int, int, int);

static void CMPI_accept_new(int);
static int  CMPI_exchange_halo(int, int, int);
static void CMPI_exchange_sum_type1(int, int, int);
static void CMPI_exchange_sum_type2(int, int, int);
static void CMPI_exchange_sum_type6(int, int, int);
static void CMPI_anull_buffers(void);

static int                       _halo_message_nmax;
static int                       _halo_message_size[3];

static Colloid_halo_message    * _halo_send;
static Colloid_halo_message    * _halo_recv;
static Colloid_sum_message_one * _halo_send_one;
static Colloid_sum_message_one * _halo_recv_one;
static Colloid_sum_message_two * _halo_send_two;
static Colloid_sum_message_two * _halo_recv_two;
static Colloid_sum_message_six * _halo_send_six;
static Colloid_sum_message_six * _halo_recv_six;


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

  double vcell;
  int    ncell;

  /* Work out the volume of each halo region and take the
   * largest. Then work out how many particles will fit
   * in the volume. */

  vcell = L(X)*L(Y)*L(Z)/(Ncell(X)*Ncell(Y)*Ncell(Z));
  ncell = imax(Ncell(X), Ncell(Y));
  ncell = imax(Ncell(Z), ncell);
  ncell += 2; /* Add halo cells */

  /* Computing this number is a problem */
  _halo_message_nmax = ncell*ncell*vcell / (pow(dmax(1.0,2.3), 3));

  info("\nCCOM_init\n");
  info("Allocating space for %d particles in halo buffers\n",
       _halo_message_nmax);

  _halo_message_size[CHALO_TYPE1] = sizeof(Colloid_sum_message_one);
  _halo_message_size[CHALO_TYPE2] = sizeof(Colloid_sum_message_two);
  _halo_message_size[CHALO_TYPE6] = sizeof(Colloid_sum_message_six);

  info("Requesting %d bytes for halo messages\n",
       2*_halo_message_nmax*sizeof(Colloid_halo_message));
  info("Requesting %d bytes for type one messages\n",
       2*_halo_message_nmax*_halo_message_size[CHALO_TYPE1]);
  info("Requesting %d bytes for type two messages\n",
       2*_halo_message_nmax*_halo_message_size[CHALO_TYPE2]);
  info("Requesting %d bytes for type six messages\n",
       2*_halo_message_nmax*_halo_message_size[CHALO_TYPE6]);

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
  _halo_send_six = (Colloid_sum_message_six *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_six));
  _halo_recv_six = (Colloid_sum_message_six *)
    calloc(_halo_message_nmax, sizeof(Colloid_sum_message_six));


  if (_halo_send      == NULL) fatal("_halo_send failed");
  if (_halo_recv      == NULL) fatal("_halo_recv failed");
  if (_halo_send_one  == NULL) fatal("_halo_send_one failed");
  if (_halo_recv_one  == NULL) fatal("_halo_recv_one failed");
  if (_halo_send_two  == NULL) fatal("_halo_send_two failed");
  if (_halo_recv_two  == NULL) fatal("_halo_recv_two failed");
  if (_halo_send_six  == NULL) fatal("_halo_send_six failed");
  if (_halo_recv_six  == NULL) fatal("_halo_recv_six failed");

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
  FVector     rperiod;


  /* Non-periodic system requires no halo exchanges */
  if (is_periodic(X) == 0) return;

  /* This is for testing purposes: it sets all the indexes in the
   * communication buffers to -1 in order to pick up attempted
   * access to stale information. It does little harm to keep. */

  CMPI_anull_buffers();


  /* Periodic boundary conditions are slightly clunky at the moment,
   * so could be improved */

  rperiod = UTIL_fvector_zero();

  /* x-direction */

  nforw = 0;
  nback = 0;

  /* Backward x direction. */

  for (jc = 1; jc <= Ncell(Y); jc++) {
    for (kc = 1; kc <= Ncell(Z); kc++) {

      ic = 1;
      p_colloid = CELL_get_head_of_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(X) == 0) rperiod.x = L(X);
	CCOM_load_halo_buffer(p_colloid, nback, rperiod);
	rperiod.x = 0.0;
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
      p_colloid = CELL_get_head_of_list(ic, jc, kc);

      while (p_colloid) {

	VERBOSE(("XFORWARD: [%d] has export = %d\n", p_colloid->index,
		p_colloid->export));

	if (cart_coords(X) == cart_size(X)-1) rperiod.x = -L(X);
	CCOM_load_halo_buffer(p_colloid, nback + nforw, rperiod);
	rperiod.x = 0.0;
	++nforw;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }
    }
  }

  VERBOSE(("Halo x-send forward: %d back: %d\n", nforw, nback));
  nrecv = CMPI_exchange_halo(X, nforw, nback);
  CMPI_accept_new(nrecv);

  if (is_periodic(Y) == 0) return;

  /* Backward y direction. */

  nforw = 0;
  nback = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (kc = 1; kc <= Ncell(Z)    ; kc++) {

      jc = 1;
      p_colloid = CELL_get_head_of_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Y) == 0) rperiod.y = +L(Y);
	CCOM_load_halo_buffer(p_colloid, nback, rperiod);
	rperiod.y = 0.0;
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
      p_colloid = CELL_get_head_of_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Y) == cart_size(Y)-1) rperiod.y = -L(Y);
	CCOM_load_halo_buffer(p_colloid, nback + nforw, rperiod);
	rperiod.y = 0.0;
	++nforw;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }

    }
  }

  VERBOSE(("Halo y-send forward: %d back: %d\n", nforw, nback));
  nrecv = CMPI_exchange_halo(Y, nforw, nback);
  CMPI_accept_new(nrecv);


  if (is_periodic(Z) == 0) return;

  /* Backward z-direction. */

  nforw = 0;
  nback = 0;

  for (ic = 0; ic <= Ncell(X) + 1; ic++) {
    for (jc = 0; jc <= Ncell(Y) + 1; jc++) {

      kc = 1;
      p_colloid = CELL_get_head_of_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Z) == 0) rperiod.z = +L(Z);
	CCOM_load_halo_buffer(p_colloid, nback, rperiod);
	rperiod.z = 0.0;
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
      p_colloid = CELL_get_head_of_list(ic, jc, kc);

      while (p_colloid) {

	if (cart_coords(Z) == cart_size(Z)-1) rperiod.z = -L(Z);
	CCOM_load_halo_buffer(p_colloid, nback + nforw, rperiod);
	rperiod.z = 0.0;
	++nforw;

	/* Next colloid */
	p_colloid = p_colloid->next;
      }
    }
  }

  VERBOSE(("Halo z-send forward: %d back: %d\n", nforw, nback));
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
 *
 *****************************************************************************/

void CMPI_accept_new(int nrecv) {

  Colloid * p_colloid;
  Colloid * p_existing;
  IVector   new;
  int       exists;
  int       n;

  for (n = 0; n < nrecv; n++) {

    /* Load the new particle */
    p_colloid = allocate_colloid();
    CCOM_unload_halo_buffer(p_colloid, n);

    /* See where it wants to go in the cell list */

    new = cell_coords(p_colloid->r);

    /* Have a look at the existing particles in the list location:
     * if it's already present, just flag the existing particle so
     * it gets copied in the next direction; if it's new, the incoming
     * particle does get added to the cell list. */

    exists = 0;
    p_existing = CELL_get_head_of_list(new.x, new.y, new.z);

    while (p_existing) {
      if (p_colloid->index == p_existing->index) {
	/* Just copy the state information */
	p_existing->r.x = p_colloid->r.x;
	p_existing->r.y = p_colloid->r.y;
	p_existing->r.z = p_colloid->r.z;
	p_existing->v.x = p_colloid->v.x;
	p_existing->v.y = p_colloid->v.y;
	p_existing->v.z = p_colloid->v.z;
	p_existing->omega.x = p_colloid->omega.x;
	p_existing->omega.y = p_colloid->omega.y;
	p_existing->omega.z = p_colloid->omega.z;
	p_existing->random[0] = p_colloid->random[0];
	p_existing->random[1] = p_colloid->random[1];
	p_existing->random[2] = p_colloid->random[2];
	p_existing->random[3] = p_colloid->random[3];
	p_existing->random[4] = p_colloid->random[4];
	p_existing->random[5] = p_colloid->random[5];
	exists = 1;
      }
      p_existing = p_existing->next;
    }

    if (exists) {
      /* Just drop the incoming copy */
      VERBOSE(("Dropped copy of [index %d] at [%d,%d,%d]\n", p_colloid->index,
	       new.x, new.y, new.z));
      free_colloid(p_colloid);
    }
    else {
      /* Add the incoming colloid */
      cell_insert_colloid(p_colloid);
      VERBOSE(("Added copy of [index %d] to [%d,%d,%d]\n", p_colloid->index,
	       new.x, new.y, new.z));
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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

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
  case CHALO_TYPE6:
    if (cart_size(dimension) > 1)
      CMPI_exchange_sum_type6(dimension, nforw, nback);
    else {
      memcpy(_halo_recv_six, _halo_send_six,
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

void CCOM_load_halo_buffer(Colloid * p_colloid, int n, FVector rperiod) {

  if (n >= _halo_message_nmax) fatal("_halo_send buffer too small\n");

  VERBOSE(("Loading particle (index %d) (order = %d)\n", p_colloid->index, n));

  _halo_send[n].index     = p_colloid->index;
  _halo_send[n].a0        = p_colloid->a0;
  _halo_send[n].ah        = p_colloid->ah;
  _halo_send[n].r.x       = p_colloid->r.x + rperiod.x;
  _halo_send[n].r.y       = p_colloid->r.y + rperiod.y;
  _halo_send[n].r.z       = p_colloid->r.z + rperiod.z;
  _halo_send[n].v.x       = p_colloid->v.x;
  _halo_send[n].v.y       = p_colloid->v.y;
  _halo_send[n].v.z       = p_colloid->v.z;
  _halo_send[n].omega.x   = p_colloid->omega.x;
  _halo_send[n].omega.y   = p_colloid->omega.y;
  _halo_send[n].omega.z   = p_colloid->omega.z;
  _halo_send[n].dir.x     = p_colloid->dir.x;
  _halo_send[n].dir.y     = p_colloid->dir.y;
  _halo_send[n].dir.z     = p_colloid->dir.z;
  _halo_send[n].dp        = p_colloid->dp;
  _halo_send[n].cosine_ca = p_colloid->cosine_ca;

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

  /* Check for stale particle data in the buffer */
  if (_halo_recv[nrecv].index < 0) {
    fatal("Unloaded stale particle (order %d)\n", nrecv);
  }

  p_colloid->index     = _halo_recv[nrecv].index;
  p_colloid->a0        = _halo_recv[nrecv].a0;
  p_colloid->ah        = _halo_recv[nrecv].ah;
  p_colloid->r.x       = _halo_recv[nrecv].r.x;
  p_colloid->r.y       = _halo_recv[nrecv].r.y;
  p_colloid->r.z       = _halo_recv[nrecv].r.z;
  p_colloid->v.x       = _halo_recv[nrecv].v.x;
  p_colloid->v.y       = _halo_recv[nrecv].v.y;
  p_colloid->v.z       = _halo_recv[nrecv].v.z;
  p_colloid->omega.x   = _halo_recv[nrecv].omega.x;
  p_colloid->omega.y   = _halo_recv[nrecv].omega.y;
  p_colloid->omega.z   = _halo_recv[nrecv].omega.z;
  p_colloid->dir.x     = _halo_recv[nrecv].dir.x;
  p_colloid->dir.y     = _halo_recv[nrecv].dir.y;
  p_colloid->dir.z     = _halo_recv[nrecv].dir.z;
  p_colloid->dp        = _halo_recv[nrecv].dp;
  p_colloid->cosine_ca = _halo_recv[nrecv].cosine_ca;

  p_colloid->random[0] = _halo_recv[nrecv].random[0];
  p_colloid->random[1] = _halo_recv[nrecv].random[1];
  p_colloid->random[2] = _halo_recv[nrecv].random[2];
  p_colloid->random[3] = _halo_recv[nrecv].random[3];
  p_colloid->random[4] = _halo_recv[nrecv].random[4];
  p_colloid->random[5] = _halo_recv[nrecv].random[5];

  /* Additionally, must set all accumulated quantities to zero. */
  p_colloid->rebuild = 1;
  p_colloid->f0      = UTIL_fvector_zero();
  p_colloid->t0      = UTIL_fvector_zero();
  p_colloid->force   = UTIL_fvector_zero();
  p_colloid->torque  = UTIL_fvector_zero();
  p_colloid->cbar    = UTIL_fvector_zero();
  p_colloid->rxcbar  = UTIL_fvector_zero();
  p_colloid->sumw    = 0.0;
  p_colloid->deltam  = 0.0;
  p_colloid->deltaphi= 0.0;
  p_colloid->n1_nodes= 0;
  p_colloid->n2_nodes= 0;
  p_colloid->lnk     = NULL;
  p_colloid->next    = NULL;

  VERBOSE(("Unloaded particle (index %d) (order = %d)\n", p_colloid->index,
	  nrecv));

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

  int iz;

  if (n >= _halo_message_nmax) fatal("_halo_send_one too small\n");

#ifdef _VERY_VERBOSE_
  VERBOSE(("Loading sum message type%d [index %d] (order %d)\n", type,
	  p_colloid->index, n));
#endif

  switch (type) {
  case CHALO_TYPE1:
    _halo_send_one[n].index    = p_colloid->index;
    _halo_send_one[n].cbar.x   = p_colloid->cbar.x;
    _halo_send_one[n].cbar.y   = p_colloid->cbar.y;
    _halo_send_one[n].cbar.z   = p_colloid->cbar.z;
    _halo_send_one[n].rxcbar.x = p_colloid->rxcbar.x;
    _halo_send_one[n].rxcbar.y = p_colloid->rxcbar.y;
    _halo_send_one[n].rxcbar.z = p_colloid->rxcbar.z;
    _halo_send_one[n].sumw     = p_colloid->sumw;
    _halo_send_one[n].fex.x    = p_colloid->force.x;
    _halo_send_one[n].fex.y    = p_colloid->force.y;
    _halo_send_one[n].fex.z    = p_colloid->force.z;
    _halo_send_one[n].tex.x    = p_colloid->torque.x;
    _halo_send_one[n].tex.y    = p_colloid->torque.y;
    _halo_send_one[n].tex.z    = p_colloid->torque.z;
    _halo_send_one[n].deltam   = p_colloid->deltam;
    _halo_send_one[n].deltaphi = p_colloid->deltaphi;
    break;
  case CHALO_TYPE2:
    _halo_send_two[n].index = p_colloid->index;
    _halo_send_two[n].f0.x  = p_colloid->f0.x;
    _halo_send_two[n].f0.y  = p_colloid->f0.y;
    _halo_send_two[n].f0.z  = p_colloid->f0.z;
    _halo_send_two[n].t0.x  = p_colloid->t0.x;
    _halo_send_two[n].t0.y  = p_colloid->t0.y;
    _halo_send_two[n].t0.z  = p_colloid->t0.z;
    for (iz = 0; iz < 21; iz++) {
      _halo_send_two[n].zeta[iz] = p_colloid->zeta[iz];
    }
    break;
  case CHALO_TYPE6:
    _halo_send_six[n].index    = p_colloid->index;
    _halo_send_six[n].n1_nodes = p_colloid->n1_nodes;
    _halo_send_six[n].n2_nodes = p_colloid->n2_nodes;
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

  int iz;

  switch (type) {
  case CHALO_TYPE1:

    if (p_colloid->index != _halo_recv_one[n].index) {
      verbose("Type one does not match (order %d) [expected %d got %d]\n",
	      n, p_colloid->index, _halo_recv_one[n].index);
      fatal("");
    }

    p_colloid->cbar.x   += _halo_recv_one[n].cbar.x;
    p_colloid->cbar.y   += _halo_recv_one[n].cbar.y;
    p_colloid->cbar.z   += _halo_recv_one[n].cbar.z;
    p_colloid->rxcbar.x += _halo_recv_one[n].rxcbar.x;
    p_colloid->rxcbar.y += _halo_recv_one[n].rxcbar.y;
    p_colloid->rxcbar.z += _halo_recv_one[n].rxcbar.z;
    p_colloid->sumw     += _halo_recv_one[n].sumw;
    p_colloid->force.x  += _halo_recv_one[n].fex.x;
    p_colloid->force.y  += _halo_recv_one[n].fex.y;
    p_colloid->force.z  += _halo_recv_one[n].fex.z;
    p_colloid->torque.x += _halo_recv_one[n].tex.x;
    p_colloid->torque.y += _halo_recv_one[n].tex.y;
    p_colloid->torque.z += _halo_recv_one[n].tex.z;
    p_colloid->deltam   += _halo_recv_one[n].deltam;
    p_colloid->deltaphi += _halo_recv_one[n].deltaphi;
    break;

  case CHALO_TYPE2:

    if (p_colloid->index != _halo_recv_two[n].index) {
      verbose("Type two does not match (order %d) (expected %d got %d)\n",
	      n, p_colloid->index, _halo_recv_two[n].index);
      fatal("");
    }

    p_colloid->f0.x += _halo_recv_two[n].f0.x;
    p_colloid->f0.y += _halo_recv_two[n].f0.y;
    p_colloid->f0.z += _halo_recv_two[n].f0.z;
    p_colloid->t0.x += _halo_recv_two[n].t0.x;
    p_colloid->t0.y += _halo_recv_two[n].t0.y;
    p_colloid->t0.z += _halo_recv_two[n].t0.z;
    for (iz = 0; iz < 21; iz++) {
      p_colloid->zeta[iz] += _halo_recv_two[n].zeta[iz];
    }
    break;

  case CHALO_TYPE6:

    if (p_colloid->index != _halo_recv_six[n].index) {
      verbose("Type six does not match (order %d) (expected %d got %d)\n",
	      n, p_colloid->index, _halo_recv_six[n].index);
      fatal("");
    }

    p_colloid->n1_nodes += _halo_recv_six[n].n1_nodes;
    p_colloid->n2_nodes += _halo_recv_six[n].n2_nodes;
    break;

  default:
    fatal("Internal error: invalid message type\n");
  }

#ifdef _VERY_VERBOSE_
  VERBOSE(("Unloaded sum message type%d (order = %d) [index %d]\n", type,
	   n, p_colloid->index));
#endif

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

#ifdef _MPI_

  MPI_Type_contiguous(sizeof(Colloid_halo_message),    MPI_BYTE, &_mpi_halo);
  MPI_Type_contiguous(sizeof(Colloid_sum_message_one), MPI_BYTE, &_mpi_sum1);
  MPI_Type_contiguous(sizeof(Colloid_sum_message_two), MPI_BYTE, &_mpi_sum2);
  MPI_Type_contiguous(sizeof(Colloid_sum_message_six), MPI_BYTE, &_mpi_sum6);
  MPI_Type_commit(&_mpi_halo);
  MPI_Type_commit(&_mpi_sum1);
  MPI_Type_commit(&_mpi_sum2);
  MPI_Type_commit(&_mpi_sum6);

#endif

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

#ifdef _MPI_
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
    VERBOSE(("Sent: (%d to %d, %d to %d) Expecting: (%d, %d)\n",
	    nforw, edge[FORWARD][dimension], nback,
	     cart_neighb(BACKWARD,dimension), nrecv_forw, nrecv_back));

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
#endif

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

#ifdef _MPI_

  const int   tagf = 1001, tagb = 1002;
  MPI_Request requests[4];
  MPI_Status  status[4];
  int         nr = 0;

  VERBOSE(("Sum 1 (direction %d) Sending %d and %d\n",
	   dimension, nforw, nback));

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
  VERBOSE(("Sum 1 waitall returned\n"));

#endif

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

#ifdef _MPI_

  const int   tagf = 1001, tagb = 1002;
  MPI_Request requests[4];
  MPI_Status  status[4];
  int         nr = 0;

  VERBOSE(("Sum 2 (direction %d) Sending %d and %d\n",
	   dimension, nforw, nback));

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
  VERBOSE(("Sum 2 waitall returned\n"));
#endif

  return;
}


/*****************************************************************************
 *
 *  CMPI_exchange_sum_type6
 *
 *  This is exactly the same as the above, except for type six
 *  messages.
 *
 *****************************************************************************/

void CMPI_exchange_sum_type6(int dimension, int nforw, int nback) {

#ifdef _MPI_

  const int   tagf = 1001, tagb = 1002;
  MPI_Request requests[4];
  MPI_Status  status[4];
  int         nr = 0;

  VERBOSE(("Sum 6 (direction %d) Sending %d and %d\n",
	   dimension, nforw, nback));

  if (nback) {
    MPI_Issend(_halo_send_six, nback, _mpi_sum6,
	       cart_neighb(BACKWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_six + nforw, nback, _mpi_sum6,
	       cart_neighb(BACKWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
  }

  if (nforw) {
    MPI_Issend(_halo_send_six + nback, nforw, _mpi_sum6,
	       cart_neighb(FORWARD,dimension), tagf, cart_comm(),
	       &requests[nr++]);
    MPI_Irecv (_halo_recv_six, nforw, _mpi_sum6,
	       cart_neighb(FORWARD,dimension), tagb, cart_comm(),
	       &requests[nr++]);
  }

  MPI_Waitall(nr, requests, status);
  VERBOSE(("Sum 6 waitall returned\n"));
#endif

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
  }

  return;
}
