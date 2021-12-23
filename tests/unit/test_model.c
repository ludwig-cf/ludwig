/*****************************************************************************
 *
 *  test_model.c
 *
 *  Unit test for the currently compiled model (D3Q15 or D3Q19).
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "util.h"
#include "cs_limits.h"
#include "lb_model_s.h"
#include "tests.h"

static void test_model_constants(void);
static void test_model_velocity_set(void);

int do_test_model_distributions(pe_t * pe, cs_t * cs);
int do_test_model_halo_swap(pe_t * pe, cs_t * cs);
int do_test_model_reduced_halo_swap(pe_t * pe, cs_t * cs);
int do_test_lb_model_io(pe_t * pe, cs_t * cs);
static  int test_model_is_domain(cs_t * cs, int ic, int jc, int kc);





typedef struct lb_data_options_s {
  int ndim;
  int nvel;
  int ndist;
} lb_data_options_t;


/* Utility to return a unique value for global (ic,jc,kc,p) */
/* This allows e.g., tests to check distribution values in parallel
 * exchanges. */

/* (ic, jc, kc) are local indices */
/* Result could be unsigned integer... */

#include <stdint.h>

int64_t lb_data_index(lb_t * lb, int ic, int jc, int kc, int p) {

  int64_t index = INT64_MIN;
  int64_t nall[3] = {};
  int64_t nstr[3] = {};
  int64_t pstr    = 0;

  int ntotal[3] = {};
  int offset[3] = {};
  int nhalo = 0;

  assert(lb);
  assert(0 <= p && p < lb->model.nvel);

  cs_ntotal(lb->cs, ntotal);
  cs_nlocal_offset(lb->cs, offset);
  cs_nhalo(lb->cs, &nhalo);

  nall[X] = ntotal[X] + 2*nhalo;
  nall[Y] = ntotal[Y] + 2*nhalo;
  nall[Z] = ntotal[Z] + 2*nhalo;
  nstr[Z] = 1;
  nstr[Y] = nstr[Z]*nall[Z];
  nstr[X] = nstr[Y]*nall[Y];
  pstr    = nstr[X]*nall[X];

  {
    int igl = offset[X] + ic;
    int jgl = offset[Y] + jc;
    int kgl = offset[Z] + kc;

    /* A periodic system */
    igl = igl % ntotal[X];
    jgl = jgl % ntotal[Y];
    kgl = kgl % ntotal[Z];
    if (igl < 1) igl = igl + ntotal[X];
    if (jgl < 1) jgl = jgl + ntotal[Y];
    if (kgl < 1) kgl = kgl + ntotal[Z];

    assert(1 <= igl && igl <= ntotal[X]);
    assert(1 <= jgl && jgl <= ntotal[Y]);
    assert(1 <= kgl && kgl <= ntotal[Z]);

    index = pstr*p + nstr[X]*igl + nstr[Y]*jgl + nstr[Z]*kgl;
  }

  return index;
}

int lb_data_create(pe_t * pe, cs_t * cs, const lb_data_options_t * options,
		   lb_t ** lb);

int lb_data_create(pe_t * pe, cs_t * cs, const lb_data_options_t * options,
		   lb_t ** lb) {

  lb_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(lb);

  obj = (lb_t *) calloc(1, sizeof(lb_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(1, lb_t) failed\n");

  /* Check options */

  obj->pe = pe;
  obj->cs = cs;
  obj->ndim = options->ndim;
  obj->nvel = options->nvel;
  obj->ndist = options->ndist;

  lb_model_create(obj->nvel, &obj->model);

  /* Storage */

  {
    /* Allocate storage following cs specification */
    int nhalo = 1;
    int nlocal[3] = {};
    cs_nhalo(cs, &nhalo);
    cs_nlocal(cs, nlocal);

    {
      int nx = nlocal[X] + 2*nhalo;
      int ny = nlocal[Y] + 2*nhalo;
      int nz = nlocal[Z] + 2*nhalo;
      obj->nsite = nx*ny*nz;
    }
    {
      size_t sz = sizeof(double)*obj->nsite*obj->nvel;
      assert(sz > 0); /* Should not overflow in size_t I hope! */
      obj->f = (double *) mem_aligned_malloc(MEM_PAGESIZE, sz);
      assert(obj->f);
      if (obj->f == NULL) pe_fatal(pe, "malloc(lb->f) failed\n");
    }
  }
  
  *lb = obj;

  return 0;
}

int lb_data_free(lb_t * lb) {

  assert(lb);

  free(lb->f);
  lb_model_free(&lb->model);
  free(lb);

  return 0;
}

/* We will not exceed 27 directions! Direction index 0, in keeping
 * with the LB model definition, is (0,0,0) - so no communication. */

typedef struct lb_halo_s {

  MPI_Comm comm;                  /* coords: Cartesian communicator */
  int nbrrank[3][3][3];           /* coords: neighbour rank look-up */
  int nlocal[3];                  /* coords: local domain size */

  lb_model_t map;                 /* Communication map 2d or 3d */
  int tagbase;                    /* send/recv tag */
  int full;                       /* All velocities at each site required. */
  int count[27];                  /* halo: item data count per direction */
  cs_limits_t slim[27];           /* halo: send data region (rectangular) */
  cs_limits_t rlim[27];           /* halo: recv data region (rectangular) */
  double * send[27];              /* halo: send buffer per direction */
  double * recv[27];              /* halo: recv buffer per direction */
  MPI_Request request[2*27];      /* halo: array of requests */

} lb_halo_t;

/*****************************************************************************
 *
 *  lb_halo_size
 *
 *  Utility to compute a number of sites from cs_limits_t.
 *
 *****************************************************************************/

int cs_limits_size(cs_limits_t lim) {

  int szx = 1 + lim.imax - lim.imin;
  int szy = 1 + lim.jmax - lim.jmin;
  int szz = 1 + lim.kmax - lim.kmin;

  return szx*szy*szz;
}

/*****************************************************************************
 *
 *  lb_halo_enqueue_send
 *
 *  Pack the send buffer. The ireq determines the direction of the
 *  communication.
 *
 *****************************************************************************/

int lb_halo_enqueue_send(const lb_t * lb, const lb_halo_t * h, int ireq) {

  assert(1 <= ireq && ireq < h->map.nvel);
  assert(lb->ndist == 1);

  if (h->count[ireq] > 0) {

    int8_t mx = h->map.cv[ireq][X];
    int8_t my = h->map.cv[ireq][Y];
    int8_t mz = h->map.cv[ireq][Z];
    int8_t mm = mx*mx + my*my + mz*mz;

    int ib = 0; /* Buffer index */

    assert(mm == 1 || mm == 2 || mm == 3);

    for (int ic = h->slim[ireq].imin; ic <= h->slim[ireq].imax; ic++) {
      for (int jc = h->slim[ireq].jmin; jc <= h->slim[ireq].jmax; jc++) {
        for (int kc = h->slim[ireq].kmin; kc <= h->slim[ireq].kmax; kc++) {
	  /* If full, we need p = 0 */
          for (int p = 0; p < lb->nvel; p++) {
	    int8_t px = lb->model.cv[p][X];
	    int8_t py = lb->model.cv[p][Y];
	    int8_t pz = lb->model.cv[p][Z];
            int dot = mx*px + my*py + mz*pz;
            if (h->full || dot == mm) {
	      int index = cs_index(lb->cs, ic, jc, kc);
	      int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, 0, p);
	      h->send[ireq][ib++] = lb->f[laddr];
	    }
          }
        }
      }
    }
    assert(ib == h->count[ireq]*cs_limits_size(h->slim[ireq]));
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_dequeue_recv
 *
 *  Unpack the recv buffer into place in the distributions.
 *
 *****************************************************************************/

int lb_halo_dequeue_recv(lb_t * lb, const lb_halo_t * h, int ireq) {

  assert(lb);
  assert(h);
  assert(0 < ireq && ireq < h->map.nvel);
  assert(lb->ndist == 1);

  if (h->count[ireq] > 0) {

    /* The communication direction is reversed cf. the send... */
    int8_t mx = h->map.cv[h->map.nvel-ireq][X];
    int8_t my = h->map.cv[h->map.nvel-ireq][Y];
    int8_t mz = h->map.cv[h->map.nvel-ireq][Z];
    int8_t mm = mx*mx + my*my + mz*mz;

    int ib = 0; /* Buffer index */
    double * recv = h->recv[ireq];

    {
      int i = 1 + mx;
      int j = 1 + my;
      int k = 1 + mz;
      /* If Cartesian neighbour is self, just copy out of send buffer. */
      if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) recv = h->send[ireq]; 
    }

    assert(mm == 1 || mm == 2 || mm == 3);

    for (int ic = h->rlim[ireq].imin; ic <= h->rlim[ireq].imax; ic++) {
      for (int jc = h->rlim[ireq].jmin; jc <= h->rlim[ireq].jmax; jc++) {
        for (int kc = h->rlim[ireq].kmin; kc <= h->rlim[ireq].kmax; kc++) {
          for (int p = 0; p < lb->nvel; p++) {
	    /* For reduced swap, we must have -cv[p] here... */
	    int8_t px = lb->model.cv[lb->nvel-p][X];
	    int8_t py = lb->model.cv[lb->nvel-p][Y];
	    int8_t pz = lb->model.cv[lb->nvel-p][Z];
            int dot = mx*px + my*py + mz*pz;
            if (h->full || dot == mm) {
	      int index = cs_index(lb->cs, ic, jc, kc);
              int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, 0, p);
	      lb->f[laddr] = recv[ib++];
	    }
          }
        }
      }
    }
    assert(ib == h->count[ireq]*cs_limits_size(h->rlim[ireq]));
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_create
 *
 *  Currently: generate all send and receive requests.
 *
 *****************************************************************************/

int lb_halo_create(const lb_t * lb, lb_halo_t * h, int full) {

  lb_halo_t hnull = {};

  assert(lb);
  assert(h);

  *h = hnull;

  /* Communication model */
  if (lb->model.ndim == 2) lb_model_create( 9, &h->map);
  if (lb->model.ndim == 3) lb_model_create(27, &h->map);

  assert(h->map.ndim == lb->model.ndim);

  cs_nlocal(lb->cs, h->nlocal);
  cs_cart_comm(lb->cs, &h->comm);
  h->tagbase = 211216;
  h->full = full;

  /* Determine look-up table of ranks of neighbouring processes */
  {
    int dims[3] = {};
    int periods[3] = {};
    int coords[3] = {};

    MPI_Cart_get(h->comm, h->map.ndim, dims, periods, coords);

    for (int p = 0; p < h->map.nvel; p++) {
      int nbr[3] = {};
      int out[3] = {};  /* Out-of-range is erroneous for non-perioidic dims */
      int i = 1 + h->map.cv[p][X];
      int j = 1 + h->map.cv[p][Y];
      int k = 1 + h->map.cv[p][Z];

      nbr[X] = coords[X] + h->map.cv[p][X];
      nbr[Y] = coords[Y] + h->map.cv[p][Y];
      nbr[Z] = coords[Z] + h->map.cv[p][Z];
      out[X] = (!periods[X] && (nbr[X] < 0 || nbr[X] > dims[X]));
      out[Y] = (!periods[Y] && (nbr[Y] < 0 || nbr[Y] > dims[Y]));
      out[Z] = (!periods[Z] && (nbr[Z] < 0 || nbr[Z] > dims[Z]));

      if (out[X] || out[Y] || out[Z]) {
	h->nbrrank[i][j][k] = MPI_PROC_NULL;
      }
      else {
	MPI_Cart_rank(h->comm, nbr, &h->nbrrank[i][j][k]);
      }
    }
    /* I must be in the middle */
    assert(h->nbrrank[1][1][1] == cs_cart_rank(lb->cs));
  }


  /* Limits of the halo regions in each communication direction */

  for (int p = 1; p < h->map.nvel; p++) {

    /* Limits for send and recv regions*/
    int8_t cx = h->map.cv[p][X];
    int8_t cy = h->map.cv[p][Y];
    int8_t cz = h->map.cv[p][Z];

    cs_limits_t send = {1, h->nlocal[X], 1, h->nlocal[Y], 1, h->nlocal[Z]};
    cs_limits_t recv = {1, h->nlocal[X], 1, h->nlocal[Y], 1, h->nlocal[Z]};

    if (cx == -1) send.imax = 1;
    if (cx == +1) send.imin = send.imax;
    if (cy == -1) send.jmax = 1;
    if (cy == +1) send.jmin = send.jmax;
    if (cz == -1) send.kmax = 1;
    if (cz == +1) send.kmin = send.kmax;

    /* velocity is reversed... */
    if (cx == +1) recv.imax = recv.imin = 0;
    if (cx == -1) recv.imin = recv.imax = recv.imax + 1;
    if (cy == +1) recv.jmax = recv.jmin = 0;
    if (cy == -1) recv.jmin = recv.jmax = recv.jmax + 1;
    if (cz == +1) recv.kmax = recv.kmin = 0;
    if (cz == -1) recv.kmin = recv.kmax = recv.kmax + 1;

    h->slim[p] = send;
    h->rlim[p] = recv;
  }

  /* Message count (velocities) for each communication direction */

  for (int p = 1; p < h->map.nvel; p++) {

    int count = 0;

    if (h->full) {
      count = lb->model.nvel;
    }
    else {
      int8_t mx = h->map.cv[p][X];
      int8_t my = h->map.cv[p][Y];
      int8_t mz = h->map.cv[p][Z];
      int8_t mm = mx*mx + my*my + mz*mz;

      /* Consider each model velocity in turn */
      for (int q = 1; q < lb->model.nvel; q++) {
	int8_t qx = lb->model.cv[q][X];
	int8_t qy = lb->model.cv[q][Y];
	int8_t qz = lb->model.cv[q][Z];
	int8_t dot = mx*qx + my*qy + mz*qz;

	if (mm == 3 && dot == mm) count +=1;   /* This is a corner */
	if (mm == 2 && dot == mm) count +=1;   /* This is an edge */
	if (mm == 1 && dot == mm) count +=1;   /* This is a side */
      }
    }

    h->count[p] = count;
    /* Allocate send buffer for send region */
    if (count > 0) {
      int scount = count*cs_limits_size(h->slim[p]);
      h->send[p] = (double *) malloc(scount*sizeof(double));
      assert(h->send[p]);
    }
    /* Allocate recv buffer */
    if (count > 0) {
      int rcount = count*cs_limits_size(h->rlim[p]);
      h->recv[p] = (double *) malloc(rcount*sizeof(double));
      assert(h->recv[p]);
    }
  }

  /* Post recvs (from opposite direction cf send) */

  for (int ireq = 0; ireq < h->map.nvel; ireq++) {

    h->request[ireq] = MPI_REQUEST_NULL;

    if (h->count[ireq] > 0) {
      int i = 1 + h->map.cv[h->map.nvel-ireq][X];
      int j = 1 + h->map.cv[h->map.nvel-ireq][Y];
      int k = 1 + h->map.cv[h->map.nvel-ireq][Z];
      int mcount = h->count[ireq]*cs_limits_size(h->rlim[ireq]);

      if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) mcount = 0;
      
      MPI_Irecv(h->recv[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
		h->tagbase + ireq, h->comm, h->request + ireq);
    }
  }

  /* Enqueue sends (upper half of request array) */

  #pragma omp parallel for schedule(dynamic, 1)
  for (int ireq = 0; ireq < h->map.nvel; ireq++) {

    h->request[27+ireq] = MPI_REQUEST_NULL;

    if (h->count[ireq] > 0) {
      int i = 1 + h->map.cv[ireq][X];
      int j = 1 + h->map.cv[ireq][Y];
      int k = 1 + h->map.cv[ireq][Z];
      int mcount = h->count[ireq]*cs_limits_size(h->slim[ireq]);

      lb_halo_enqueue_send(lb, h, ireq);

      /* Short circuit messages to self. */
      if (h->nbrrank[i][j][k] == h->nbrrank[1][1][1]) mcount = 0;

      #pragma omp critical
      {
	MPI_Isend(h->send[ireq], mcount, MPI_DOUBLE, h->nbrrank[i][j][k],
		  h->tagbase + ireq, h->comm, h->request + 27 + ireq);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_halo_free
 *
 *  Complete all the send and receive requests.
 *
 *****************************************************************************/

int lb_halo_free(lb_t * lb, lb_halo_t * h) {

  assert(lb);
  assert(h);

  /* Can free() be used with thread safety? */

  #pragma omp parallel for schedule(dynamic, 1)
  for (int ireq = 0; ireq < 2*h->map.nvel; ireq++) {

    int issatisfied = -1;
    MPI_Status status = {};

    #pragma omp critical
    {
      MPI_Waitany(2*h->map.nvel, h->request, &issatisfied, &status);
    }
    /* Check status is what we expect? */

    if (issatisfied == MPI_UNDEFINED) {
      /* No action e.g., for (0,0,0) case */
    }
    else {
      /* Handle either send or recv request completion */
      if (issatisfied < h->map.nvel) {
	/* This is a recv */
	int irreq = issatisfied;
	lb_halo_dequeue_recv(lb, h, irreq);
	free(h->recv[irreq]);
      }
      else {
	/* This was a send */
	int isreq = issatisfied - 27;
	free(h->send[isreq]);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_lb_data_check_set
 *
 *  Set unique test values in the distribution.
 * 
 *****************************************************************************/

int util_lb_data_check_set(lb_t * lb) {

  int nlocal[3] = {};

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	for (int p = 0 ; p < lb->model.nvel; p++) {
	  int index = cs_index(lb->cs, ic, jc, kc);
	  int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, 0, p);
	  lb->f[laddr] = 1.0*lb_data_index(lb, ic, jc, kc, p); 
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_lb_data_check
 *
 *  Examine halo values and check they are as expected.
 *
 *****************************************************************************/

int util_lb_data_check(lb_t * lb, int full) {

  int ifail = 0;
  int nh = 1;
  int nhk = nh;
  int nlocal[3] = {};

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  /* Fix for 2d, where there should be no halo regions in Z */
  if (lb->ndim == 2) nhk = 0;

  for (int ic = 1 - nh; ic <= nlocal[X] + nh; ic++) {
    for (int jc = 1 - nh; jc <= nlocal[Y] + nh; jc++) {
      for (int kc = 1 - nhk; kc <= nlocal[Z] + nhk; kc++) {

	int is_halo = (ic < 1 || jc < 1 || kc < 1 ||
		       ic > nlocal[X] || jc > nlocal[Y] || kc > nlocal[Z]);

	if (is_halo == 0) continue;

	int index = cs_index(lb->cs, ic, jc, kc);

	for (int p = 0; p < lb->model.nvel; p++) {

	  /* Look for propagating distributions (into domain). */
	  int icdt = ic + lb->model.cv[p][X];
	  int jcdt = jc + lb->model.cv[p][Y];
	  int kcdt = kc + lb->model.cv[p][Z];

	  is_halo = (icdt < 1 || jcdt < 1 || kcdt < 1 ||
		     icdt > nlocal[X] || jcdt > nlocal[Y] || kcdt > nlocal[Z]);

	  if (full || is_halo == 0) {
	    /* Check */
	    int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, 0, p);
	    double fex = 1.0*lb_data_index(lb, ic, jc, kc, p);
	    if (fabs(fex - lb->f[laddr]) > DBL_EPSILON) ifail += 1;
	    assert(fabs(fex - lb->f[laddr]) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_halo_create
 *
 *****************************************************************************/

int test_lb_halo_create(pe_t * pe, cs_t * cs, int ndim, int nvel, int full) {

  lb_data_options_t options = {.ndim = ndim, .nvel = nvel, .ndist = 1};
  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_data_create(pe, cs, &options, &lb);

  util_lb_data_check_set(lb);

  {
    lb_halo_t h = {};
    lb_halo_create(lb, &h, full);
    lb_halo_free(lb, &h);
  }

  util_lb_data_check(lb, full);

  lb_data_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_halo
 *
 *****************************************************************************/

int test_lb_halo(pe_t * pe) {

  assert(pe);

  /* Two dimensional system */
  {
    cs_t * cs = NULL;
    int ntotal[3] = {64, 64, 1};

    cs_create(pe, &cs);
    cs_ntotal_set(cs, ntotal);
    cs_init(cs);

    test_lb_halo_create(pe, cs, 2, 9, 0);
    test_lb_halo_create(pe, cs, 2, 9, 1);

    cs_free(cs);
  }

  /* Three dimensional system */
  {
    cs_t * cs = NULL;

    cs_create(pe, &cs);
    cs_init(cs);

    test_lb_halo_create(pe, cs, 3, 15, 0);
    test_lb_halo_create(pe, cs, 3, 15, 1);
    test_lb_halo_create(pe, cs, 3, 19, 0);
    test_lb_halo_create(pe, cs, 3, 19, 1);
    test_lb_halo_create(pe, cs, 3, 27, 0);
    test_lb_halo_create(pe, cs, 3, 27, 1);
    
    cs_free(cs);
  }

  return 0;
}


/*****************************************************************************
 *
 *  test_model_suite
 *
 *****************************************************************************/

int test_model_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lb_halo(pe);

  cs_create(pe, &cs);
  cs_init(cs);

  /* Test model structure (coordinate-independent stuff) */

  test_model_constants();
  test_model_velocity_set();

  /* Now test actual distributions */

  do_test_model_distributions(pe, cs);
  do_test_model_halo_swap(pe, cs);
  if (DATA_MODEL == DATA_MODEL_AOS && NSIMDVL == 1) {
    do_test_model_reduced_halo_swap(pe, cs);
  }
  do_test_lb_model_io(pe, cs);

  pe_info(pe, "PASS     ./unit/test_model\n");
  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_model_constants
 *
 *  Check the various constants associated with the reduced halo swap.
 *
 *****************************************************************************/

static void test_model_constants(void) {

#ifdef TEST_TO_BE_REMOVED_WITH_GLOBAL_SYMBOLS
  int i, k, p;

  for (i = 0; i < CVXBLOCK; i++) {
    for (k = 0; k < xblocklen_cv[i]; k++) {
      p = xdisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][X] == +1);
      p = xdisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][X] == -1);
    }
  }

  for (i = 0; i < CVYBLOCK; i++) {
    for (k = 0; k < yblocklen_cv[i]; k++) {
      p = ydisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Y] == +1);
      p = ydisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Y] == -1);
    }
  }

  for (i = 0; i < CVZBLOCK; i++) {
    for (k = 0; k < zblocklen_cv[i]; k++) {
      p = zdisp_fwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Z] == +1);
      p = zdisp_bwd_cv[i] + k;
      test_assert(p >= 0 && p < NVEL);
      test_assert(cv[p][Z] == -1);
    }
  }
#endif
  return;
}

/*****************************************************************************
 *
 *  test_model_velocity_set
 *
 *  Check the velocities, kinetic projector, tables of eigenvectors
 *  etc etc are all consistent for the current model.
 *
 *****************************************************************************/

static void test_model_velocity_set(void) {

  test_assert(NHYDRO == (1 + NDIM + NDIM*(NDIM+1)/2));

  return;
}

/*****************************************************************************
 *
 *  do_test_model_distributions
 *
 *  Test the distribution interface.
 *
 *****************************************************************************/

int do_test_model_distributions(pe_t * pe, cs_t * cs) {

  int i, n, p;
  int index = 1;
  int ndist = 2;
  double fvalue, fvalue_expected;
  double u[3];

  lb_t * lb;

  assert(pe);
  assert(cs);

  /* Tests of the basic distribution functions. */

  lb_create(pe, cs, &lb);
  assert(lb);
  lb_ndist(lb, &n);
  assert(n == 1); /* Default */

  lb_ndist_set(lb, ndist);
  lb_init(lb);

  /* Report the number of distributions */

  lb_ndist(lb, &n);
  assert(n == ndist);

  for (n = 0; n < ndist; n++) {
    for (p = 0; p < lb->model.nvel; p++) {
      fvalue_expected = 0.01*n + lb->model.wv[p];
      lb_f_set(lb, index, p, n, fvalue_expected);
      lb_f(lb, index, p, n, &fvalue);
      assert(fabs(fvalue - fvalue_expected) < DBL_EPSILON);
    }

    /* Check zeroth moment... */

    fvalue_expected = 0.01*n*lb->model.nvel + 1.0;
    lb_0th_moment(lb, index, (lb_dist_enum_t) n, &fvalue);
    assert(fabs(fvalue - fvalue_expected) <= DBL_EPSILON);

    /* Check first moment... */

    lb_1st_moment(lb, index, (n == 0) ? LB_RHO : LB_PHI, u);

    for (i = 0; i < lb->model.ndim; i++) {
      assert(fabs(u[i] - 0.0) < DBL_EPSILON);
    }
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_model_halo_swap
 *
 *  Test full halo swap.
 *
 *****************************************************************************/

int do_test_model_halo_swap(pe_t * pe, cs_t * cs) {

  int i, j, k, p;
  int n, ndist = 2;
  int index, nlocal[3];
  const int nextra = 1;  /* Distribution halo width always 1 */
  double f_expect;
  double f_actual;

  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_create(pe, cs, &lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);

  cs_nlocal(cs, nlocal);

  /* The test relies on a uniform decomposition in parallel:
   *
   * f[0] or f[X] is set to local x index,
   * f[1] or f[Y] is set to local y index
   * f[2] or f[Z] is set to local z index
   * remainder are set to velocity index. */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {

	index = cs_index(cs, i, j, k);

	for (n = 0; n < ndist; n++) {
	  lb_f_set(lb, index, X, n, (double) (i));
	  lb_f_set(lb, index, Y, n, (double) (j));
	  lb_f_set(lb, index, Z, n, (double) (k));

	  for (p = 3; p < lb->model.nvel; p++) {
	    lb_f_set(lb, index, p, n, (double) p);
	  }
	}
      }
    }
  }

  lb_memcpy(lb, tdpMemcpyHostToDevice);
  lb_halo(lb);
  lb_memcpy(lb, tdpMemcpyDeviceToHost);

  /* Test all the sites not in the interior */

  for (i = 1 - nextra; i <= nlocal[X] + nextra; i++) {
    if (i >= 1 && i <= nlocal[X]) continue;
    for (j = 1 - nextra; j <= nlocal[Y] + nextra; j++) {
      if (j >= 1 && j <= nlocal[Y]) continue;
      for (k = 1 - nextra; k <= nlocal[Z] + nextra; k++) {
	if (k >= 1 && k <= nlocal[Z]) continue;

	index = cs_index(cs, i, j, k);

	for (n = 0; n < ndist; n++) {

	  f_expect = 1.0*abs(i - nlocal[X]);
	  lb_f(lb, index, X, n, &f_actual);
	  test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  f_expect = 1.0*abs(j - nlocal[Y]);
	  lb_f(lb, index, Y, n, &f_actual);
	  test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  f_expect = 1.0*abs(k - nlocal[Z]);
	  lb_f(lb, index, Z, n, &f_actual);
	  test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);

	  for (p = 3; p < lb->model.nvel; p++) {
	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = (double) p;
	    test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_model_reduced_halo_swap
 *
 *****************************************************************************/

int do_test_model_reduced_halo_swap(pe_t * pe, cs_t * cs) {  

  int i, j, k, p;
  int icdt, jcdt, kcdt;
  int index, nlocal[3];
  int n, ndist = 2;
  const int nextra = 1;

  double f_expect;
  double f_actual;

  lb_t * lb = NULL;

  assert(pe);
  assert(cs);

  lb_create(pe, cs, &lb);
  assert(lb);
  lb_ndist_set(lb, ndist);
  lb_init(lb);
  lb_halo_set(lb, LB_HALO_REDUCED);

  cs_nlocal(cs, nlocal);

  /* Set everything which is NOT in a halo */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = cs_index(cs, i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    f_expect = 1.0*(n*lb->model.nvel + p);
	    lb_f_set(lb, index, p, n, f_expect);
	  }
	}
      }
    }
  }

  lb_halo_via_struct(lb);

  /* Now check that the interior sites are unchanged */

  for (i = 1; i <= nlocal[X]; i++) {
    for (j = 1; j <= nlocal[Y]; j++) {
      for (k = 1; k <= nlocal[Z]; k++) {
	index = cs_index(cs, i, j, k);
	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < lb->model.nvel; p++) {
	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*lb->model.nvel +  p);
	    test_assert(fabs(f_expect - f_actual) < DBL_EPSILON);
	  }
	}
      }
    }
  }

  /* Also check the halos sites. The key test of the reduced halo
   * swap is that distributions for which r + c_i dt takes us into
   * the domain proper must be correct. */

  for (i = 1 - nextra; i <= nlocal[X] + nextra; i++) {
    if (i >= 1 && i <= nlocal[X]) continue;
    for (j = 1 - nextra; j <= nlocal[Y] + nextra; j++) {
      if (j >= 1 && j <= nlocal[Y]) continue;
      for (k = 1 - nextra; k <= nlocal[Z] + nextra; k++) {
	if (k >= 1 && k <= nlocal[Z]) continue;

	index = cs_index(cs, i, j, k);

	for (n = 0; n < ndist; n++) {
	  for (p = 0; p < lb->model.nvel; p++) {

	    lb_f(lb, index, p, n, &f_actual);
	    f_expect = 1.0*(n*lb->model.nvel + p);

	    icdt = i + lb->model.cv[p][X];
	    jcdt = j + lb->model.cv[p][Y];
	    kcdt = k + lb->model.cv[p][Z];

	    if (test_model_is_domain(cs, icdt, jcdt, kcdt)) {
	      test_assert(fabs(f_actual - f_expect) < DBL_EPSILON);
	    }
	  }
	}

	/* Next site */
      }
    }
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  test_model_is_domain
 *
 *  Is (ic, jc, kc) in the domain proper?
 *
 *****************************************************************************/

static int test_model_is_domain(cs_t * cs, int ic, int jc, int kc) {

  int nlocal[3];
  int iam = 1;

  assert(cs);

  cs_nlocal(cs, nlocal);

  if (ic < 1) iam = 0;
  if (jc < 1) iam = 0;
  if (kc < 1) iam = 0;
  if (ic > nlocal[X]) iam = 0;
  if (jc > nlocal[Y]) iam = 0;
  if (kc > nlocal[Z]) iam = 0;

  return iam;
}

/*****************************************************************************
 *
 *  do_test_lb_model_io
 *
 *****************************************************************************/

int do_test_lb_model_io(pe_t * pe, cs_t * cs) {

  int ndist = 2;
  lb_t * lbrd = NULL;
  lb_t * lbwr = NULL;

  assert(pe);
  assert(cs);

  lb_create_ndist(pe, cs, ndist, &lbrd);
  lb_create_ndist(pe, cs, ndist, &lbwr);

  lb_init(lbwr);
  lb_init(lbrd);

  /* Write */

  /* Read */

  /* Compare */

  lb_free(lbwr);
  lb_free(lbrd);

  return 0;
}
