/*****************************************************************************
 *
 *  test_lb_data.c
 *
 *  Distribution data, including halo swap and i/o.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025 The University of Edinburgh
 *
 *  Contributing author:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "lb_data.h"


int test_lb_data_create(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);
int test_lb_f(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);
int test_lb_f_set(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);

int test_lb_data_halo(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);
  
int test_lb_data_io(pe_t * pe, cs_t * cs);
int test_lb_write_buf(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);
int test_lb_write_buf_ascii(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);
int test_lb_io_aggr_pack(pe_t * pe, cs_t * cs, const lb_data_options_t * opts);


int util_lb_data_check_initialise(lb_t * lb);
int util_lb_data_check(lb_t * lb, int full);
int util_lb_data_check_no_halo(lb_t * lb);

/*****************************************************************************
 *
 *  test_lb_data_suite
 *
 *****************************************************************************/

int test_lb_data_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* Two dimensional system */
  {
    int ntotal[3] = {64, 64, 1};
    cs_t * cs = NULL;

    cs_create(pe, &cs);
    cs_ntotal_set(cs, ntotal);
    cs_init(cs);

    /* D2Q9 ndist = 1 */
    {
      int ndist = 1;
      lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(2, 9, ndist);

      test_lb_data_create(pe, cs, &opts);

      test_lb_f(pe, cs, &opts);
      test_lb_f_set(pe, cs, &opts);
      test_lb_data_halo(pe, cs, &opts);

    }

    /* D2Q9 ndist = 2 */
    {
      int ndist = 2;
      lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(2, 9, ndist);

      test_lb_data_create(pe, cs, &opts);

      test_lb_f(pe, cs, &opts);
      test_lb_f_set(pe, cs, &opts);
      test_lb_data_halo(pe, cs, &opts);
    }

    cs_free(cs);
  }

  /* Three dimension system */
  {
    int ntotal[3] = {32, 32, 32};
    cs_t * cs = NULL;

    cs_create(pe, &cs);
    cs_ntotal_set(cs, ntotal);
    cs_init(cs);

    /* D3Q15 (not much used thse days) */
    {
      int ndist = 1;
      lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 15, ndist);

      test_lb_data_create(pe, cs, &opts);

      test_lb_f(pe, cs, &opts);
      test_lb_f_set(pe, cs, &opts);
      test_lb_data_halo(pe, cs, &opts);
    }

    /* D3Q19 */
    {
      int ndist = 1;
      lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 19, ndist);

      test_lb_data_create(pe, cs, &opts);

      test_lb_f(pe, cs, &opts);
      test_lb_f_set(pe, cs, &opts);
      test_lb_data_halo(pe, cs, &opts);
    }
    /* D3Q19 (used ndist = 2 in the past) */
    {
      int ndist = 2;
      lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 19, ndist);

      test_lb_data_create(pe, cs, &opts);

      test_lb_f(pe, cs, &opts);
      test_lb_f_set(pe, cs, &opts);
      test_lb_data_halo(pe, cs, &opts);
    }

    /* D3Q27 */
    {
      int ndist = 1;
      lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 27, ndist);

      test_lb_data_create(pe, cs, &opts);

      test_lb_f(pe, cs, &opts);
      test_lb_f_set(pe, cs, &opts);
      test_lb_data_halo(pe, cs, &opts);
    }
    
    test_lb_data_io(pe, cs);
    cs_free(cs);
  }

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_data_create
 *
 *****************************************************************************/

int test_lb_data_create(pe_t * pe, cs_t * cs, const lb_data_options_t * opts) {

  int ifail = 0;
  lb_t * lb = NULL;

  assert(NVELMAX == 27);
  assert(LB_RECORD_LENGTH_ASCII == 23);

  /* Really a collision concern ... */
  assert(NHYDRO == (1 + NDIM + NDIM*(NDIM+1)/2));

  ifail = lb_data_create(pe, cs, opts, &lb);
  assert(ifail == 0);

  /* Host */

  assert(lb->ndim  == opts->ndim);
  assert(lb->nvel  == opts->nvel);
  assert(lb->ndist == opts->ndist);
  assert(lb->nsite == cs->param->nsites);

  assert(lb->pe    == pe);
  assert(lb->cs    == cs);

  /* We will assume this is a sufficient check of the model ... */
  assert(lb->model.ndim == lb->ndim);
  assert(lb->model.nvel == lb->nvel);

  /* i/o quantities are dealt with separately */

  /* distribution storage */
  assert(lb->f);
  assert(lb->fprime);

  assert(lb->nrelax     == LB_RELAXATION_M10); /* Default */
  assert(lb->haloscheme == LB_HALO_FULL);      /* Default */

  if (cs->leopts.nplanes > 0) {
    assert(lb->sbuff);
    assert(lb->rbuff);
  }

  /* Target */
  /* Should really have a kernel to check the device copy */

  lb_free(lb);

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_f
 *
 *****************************************************************************/

int test_lb_f(pe_t * pe, cs_t * cs, const lb_data_options_t * opts) {

  int ifail = 0;
  lb_t * lb = NULL;

  ifail = lb_data_create(pe, cs, opts, &lb);
  assert(ifail == 0);

  /* Assign some non-zero values */
  {
    int index = 13;

    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->nvel; p++) {
	int iaddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	lb->f[iaddr] = 1.0*((1 + n)*lb->nvel + 1 + p);
      }
    }

    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->nvel; p++) {
	double f = 0.0;
	lb_f(lb, index, p, (n == 0) ? LB_RHO : LB_PHI, &f);
	assert(fabs(f - 1.0*((1 + n)*lb->nvel + 1 + p) < DBL_EPSILON));
      }
    }
  }

  lb_free(lb);

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_f_set
 *
 *****************************************************************************/

int test_lb_f_set(pe_t * pe, cs_t * cs, const lb_data_options_t * opts) {

  int ifail = 0;
  lb_t * lb = NULL;

  ifail = lb_data_create(pe, cs, opts, &lb);
  assert(ifail == 0);

  {
    /* Assign some values */
    int index = 12;

    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->nvel; p++) {
	double f = 1.0*((1 + n)*lb->nvel + 1 + p);
	lb_f_set(lb, index, p, (n == 0) ? LB_RHO : LB_PHI, f);
      }
    }
    /* .. and chcek */
    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->nvel; p++) {
	int iaddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	double f = 1.0*((1 + n)*lb->nvel + 1 + p);
	ifail = !(fabs(lb->f[iaddr] - f) < DBL_EPSILON);
	assert(ifail == 0);
      }
    }
  }

  lb_free(lb);

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_data_halo
 *
 *  Driver.
 *
 *****************************************************************************/

int test_lb_data_halo(pe_t * pe, cs_t * cs, const lb_data_options_t * opts) {

  int ifail = 0;

  /* Full halo should be default */
  assert(opts->halo == LB_HALO_FULL);

  /* Full halo */
  {
    lb_t * lb = NULL;

    ifail = lb_data_create(pe, cs, opts, &lb);
    assert(ifail == 0);

    util_lb_data_check_initialise(lb);
    lb_memcpy(lb, tdpMemcpyHostToDevice);

    lb_halo(lb);

    lb_memcpy(lb, tdpMemcpyDeviceToHost);
    util_lb_data_check(lb, 1);
    lb_free(lb);
  }

  /* Reduced halo */
  {
    lb_t * lb = NULL;
    lb_data_options_t reduced_opts = *opts;
    reduced_opts.halo = LB_HALO_REDUCED;

    ifail = lb_data_create(pe, cs, &reduced_opts, &lb);
    assert(ifail == 0);

    util_lb_data_check_initialise(lb);
    lb_memcpy(lb, tdpMemcpyHostToDevice);

    lb_halo(lb);

    lb_memcpy(lb, tdpMemcpyDeviceToHost);
    util_lb_data_check(lb, 0);

    lb_free(lb);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_data_io
 *
 *  Driver for all read/write io routines.
 *  E.g., test_lb_write_buf() also deals with lb_read_buf()
 *
 *****************************************************************************/

int test_lb_data_io(pe_t * pe, cs_t * cs) {

  assert(NVELMAX == 27);

  {
    lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(2, 9, 1);
    test_lb_write_buf(pe, cs, &opts);
    test_lb_write_buf_ascii(pe, cs, &opts);
    test_lb_io_aggr_pack(pe, cs, &opts);
  }

  {
    lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 15, 1);
    test_lb_write_buf(pe, cs, &opts);
    test_lb_write_buf_ascii(pe, cs, &opts);
    test_lb_io_aggr_pack(pe, cs, &opts);
  }

  {
    lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 19, 1);
    test_lb_write_buf(pe, cs, &opts);
    test_lb_write_buf_ascii(pe, cs, &opts);
    test_lb_io_aggr_pack(pe, cs, &opts);
  }

  {
    /* As D3Q19 is typically what was used for ndist = 2, here it is ... */
    lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 19, 2);
    test_lb_write_buf(pe, cs, &opts);
    test_lb_write_buf_ascii(pe, cs, &opts);
    test_lb_io_aggr_pack(pe, cs, &opts);
  }

  {
    lb_data_options_t opts = lb_data_options_ndim_nvel_ndist(3, 27, 1);
    test_lb_write_buf(pe, cs, &opts);
    test_lb_write_buf_ascii(pe, cs, &opts);
    test_lb_io_aggr_pack(pe, cs, &opts);
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_lb_write_buf
 *
 *  It is convenient to test lb_read_buf() at the same time.
 *
 *****************************************************************************/

int test_lb_write_buf(pe_t * pe, cs_t * cs, const lb_data_options_t * opts) {

  int ifail = 0;
  lb_t * lb = NULL;
  char buf[BUFSIZ] = {0};

  assert(pe);
  assert(cs);
  assert(opts);

  lb_data_create(pe, cs, opts, &lb);

  assert(lb->ndist*lb->model.nvel*sizeof(double) < sizeof(buf));

  {
    /* Set some data at position (2,3,4) */
    int index = cs_index(cs, 2, 3, 4);

    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->model.nvel; p++) {
	double f = 1.0*(1 + n*lb->model.nvel + p); /* Test data, avoid zero */
	lb_f_set(lb, index, p, n, f);
      }
    }

    lb_write_buf(lb, index, buf);
  }

  {
    /* Read same buf in a different location */
    int index = cs_index(cs, 3, 4, 5);
    lb_read_buf(lb, index, buf);

    /* Check the result in new position */
    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->model.nvel; p++) {
	double fref = 1.0*(1 + n*lb->model.nvel + p);
	double f = -1.0;
	lb_f(lb, index, p, n, &f);
	assert(fabs(f - fref) < DBL_EPSILON);
	if (fabs(f - fref) >= DBL_EPSILON) ifail += 1;
      }
    }
  }

  lb_free(lb);

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_write_buf_ascii
 *
 *****************************************************************************/

int test_lb_write_buf_ascii(pe_t * pe, cs_t * cs,
			    const lb_data_options_t * opts) {

  int ifail = 0;
  lb_t * lb = NULL;
  char buf[BUFSIZ] = {0};

  assert(pe);
  assert(cs);
  assert(opts);

  /* Size of ascii record musst fir in buffer ... */
  assert(opts->nvel*(opts->ndist*LB_RECORD_LENGTH_ASCII + 1) < BUFSIZ);

  lb_data_create(pe, cs, opts, &lb);

  /* Write some data */

  {
    /* Set some data at position (2,3,4) */
    int index = cs_index(cs, 2, 3, 4);

    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->model.nvel; p++) {
	double f = 1.0*(1 + n*lb->model.nvel + p); /* Test data, avoid zero */
	lb_f_set(lb, index, p, n, f);
      }
    }

    lb_write_buf_ascii(lb, index, buf);

    {
      /* Have we got the correct size? */
      int count = lb->nvel*(lb->ndist*LB_RECORD_LENGTH_ASCII + 1);
      size_t sz = count*sizeof(char);
      if (sz != strnlen(buf, BUFSIZ)) ifail = -1;
      assert(ifail == 0);
    }
  }

  {
    /* Read back in different memory position */
    int index = cs_index(cs, 4, 5, 6);
    lb_read_buf_ascii(lb, index, buf);

    /* Check the result in new position */
    for (int n = 0; n < lb->ndist; n++) {
      for (int p = 0; p < lb->model.nvel; p++) {
	double fref = 1.0*(1 + n*lb->model.nvel + p);
	double f = -1.0;
	lb_f(lb, index, p, n, &f);
	if (fabs(f - fref) >= DBL_EPSILON) ifail = -1;
	assert(ifail == 0);
      }
    }
  }

  lb_free(lb);

  return ifail;
}

/*****************************************************************************
 *
 *  test_lb_io_aggr_pack
 *
 *  It is convenient to test lb_io_aggr_unpack() at the same time.
 *
 *****************************************************************************/

int test_lb_io_aggr_pack(pe_t * pe, cs_t * cs, const lb_data_options_t *opts) {

  lb_t * lb = NULL;
  int nlocal[3] = {0};

  assert(pe);
  assert(cs);
  assert(opts);

  cs_nlocal(cs, nlocal);

  lb_data_create(pe, cs, opts, &lb);

  assert(lb->ascii.datatype == MPI_CHAR);
  assert(lb->ascii.datasize == sizeof(char));
  assert(lb->ascii.count    == lb->nvel*(1 + lb->ndist*LB_RECORD_LENGTH_ASCII));
  assert(lb->binary.datatype == MPI_DOUBLE);
  assert(lb->binary.datasize == sizeof(double));
  assert(lb->binary.count    == lb->nvel*lb->ndist);

  /* ASCII */
  /* Aggregator */

  {
    /* We don't use the metadata quantities here */
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    io_aggregator_t aggr = {0};

    io_aggregator_initialise(lb->ascii, lim, &aggr);
    util_lb_data_check_initialise(lb);
    lb_io_aggr_pack(lb, &aggr);

    /* Clear the ditributions, unpack, and check */
    memset(lb->f, 0, sizeof(double)*lb->nvel*lb->ndist*lb->nsite);

    lb_io_aggr_unpack(lb, &aggr);
    util_lb_data_check_no_halo(lb);

    io_aggregator_finalise(&aggr);
  }

  /* BINARY */

  {
    /* We don't use the metadata quantities here */
    cs_limits_t lim = {1, nlocal[X], 1, nlocal[Y], 1, nlocal[Z]};
    io_aggregator_t aggr = {0};

    io_aggregator_initialise(lb->binary, lim, &aggr);
    util_lb_data_check_initialise(lb);
    lb_io_aggr_pack(lb, &aggr);

    /* Clear the ditributions, unpack, and check */
    memset(lb->f, 0, sizeof(double)*lb->nvel*lb->ndist*lb->nsite);

    lb_io_aggr_unpack(lb, &aggr);
    util_lb_data_check_no_halo(lb);

    io_aggregator_finalise(&aggr);
  }

  lb_free(lb);

  return 0;
}

/*****************************************************************************
 *
 *  util_lb_data_index
 *
 *  Utility to return a unique value for global (ic,jc,kc,p)
 *  This allows e.g., tests to check distribution values in parallel
 *  exchanges.
 *
 *  (ic, jc, kc) are local indices
 *  The result could be an unsigned integer, but does need 64 bit...
 *
 *****************************************************************************/

int64_t util_lb_data_index(lb_t * lb, int ic, int jc, int kc, int n, int p) {

  int64_t index = INT64_MIN;
  int64_t nall[3] = {0};
  int64_t nstr[3] = {0};
  int64_t pstr    = 0;
  int64_t dstr    = 0;

  int ntotal[3] = {0};
  int offset[3] = {0};
  int nhalo = 0;

  assert(lb);
  assert(0 <= p && p < lb->model.nvel);
  assert(lb->ndist == 1 || lb->ndist == 2);
  assert(0 <= n && n < lb->ndist);

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
  dstr    = pstr*lb->model.nvel;

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

    index = dstr*n + pstr*p + nstr[X]*igl + nstr[Y]*jgl + nstr[Z]*kgl;
  }

  return index;
}

/*****************************************************************************
 *
 *  util_lb_data_check_set
 *
 *  Set unique test values in the distribution.
 * 
 *****************************************************************************/

int util_lb_data_check_initialise(lb_t * lb) {

  int nlocal[3] = {0};

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	for (int n = 0; n < lb->ndist; n++) {
	  for (int p = 0 ; p < lb->model.nvel; p++) {
	    int index = cs_index(lb->cs, ic, jc, kc);
	    int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	    lb->f[laddr] = 1.0*util_lb_data_index(lb, ic, jc, kc, n, p);
	  }
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
  int nlocal[3] = {0};

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

	for (int n = 0; n < lb->ndist; n++) {
	  for (int p = 0; p < lb->model.nvel; p++) {

	    /* Look for propagating distributions (into domain). */
	    int icdt = ic + lb->model.cv[p][X];
	    int jcdt = jc + lb->model.cv[p][Y];
	    int kcdt = kc + lb->model.cv[p][Z];

	    int is_prop = (icdt < 1 || jcdt < 1 || kcdt < 1 ||
		     icdt > nlocal[X] || jcdt > nlocal[Y] || kcdt > nlocal[Z]);

	    if (full || is_prop == 0) {
	      /* Check */
	      int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	      double fex = 1.0*util_lb_data_index(lb, ic, jc, kc, n, p);
	      if (fabs(fex - lb->f[laddr]) > DBL_EPSILON) ifail += 1;
	      assert(fabs(fex - lb->f[laddr]) < DBL_EPSILON);
	    }
	  }
	}
	/* Next (ic,jc,kc) */
      }
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_lb_data_check_no_halo
 *
 *  Examine non-halo values.
 *
 *****************************************************************************/

int util_lb_data_check_no_halo(lb_t * lb) {

  int ifail = 0;
  int nlocal[3] = {0};

  assert(lb);

  cs_nlocal(lb->cs, nlocal);

  /* Fix for 2d, where there should be no halo regions in Z */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(lb->cs, ic, jc, kc);

	for (int n = 0; n < lb->ndist; n++) {
	  for (int p = 0; p < lb->model.nvel; p++) {
	    /* Check */
	    int laddr = LB_ADDR(lb->nsite, lb->ndist, lb->nvel, index, n, p);
	    double fex = 1.0*util_lb_data_index(lb, ic, jc, kc, n, p);
	    if (fabs(fex - lb->f[laddr]) > DBL_EPSILON) ifail += 1;
	    assert(fabs(fex - lb->f[laddr]) < DBL_EPSILON);
	  }
	}
	/* Next (ic,jc,kc) */
      }
    }
  }

  return ifail;
}
