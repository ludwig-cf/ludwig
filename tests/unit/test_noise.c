/*****************************************************************************
 *
 *  test_noise.c
 *
 *  Test the basic lattice noise generator type.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "noise.h"

int test_noise_create(pe_t * pe, cs_t * cs);
int test_noise_initialise(pe_t * pe, cs_t * cs);
int test_noise_reap_n(pe_t * ps, cs_t * cs);

int test_noise_read_buf(pe_t * pe, cs_t * cs);
int test_noise_read_buf_ascii(pe_t * pe, cs_t * cs);
int test_noise_write_buf(pe_t * pe, cs_t * cs);
int test_noise_write_buf_ascii(pe_t * pe, cs_t * cs);
int test_noise_io_aggr_pack(pe_t * pe, cs_t * cs);
int test_noise_io_aggr_unpack(pe_t * pe, cs_t * cs);
int test_noise_io_write(pe_t * pe, cs_t * cs);
int test_noise_io_read(pe_t * pe, cs_t * cs);

int test_ns_uniform(void);
int test_ns_statistical_test(pe_t * pe);
int test_ns_statistical_testx(pe_t * pe);

/*****************************************************************************
 *
 *  test_noise_suite
 * 
 *****************************************************************************/

int test_noise_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* Class method */
  test_ns_uniform();

  {
    int ntotal[] = {8, 8, 8};
    cs_t * cs = NULL;
    cs_create(pe, &cs);
    cs_ntotal_set(cs, ntotal);
    cs_init(cs);

    test_noise_create(pe, cs);
    test_noise_initialise(pe, cs);
    test_noise_reap_n(pe, cs);

    test_noise_read_buf(pe, cs);
    test_noise_read_buf_ascii(pe, cs);
    test_noise_write_buf(pe, cs);
    test_noise_write_buf_ascii(pe, cs);
    test_noise_io_aggr_pack(pe, cs);
    test_noise_io_aggr_unpack(pe, cs);
    test_noise_io_write(pe, cs);
    test_noise_io_read(pe, cs);

    cs_free(cs);
  }

  /* Statistical tests */

  test_ns_statistical_test(pe);
  test_ns_statistical_testx(pe);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_ns_uniform
 *
 *****************************************************************************/

int test_ns_uniform(void) {

  int ifail = 0;

  /* state = zero */
  {
    unsigned int s[4] = {0, 0, 0, 0};
    unsigned int b = ns_uniform(s);

    assert(s[0] == 1234567);
    assert(s[1] == 0);
    assert(s[2] == 0);
    assert(s[3] == 0);
    if (b != s[0]) ifail = -1;
    assert(ifail == 0);
  }

  /* state non-zero */
  {
    unsigned int s[4] = {123, 456, 78, 9};
    unsigned int b = ns_uniform(s);

    assert(s[0] ==    9730054);
    assert(s[1] == 1905505352);
    assert(s[2] ==    2883582);
    assert(s[3] ==     162000);
    if (b != 1915204894) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_reap_n
 *
 *****************************************************************************/

int test_noise_reap_n(pe_t * pe, cs_t * cs) {

  noise_options_t opts = noise_options_default();
  noise_t * noise = NULL;

  noise_create(pe, cs, &opts, &noise);

  /* We expect the default initialisation ... */
  assert(noise->options.seed == 13);

  /* known case at (1, 1, 1) */
  {
    int index = cs_index(cs, 1, 1, 1);

    double a1 = sqrt(2.0 + sqrt(2.0));
    double a2 = sqrt(2.0 - sqrt(2.0));
    double r[NNOISE_MAX] = {0};

    noise_reap_n(noise, index, 10, r);

    /* Rank zero in Cartesian communicator owns (1, 1, 1) */
    MPI_Bcast(r, 10, MPI_DOUBLE, 0, cs->commcart);

    assert(fabs(r[0] - 0.0) < DBL_EPSILON);
    assert(fabs(r[1] - 0.0) < DBL_EPSILON);
    assert(fabs(r[2] - +a1) < DBL_EPSILON);
    assert(fabs(r[3] - -a2) < DBL_EPSILON);
    assert(fabs(r[4] - 0.0) < DBL_EPSILON);
    assert(fabs(r[5] - -a1) < DBL_EPSILON);
    assert(fabs(r[6] - -a1) < DBL_EPSILON);
    assert(fabs(r[7] - -a1) < DBL_EPSILON);
    assert(fabs(r[8] - 0.0) < DBL_EPSILON);
    assert(fabs(r[9] - -a2) < DBL_EPSILON);
  }

  noise_free(&noise);

  return 0;
}

/*****************************************************************************
 *
 *  test_noise_create
 *
 *****************************************************************************/

int test_noise_create(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* Most tests are deferred to test_noise_initialise() */
  {
    noise_options_t opts = noise_options_default();
    noise_t * ns = NULL;

    ifail = noise_create(pe, cs, &opts, &ns);
    assert(ifail == 0);
    assert(ns->pe == pe);
    assert(ns->cs == cs);
    assert(ns->state);

    ifail = noise_free(&ns);
    assert(ifail == 0);
    assert(ns == NULL);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_initialise
 *
 *****************************************************************************/

int test_noise_initialise(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  assert(pe);
  assert(cs);
  assert(NNOISE_MAX   == 10);
  assert(NNOISE_STATE == 4);

  const double a1 = sqrt(2.0 + sqrt(2.0));
  const double a2 = sqrt(2.0 - sqrt(2.0));

  /* Default */
  {
    noise_options_t opts = noise_options_default();
    noise_t ns = {};

    ifail = noise_initialise(pe, cs, &opts, &ns);
    assert(ifail == 0);
    assert(ns.pe == pe);
    assert(ns.cs == cs);
    assert(ns.options.seed == 13);
    assert(ns.nsites == cs->param->nsites);
    assert(ns.state);
    /* the table ... */
    assert(fabs(ns.rtable[0] - -a1) < DBL_EPSILON);
    assert(fabs(ns.rtable[1] - -a2) < DBL_EPSILON);
    assert(fabs(ns.rtable[2] - 0.0) < DBL_EPSILON);
    assert(fabs(ns.rtable[3] - 0.0) < DBL_EPSILON);
    assert(fabs(ns.rtable[4] - 0.0) < DBL_EPSILON);
    assert(fabs(ns.rtable[5] - 0.0) < DBL_EPSILON);
    assert(fabs(ns.rtable[6] - +a2) < DBL_EPSILON);
    assert(fabs(ns.rtable[7] - +a1) < DBL_EPSILON);

    /* State initial values (parallel) */

    ifail = noise_finalise(&ns);
    assert(ifail == 0);
    assert(ns.state == NULL);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_read_buf
 *
 *****************************************************************************/

int test_noise_read_buf(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* Check read against resulting state. Some values are ... */
  {
    unsigned int iv[4] = {1234567890, 2345678901, 13, 3456789012};
    noise_options_t opts = noise_options_default();
    noise_t ns = {};
    int index = cs_index(cs, 1, 1, 1);
    char buf[BUFSIZ] = {0};

    memcpy(buf, iv, 4*sizeof(unsigned int));

    noise_initialise(pe, cs, &opts, &ns);
    ifail = noise_read_buf(&ns, index, buf);
    assert(ifail == 0);
    for (int is = 0; is < 4; is++) {
      int iaddr = addr_rank1(ns.nsites, 4, index, is);
      if (iv[is] != ns.state[iaddr]) ifail -= 1;
      assert(ifail == 0);
    }
    noise_finalise(&ns);
  }


  return ifail;
}


/*****************************************************************************
 *
 *  test_noise_read_buf_ascii
 *
 *****************************************************************************/

int test_noise_read_buf_ascii(pe_t * pe, cs_t * cs) {

  const int nchar = NOISE_RECORD_LENGTH_ASCII;
  int ifail = 0;

  /* Check read against resulting state */
  {
    unsigned int iv[4] = {1234567890, 2345678901, 13, 3456789012};
    noise_options_t opts = noise_options_default();
    noise_t ns = {};
    int index = cs_index(cs, 1, 1, 1);
    char buf[BUFSIZ] = {0};

    snprintf(buf, 2 + 4*nchar, " %10u %10u %10u %10u\n",
	     iv[0], iv[1], iv[2], iv[3]);

    noise_initialise(pe, cs, &opts, &ns);
    ifail = noise_read_buf_ascii(&ns, index, buf);
    assert(ifail == 0);
    for (int is = 0; is < 4; is++) {
      int iaddr = addr_rank1(ns.nsites, 4, index, is);
      if (iv[is] != ns.state[iaddr]) ifail -= 1;
      assert(ifail == 0);
    }
    noise_finalise(&ns);
  }

  return ifail;
}


/*****************************************************************************
 *
 *  test_noise_write_buf
 *
 *****************************************************************************/

int test_noise_write_buf(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* Check output against the state */
  {
    char buf[BUFSIZ] = {0};
    unsigned int iv[4] = {0};
    int index = cs_index(cs, 1, 1, 1);
    noise_options_t opts = noise_options_default();
    noise_t ns = {};

    noise_initialise(pe, cs, &opts, &ns);

    ifail = noise_write_buf(&ns, index, buf);
    assert(ifail == 0);
    memcpy(iv, buf, 4*sizeof(unsigned int));

    for (int is = 0; is < 4; is++) {
      if (iv[is] != ns.state[addr_rank1(ns.nsites,4,index,is)]) ifail -= 1;
      assert(ifail == 0);
    }
    noise_finalise(&ns);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_write_buf_ascii
 *
 *****************************************************************************/

int test_noise_write_buf_ascii(pe_t * pe, cs_t * cs) {

  const int nchar = NOISE_RECORD_LENGTH_ASCII;
  int ifail = 0;

  /* Check output against the state */
  {
    char buf[BUFSIZ] = {0};
    int index = cs_index(cs, 1, 1, 1);
    noise_options_t opts = noise_options_default();
    noise_t ns = {};

    noise_initialise(pe, cs, &opts, &ns);
    ifail = noise_write_buf_ascii(&ns, index, buf);
    assert(ifail == 0);
    for (int is = 0; is < 4; is++) {
      unsigned long ival = strtoul(buf + is*nchar, NULL, 10);
      if (ival != ns.state[addr_rank1(ns.nsites,4,index,is)]) ifail -= 1;
      assert(ifail == 0);
    }
    noise_finalise(&ns);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_io_aggr_pack
 *
 *****************************************************************************/

int test_noise_io_aggr_pack(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  noise_options_t opts = noise_options_default();
  noise_t * ns = NULL;

  assert(pe);
  assert(cs);

  ifail = noise_create(pe, cs, &opts, &ns);
  assert(ifail == 0);

  /* Default option binary */
  {
    io_metadata_t * meta = &ns->output;
    io_aggregator_t aggr = {};

    io_aggregator_initialise(meta->element, meta->limits, &aggr);

    ifail = noise_io_aggr_pack(ns, &aggr);
    assert(ifail == 0);

    io_aggregator_finalise(&aggr);
  }

  noise_free(&ns);

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_io_aggr_unpack
 *
 *****************************************************************************/

int test_noise_io_aggr_unpack(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  noise_options_t opts = noise_options_default();
  noise_t * ns = NULL;

  assert(pe);
  assert(cs);

  ifail = noise_create(pe, cs, &opts, &ns);
  assert(ifail == 0);

  /* Default option binary */
  {
    io_metadata_t * meta = &ns->output;
    io_aggregator_t aggr = {};

    io_aggregator_initialise(meta->element, meta->limits, &aggr);

    ifail = noise_io_aggr_pack(ns, &aggr);
    assert(ifail == 0);
    ifail = noise_io_aggr_unpack(ns, &aggr);
    assert(ifail == 0);

    io_aggregator_finalise(&aggr);
  }

  noise_free(&ns);

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_io_write
 *
 *****************************************************************************/

int test_noise_io_write(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* ASCII write (time step is = 0) */
  {
    noise_options_t opts = noise_options_default();
    io_event_t ioevent = {0};
    noise_t * ns = NULL;

    opts.iodata.output.iorformat = IO_RECORD_ASCII;

    ifail = noise_create(pe, cs, &opts, &ns);
    ifail = noise_io_write(ns, 0, &ioevent);
    assert(ifail == 0);
    ifail = noise_free(&ns);
  }

  /* Binary write (time step is = 1) */
  {
    noise_options_t opts = noise_options_default();
    io_event_t ioevent = {0};
    noise_t * ns = NULL;

    ifail = noise_create(pe, cs, &opts, &ns);
    ifail = noise_io_write(ns, 1, &ioevent);
    assert(ifail == 0);
    ifail = noise_free(&ns);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_io_read
 *
 *  This reads the files from test_noise_io_write() above.
 *
 *****************************************************************************/

int test_noise_io_read(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* ASCII read (time step is = 0) */
  {
    noise_options_t opts = noise_options_default();
    io_event_t ioevent = {0};
    noise_t * ns = NULL;

    opts.iodata.output.iorformat = IO_RECORD_ASCII;

    ifail = noise_create(pe, cs, &opts, &ns);
    ifail = noise_io_read(ns, 0, &ioevent);
    assert(ifail == 0);
    ifail = noise_free(&ns);
  }

  /* Binary read (time step is = 1) */
  {
    noise_options_t opts = noise_options_default();
    io_event_t ioevent = {0};
    noise_t * ns = NULL;

    ifail = noise_create(pe, cs, &opts, &ns);
    ifail = noise_io_read(ns, 1, &ioevent);
    assert(ifail == 0);
    ifail = noise_free(&ns);
  }

  return ifail;
}
/*****************************************************************************
 *
 *  test_ns_statistical_test
 *
 *  Check mean and variance computed as sum over the whole system.
 *
 *****************************************************************************/

int test_ns_statistical_test(pe_t * pe) {

  cs_t * cs = NULL;
  noise_t * ns = NULL;
  noise_options_t opts = noise_options_default();

  int ntotal[3] = {64, 64, 64};
  int nlocal[3] = {0};

  double rstat[2] = {0};
  MPI_Comm comm = MPI_COMM_NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  cs_nlocal(cs, nlocal);
  cs_cart_comm(cs, &comm);

  noise_create(pe, cs, &opts, &ns); /* seed defualt */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	double r[NNOISE_MAX] = {0};
	int index = cs_index(cs, ic, jc, kc);

	noise_reap_n(ns, index, 10, r);

	for (int ir = 0; ir < NNOISE_MAX; ir++) {
	  rstat[0] += r[ir];
	  rstat[1] += r[ir]*r[ir];
	}
      }
    }
  }

  /* Mean and variance */

  {
    double rstat_local[2] = {rstat[0], rstat[1]};
    MPI_Allreduce(rstat_local, rstat, 2, MPI_DOUBLE, MPI_SUM, comm);
  }

  {
    double vol = 1.0*ntotal[X]*ntotal[Y]*ntotal[Z];
    rstat[0] = rstat[0]/vol;
    rstat[1] = rstat[1]/(NNOISE_MAX*vol) - rstat[0]*rstat[0];
  }

  /* These are the results for the default seeds, system size */

  assert(fabs(rstat[0] - 4.10105573e-03) < FLT_EPSILON);
  assert(fabs(rstat[1] - 1.00177840)     < FLT_EPSILON);

  noise_free(&ns);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_ns_statistical_testx
 *
 *  This checks up to 6th moment, which should see the following:
 *
 *  0th   \sum_i is the number of time steps times NNOISE_MAX
 *  1st   \sum_i x_i^1   = 0.0 (mean)
 *  2nd   \sum_i x_i^2   = 1.0 (variance)
 *  3rd   \sum_i x_i^3   = 0.0
 *  4th   \sum_i x_i^4   = 3.0
 *  5th   \sum_i x_i^5   = 0.0
 *  6th   \sum_i x_i^6   = 10.0
 *
 *  The statistics are computed at each lattice site. We can only do
 *  a small system (4x4x4) for as many as 1 million steps. All the
 *  moments are correct to a modest tolerance as this point.
 *
 *****************************************************************************/

int test_ns_statistical_testx(pe_t * pe) {

  int ntotal[3] = {4, 4, 4};
  int nlocal[3] = {0};

  cs_t * cs = NULL;
  noise_t * ns = NULL;
  noise_options_t opts = noise_options_default();

  /* Extent of the test */
  const int ntimes = 1000000;
  const double tolerance = 0.05;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  cs_nlocal(cs, nlocal);

  noise_create(pe, cs, &opts, &ns);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	double moment6[6] = {0};
	int index = cs_index(cs, ic, jc, kc);

	for (int nt = 0; nt < ntimes; nt++) {

	  double r[NNOISE_MAX] = {0};
	  noise_reap_n(ns, index, 10, r);

	  for (int n = 0; n < NNOISE_MAX; n++) {
	    double r1 = r[n];
	    double r2 = r1*r1;
	    moment6[0] += r1;
	    moment6[1] += r2;
	    moment6[2] += r2*r1;
	    moment6[3] += r2*r2;
	    moment6[4] += r2*r2*r1;
	    moment6[5] += r2*r2*r2;
	  }
	  /* Next time step */
	}

	/* Check */
	{
	  double m1 = moment6[0]/(1.0*ntimes*NNOISE_MAX);
	  double m2 = moment6[1]/(1.0*ntimes*NNOISE_MAX);
	  double m3 = moment6[2]/(1.0*ntimes*NNOISE_MAX);
	  double m4 = moment6[3]/(1.0*ntimes*NNOISE_MAX);
	  double m5 = moment6[4]/(1.0*ntimes*NNOISE_MAX);
	  double m6 = moment6[5]/(1.0*ntimes*NNOISE_MAX);

	  assert(fabs(m1 -  0.0) < tolerance);
	  assert(fabs(m2 -  1.0) < tolerance);
	  assert(fabs(m3 -  0.0) < tolerance);
	  assert(fabs(m4 -  3.0) < tolerance);
	  assert(fabs(m5 -  0.0) < tolerance);
	  assert(fabs(m6 - 10.0) < tolerance);
	}

	/* Next site. */
      }
    }
  }

  noise_free(&ns);
  cs_free(cs);

  return 0;
}
