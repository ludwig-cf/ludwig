/*****************************************************************************
 *
 *  test_field.c
 *
 *  Unit test for field structure.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "leesedwards.h"
#include "field.h"

#include "test_coords_field.h"
#include "tests.h"

static int do_test0(pe_t * pe);
static int do_test1(pe_t * pe);
static int do_test3(pe_t * pe);
static int do_test5(pe_t * pe);
static int do_test_io(pe_t * pe, int nf, int io_format_in, int io_format_out);
static int test_field_halo(cs_t * cs, field_t * phi);

int do_test_device1(pe_t * pe);
int test_field_halo_create(pe_t * pe);
int test_field_write_buf(pe_t * pe);
int test_field_write_buf_ascii(pe_t * pe);
int test_field_io_aggr_pack(pe_t * pe);

int test_field_io_read_write(pe_t * pe);
int test_field_io_write(pe_t * pe, cs_t * cs, const field_options_t * opts);
int test_field_io_read(pe_t * pe, cs_t * cs, const field_options_t * opts);

int util_field_data_check(field_t * field);
int util_field_data_check_set(field_t * field);


__global__ void do_test_field_kernel1(field_t * phi);

/*****************************************************************************
 *
 *  test_field_suite
 *
 *****************************************************************************/

int test_field_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  do_test0(pe);
  do_test1(pe);
  do_test3(pe);
  do_test5(pe);
  do_test_device1(pe);

  do_test_io(pe, 1, IO_FORMAT_ASCII_SERIAL, IO_FORMAT_ASCII);
  do_test_io(pe, 1, IO_FORMAT_BINARY_SERIAL, IO_FORMAT_BINARY);
  do_test_io(pe, 5, IO_FORMAT_ASCII_SERIAL, IO_FORMAT_ASCII);
  do_test_io(pe, 5, IO_FORMAT_BINARY_SERIAL, IO_FORMAT_BINARY);

  test_field_halo_create(pe);

  test_field_write_buf(pe);
  test_field_write_buf_ascii(pe);
  test_field_io_aggr_pack(pe);
  test_field_io_read_write(pe);

  pe_info(pe, "PASS     ./unit/test_field\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_io_read_write
 *
 *  Driver for individual i/o routine tests. We actually do write then read.
 *
 *****************************************************************************/

int test_field_io_read_write(pe_t * pe) {

  int ntotal[3] = {32, 16, 8};
  MPI_Comm comm = MPI_COMM_NULL;
  cs_t * cs = NULL;

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);
  cs_cart_comm(cs, &comm);

  /* ASCII */
  {
    io_options_t io = io_options_with_format(IO_MODE_MPIIO, IO_RECORD_ASCII);
    field_options_t opts = field_options_ndata_nhalo(3, 0);
    opts.iodata.input  = io;
    opts.iodata.output = io;

    test_field_io_write(pe, cs, &opts);
    test_field_io_read(pe, cs, &opts);

    MPI_Barrier(comm); /* Make sure we finish before any further action */
  }

  /* Binary (default) */
  {
    io_options_t io = io_options_with_format(IO_MODE_MPIIO, IO_RECORD_BINARY);
    field_options_t opts = field_options_ndata_nhalo(5, 2);
    opts.iodata.input  = io;
    opts.iodata.output = io;

    test_field_io_write(pe, cs, &opts);
    test_field_io_read(pe, cs, &opts);

    MPI_Barrier(comm); /* Make sure we finish before any further action */
  }

  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test0
 *
 *  Small system test.
 *
 *****************************************************************************/

static int do_test0(pe_t * pe) {

  int nfref = 1;
  int nhalo = 2;
  int ntotal[3] = {8, 8, 8};

  cs_t * cs = NULL;
  field_t * phi = NULL;
  field_options_t opts = field_options_ndata_nhalo(nfref, nhalo);

  assert(pe);
  
  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);

  field_create(pe, cs, NULL, "phi", &opts, &phi);

  /* Halo */
  test_field_halo(cs, phi);

  field_free(phi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *  Scalar order parameter.
 *
 *****************************************************************************/

int do_test1(pe_t * pe) {

  int nfref = 1;
  int nf;
  int nhalo = 2;
  int index = 1;
  double ref;
  double value;

  cs_t * cs = NULL;
  field_t * phi = NULL;
  field_options_t opts = field_options_ndata_nhalo(nfref, nhalo);

  assert(pe);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);

  field_create(pe, cs, NULL, "phi", &opts, &phi);
  assert(phi);

  field_nf(phi, &nf);
  assert(nf == nfref);

  ref = 1.0;
  field_scalar_set(phi, index, ref);
  field_scalar(phi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  ref = -1.0;
  field_scalar_array_set(phi, index, &ref);
  field_scalar_array(phi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  ref = 1.0/3.0;
  field_scalar_set(phi, index, ref);
  field_scalar(phi, index, &value);
  assert(fabs(value - ref) < DBL_EPSILON);

  /* Halo */
  test_field_halo(cs, phi);

  field_free(phi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_device1
 *
 *****************************************************************************/

int do_test_device1(pe_t * pe) {

  int nfref = 1;
  int nf;
  int nhalo = 2;
  dim3 nblk, ntpb;

  cs_t * cs = NULL;
  field_t * phi = NULL;
  field_options_t opts = field_options_ndata_nhalo(nfref, nhalo);

  assert(pe);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);

  field_create(pe, cs, NULL, "phi", &opts, &phi);
  assert(phi);

  field_nf(phi, &nf);
  assert(nf == nfref);

  kernel_launch_param(1, &nblk, &ntpb);
  ntpb.x = 1;

  tdpLaunchKernel(do_test_field_kernel1, nblk, ntpb, 0, 0, phi->target);
  tdpDeviceSynchronize();

  field_free(phi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_field_kernel1
 *
 *****************************************************************************/

__global__ void do_test_field_kernel1(field_t * phi) {

  int nf;
  int index = 1;
  int nsites;
  double q;
  double qref = 1.2;

  assert(phi);

  field_nf(phi, &nf);
  assert(nf == 1);

  cs_nsites(phi->cs, &nsites);
  assert(phi->nsites == nsites);

  field_scalar_set(phi, index, qref);
  field_scalar(phi, index, &q);
  assert(fabs(q - qref) < DBL_EPSILON);

  return;
}

/*****************************************************************************
 *
 *  do_test3
 *
 *  Vector order parameter.
 *
 *****************************************************************************/

static int do_test3(pe_t * pe) {

  int nfref = 3;
  int nf;
  int nhalo = 1;
  int index = 1;
  double ref[3] = {1.0, 2.0, 3.0};
  double value[3];
  double array[3];

  cs_t * cs = NULL;
  field_t * phi = NULL;
  field_options_t opts = field_options_ndata_nhalo(nfref, nhalo);

  assert(pe);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);

  field_create(pe, cs, NULL, "p", &opts, &phi);
  assert(phi);

  field_nf(phi, &nf);
  assert(nf == nfref);

  field_vector_set(phi, index, ref);
  field_vector(phi, index, value);
  assert(fabs(value[0] - ref[0]) < DBL_EPSILON);
  assert(fabs(value[1] - ref[1]) < DBL_EPSILON);
  assert(fabs(value[2] - ref[2]) < DBL_EPSILON);

  field_scalar_array(phi, index, array);
  assert(fabs(array[0] - ref[0]) < DBL_EPSILON);
  assert(fabs(array[1] - ref[1]) < DBL_EPSILON);
  assert(fabs(array[2] - ref[2]) < DBL_EPSILON);

  /* Halo */
  test_field_halo(cs, phi);

  field_free(phi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test5
 *
 *  Tensor order parameter.
 *
 *****************************************************************************/

static int do_test5(pe_t * pe) {

  int nfref = 5;
  int nf;
  int nhalo = 1;
  int index = 1;
  double qref[3][3] = {{1.0, 2.0, 3.0}, {2.0, 4.0, 5.0}, {3.0, 5.0, -5.0}};
  double qvalue[3][3];
  double array[NQAB];

  cs_t * cs = NULL;
  field_t * phi = NULL;
  field_options_t opts = field_options_ndata_nhalo(nfref, nhalo);

  assert(pe);

  cs_create(pe, &cs);
  cs_nhalo_set(cs, nhalo);
  cs_init(cs);

  field_create(pe, cs, NULL, "q", &opts, &phi);
  assert(phi);

  field_nf(phi, &nf);
  assert(nf == nfref);

  field_tensor_set(phi, index, qref);
  field_tensor(phi, index, qvalue);
  assert(fabs(qvalue[X][X] - qref[X][X]) < DBL_EPSILON);
  assert(fabs(qvalue[X][Y] - qref[X][Y]) < DBL_EPSILON);
  assert(fabs(qvalue[X][Z] - qref[X][Z]) < DBL_EPSILON);
  assert(fabs(qvalue[Y][X] - qref[Y][X]) < DBL_EPSILON);
  assert(fabs(qvalue[Y][Y] - qref[Y][Y]) < DBL_EPSILON);
  assert(fabs(qvalue[Y][Z] - qref[Y][Z]) < DBL_EPSILON);
  assert(fabs(qvalue[Z][X] - qref[Z][X]) < DBL_EPSILON);
  assert(fabs(qvalue[Z][Y] - qref[Z][Y]) < DBL_EPSILON);
  assert(fabs(qvalue[Z][Z] - qref[Z][Z]) < DBL_EPSILON);

  /* This is the upper trianle minus the ZZ component */

  field_scalar_array(phi, index, array);
  assert(fabs(array[XX] - qref[X][X]) < DBL_EPSILON);
  assert(fabs(array[XY] - qref[X][Y]) < DBL_EPSILON);
  assert(fabs(array[XZ] - qref[X][Z]) < DBL_EPSILON);
  assert(fabs(array[YY] - qref[Y][Y]) < DBL_EPSILON);
  assert(fabs(array[YZ] - qref[Y][Z]) < DBL_EPSILON);

  /* Halo */
  test_field_halo(cs, phi);

  field_free(phi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_halo
 *
 *****************************************************************************/

static int test_field_halo(cs_t * cs, field_t * phi) {

  assert(phi);
  
  test_coords_field_set(cs, phi->nf, phi->data, MPI_DOUBLE, test_ref_double1);
  field_memcpy(phi, tdpMemcpyHostToDevice);
 
  field_halo_swap(phi, FIELD_HALO_TARGET);

  field_memcpy(phi, tdpMemcpyDeviceToHost);
  test_coords_field_check(cs, phi->nhcomm, phi->nf, phi->data, MPI_DOUBLE,
			  test_ref_double1);

  return 0;
} 

/*****************************************************************************
 *
 *  do_test_io
 *
 *****************************************************************************/

static int do_test_io(pe_t * pe, int nf, int io_format_in, int io_format_out) {

  int ntotal[3] = {16, 16, 8};
  int grid[3] = {1, 1, 1};
  int nhalo;
  const char * filename = "phi-test-io";

  MPI_Comm comm;

  cs_t * cs = NULL;
  field_t * phi = NULL;
  io_info_t * iohandler = NULL;
  field_options_t opts = field_options_default();

  assert(pe);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_init(cs);
  cs_nhalo(cs, &nhalo);
  cs_cart_comm(cs, &comm);

  opts.ndata = nf;
  opts.nhcomm = nhalo;
  field_create(pe, cs, NULL, "phi-test", &opts, &phi);

  field_init_io_info(phi, grid, io_format_in, io_format_out);

  test_coords_field_set(cs, nf, phi->data, MPI_DOUBLE, test_ref_double1);

  field_io_info(phi, &iohandler);
  assert(iohandler);

  io_write_data(iohandler, filename, phi);

  field_free(phi);
  MPI_Barrier(comm);

  field_create(pe, cs, NULL, "phi-test", &opts,&phi);
  field_init_io_info(phi, grid, io_format_in, io_format_out);

  field_io_info(phi, &iohandler);
  assert(iohandler);

  /* Make sure the input format is handled correctly. */
  io_info_format_in_set(iohandler, io_format_in);
  io_info_single_file_set(iohandler);

  io_read_data(iohandler, filename, phi);

  field_halo(phi);
  test_coords_field_check(cs, 0, nf, phi->data, MPI_DOUBLE, test_ref_double1);

  MPI_Barrier(comm);

  io_remove(filename, iohandler);
  io_remove_metadata(iohandler, "phi-test");

  field_free(phi);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_halo_create
 *
 *****************************************************************************/

int test_field_halo_create(pe_t * pe) {

  cs_t * cs = NULL;
  field_t * field = NULL;
  field_options_t opts = field_options_default();

  field_halo_t h = {0};

  {
    int nhalo = 2;
    int ntotal[3] = {32, 16, 8};
    cs_create(pe, &cs);
    cs_nhalo_set(cs, nhalo);
    cs_ntotal_set(cs, ntotal);
    cs_init(cs);
  }

  opts.ndata = 2;
  opts.nhcomm = 2;
  field_create(pe, cs, NULL, "halotest", &opts, &field);

  field_halo_create(field, &h);

  test_coords_field_set(cs, 2, field->data, MPI_DOUBLE, test_ref_double1);
  field_halo_post(field, &h);
  field_halo_wait(field, &h);
  test_coords_field_check(cs, 2, 2, field->data, MPI_DOUBLE, test_ref_double1);

  field_halo_free(&h);

  field_free(field);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_write_buf
 *
 *  It is convenient to test field_read_buf() at the same time.
 *
 *****************************************************************************/

int test_field_write_buf(pe_t * pe) {

  int nf = 3; /* Test field */

  cs_t * cs = NULL;
  field_t * field = NULL;
  field_options_t options = field_options_ndata_nhalo(nf, 1);

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);
  field_create(pe, cs, NULL, "test_write_buf", &options, &field);

  {
    double array[3] = {1.0, 2.0, 3.0};
    char buf[3*sizeof(double)] = {0};
    int index = cs_index(cs, 2, 3, 4);

    field_scalar_array_set(field, index, array);
    field_write_buf(field, index, buf);

    {
      double val[3] = {0};

      field_read_buf(field, index + 1, buf);
      field_scalar_array(field, index + 1, val);

      assert(fabs(val[0] - array[0]) < DBL_EPSILON);
      assert(fabs(val[1] - array[1]) < DBL_EPSILON);
      assert(fabs(val[2] - array[2]) < DBL_EPSILON);
    }
  }

  field_free(field);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_write_buf_ascii
 *
 *  It is convenient to test field_read_buf_ascii() at the same time.
 *
 *****************************************************************************/

int test_field_write_buf_ascii(pe_t * pe) {

  int nf = 5; /* Test field */

  cs_t * cs = NULL;
  field_t * field = NULL;
  field_options_t options = field_options_ndata_nhalo(nf, 1);

  cs_create(pe, &cs);
  cs_init(cs);
  field_create(pe, cs, NULL, "test_field_write_buf_ascii", &options, &field);

  {
    double array[5] = {1.0, 3.0, 2.0, -4.0, -5.0};
    char buf[BUFSIZ] = {0};
    int index = cs_index(cs, 1, 2, 3);

    field_scalar_array_set(field, index, array);
    field_write_buf_ascii(field, index, buf);
    assert(strnlen(buf, BUFSIZ) == (23*nf + 1)*sizeof(char));

    /* Put the values back in a different location and check */

    {
      double val[5] = {0};
      field_read_buf_ascii(field, index + 1, buf);
      field_scalar_array(field, index + 1, val);

      assert((val[0] - array[0]) < DBL_EPSILON);
      assert((val[1] - array[1]) < DBL_EPSILON);
      assert((val[2] - array[2]) < DBL_EPSILON);
      assert((val[3] - array[3]) < DBL_EPSILON);
      assert((val[4] - array[4]) < DBL_EPSILON);      
    }
  }

  field_free(field);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_io_aggr_pack
 *
 *  It is convenient to test field_io_aggr_unpack() at the same time.
 *
 *****************************************************************************/

int test_field_io_aggr_pack(pe_t * pe) {

  int nf = 5; /* Test field */

  cs_t * cs = NULL;
  field_t * field = NULL;
  field_options_t options = field_options_ndata_nhalo(nf, 1);

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);
  field_create(pe, cs, NULL, "test_field_io_aggr_pack", &options, &field);

  /* Default options is binary (use output metadata) */
  {
    const io_metadata_t * meta = &field->iometadata_out;
    io_aggregator_t buf = {0};

    io_aggregator_initialise(meta->element, meta->limits, &buf);

    util_field_data_check_set(field);
    field_io_aggr_pack(field, &buf);

    /* Are the values in the buffer correct? */
    /* Clear existing values and unpack. */

    memset(field->data, 0, sizeof(double)*field->nsites*field->nf);

    field_io_aggr_unpack(field, &buf);
    util_field_data_check(field);

    io_aggregator_finalise(&buf);
  }

  /* Repeat for ASCII */

  field_free(field);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_io_write
 *
 *****************************************************************************/

int test_field_io_write(pe_t * pe, cs_t * cs, const field_options_t * opts) {

  field_t * field = NULL;

  assert(pe);
  assert(cs);
  assert(opts);

  /* Establish data and test values. */

  field_create(pe, cs, NULL, "test-field-io", opts, &field);

  util_field_data_check_set(field);

  /* Write */

  {
    int it = 0;
    io_event_t event = {0};
    field_io_write(field, it, &event);
  }

  field_free(field);

  return 0;
}

/*****************************************************************************
 *
 *  test_field_io_read
 *
 *  This needs to be co-ordinated with test_field_io_write() above.
 *
 *****************************************************************************/

int test_field_io_read(pe_t * pe, cs_t * cs, const field_options_t * opts) {

  field_t * field = NULL;

  assert(pe);
  assert(cs);
  assert(opts);

  field_create(pe, cs, NULL, "test-field-io", opts, &field);

  {
    int it = 0;  /* matches time step zero in test_field_io_write() above */
    io_event_t event = {0};

    field_io_read(field, it, &event);

    util_field_data_check(field);
  }

  field_free(field);

  return 0;
}

/*****************************************************************************
 *
 *  field_unique_value
 *
 *  Set a unique value based on global position.
 *
 *****************************************************************************/

int64_t field_unique_value(field_t * f, int ic, int jc, int kc, int n) {

  int64_t ival = INT64_MIN;

  int ntotal[3] = {0};
  int nlocal[3] = {0};
  int noffset[3] = {0};

  assert(f);

  cs_ntotal(f->cs, ntotal);
  cs_nlocal_offset(f->cs, noffset);
  cs_nlocal(f->cs, nlocal);

  {
    int strz = 1;
    int stry = strz*ntotal[Z];
    int strx = stry*ntotal[Y];
    int nstr = strx*f->nf;
    int ix = noffset[X] + ic;
    int iy = noffset[Y] + jc;
    int iz = noffset[Z] + kc;
    ival = nstr*n + strx*ix + stry*iy + strz*iz;
  }

  return ival;
}

/*****************************************************************************
 *
 *  util_field_data_check_set
 *
 *****************************************************************************/

int util_field_data_check_set(field_t * field) {

  int nlocal[3] = {0};

  assert(field);

  cs_nlocal(field->cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(field->cs, ic, jc, kc);
	for (int n = 0; n < field->nf; n++) {
	  int faddr = addr_rank1(field->nsites, field->nf, index, n);
	  field->data[faddr] = 1.0*field_unique_value(field, ic, jc, kc, n);
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_field_data_check
 *
 *****************************************************************************/

int util_field_data_check(field_t * field) {

  int ifail = 0;
  int nlocal[3] = {0};

  assert(field);

  cs_nlocal(field->cs, nlocal);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(field->cs, ic, jc, kc);
	for (int n = 0; n < field->nf; n++) {
	  int faddr = addr_rank1(field->nsites, field->nf, index, n);
	  double fval = 1.0*field_unique_value(field, ic, jc, kc, n);
	  assert(fabs(field->data[faddr] - fval) < DBL_EPSILON);
	  if (fabs(field->data[faddr] - fval) > DBL_EPSILON) ifail += 1;
	}
      }
    }
  }

  return ifail;
}
