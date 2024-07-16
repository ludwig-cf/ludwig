/*****************************************************************************
 *
 *  test_map.c
 *
 *  Unit tests for the map object.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "map.h"

int test_map_create(pe_t * pe, cs_t * cs);
int test_map_initialise(pe_t * pe, cs_t * cs);
int test_map_status(pe_t * ps, cs_t * cs);
int test_map_status_set(pe_t * pe, cs_t * cs);
int test_map_data(pe_t * pe, cs_t * cs);
int test_map_data_set(pe_t * pe, cs_t * cs);
int test_map_porous_media(pe_t * pe, cs_t * cs);
int test_map_volume_local(pe_t * pe, cs_t * cs);
int test_map_volume_allreduce(pe_t * pe, cs_t * cs);
int test_map_halo(pe_t * pe, cs_t * cs);

int test_map_read_buf(pe_t * pe, cs_t * cs);
int test_map_read_buf_ascii(pe_t * pe, cs_t * cs);
int test_map_write_buf(pe_t * pe, cs_t * cs);
int test_map_io_aggr_pack(pe_t * pe, cs_t * cs);
int test_map_io_write(pe_t * pe, cs_t * cs);
int test_map_io_read(pe_t * pe, cs_t * cs);

static int util_map_data_check_set(map_t * map);
static int util_map_data_check(const map_t * map, int nhalo);

/*****************************************************************************
 *
 *  test_map_suite
 *
 *****************************************************************************/

int test_map_suite(void) {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_map_create(pe, cs);
  test_map_initialise(pe, cs);
  test_map_status(pe, cs);
  test_map_status_set(pe, cs);
  test_map_data(pe, cs);
  test_map_data_set(pe, cs);
  test_map_porous_media(pe, cs);
  test_map_volume_local(pe, cs);
  test_map_volume_allreduce(pe, cs);
  test_map_halo(pe, cs);

  test_map_read_buf(pe, cs);
  test_map_read_buf_ascii(pe, cs);
  test_map_write_buf(pe, cs);
  test_map_io_aggr_pack(pe, cs);
  test_map_io_write(pe, cs); /* write ... */
  test_map_io_read(pe, cs);  /* .. and read in order please */

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_map_create
 *
 *****************************************************************************/

int test_map_create(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  int ndata = 2;
  map_options_t opts = map_options_ndata(ndata);
  map_t * map = NULL;

  ifail = map_create(pe, cs, &opts, &map);
  assert(ifail == 0);

  assert(map->ndata == ndata);
  assert(map->data);

  ifail = map_free(&map);
  assert(ifail == 0);
  assert(map == NULL);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_initialise
 *
 *****************************************************************************/

int test_map_initialise(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* No data */
  {
    map_options_t opts = map_options_default();
    map_t map = {0};

    ifail = map_initialise(pe, cs, &opts, &map);
    assert(ifail == 0);

    assert(map.nsite           == cs->param->nsites);
    assert(map.is_porous_media == 0);
    assert(map.ndata           == 0);
    assert(map.status          != NULL);
    assert(map.data            == NULL);
    assert(map.pe              == pe);
    assert(map.cs              == cs);

    assert(map.ascii.datatype  == MPI_CHAR);
    assert(map.ascii.datasize  == sizeof(char));
    assert(map.ascii.count     == 3 + 1);
    assert(map.ascii.endian    == io_endianness());

    assert(map.binary.datatype == MPI_CHAR);
    assert(map.binary.datasize == sizeof(char));
    assert(map.binary.count    == 1);
    assert(map.binary.endian   == io_endianness());

    assert(strcmp(map.filestub, "map") == 0);

    ifail = map_finalise(&map);
    assert(ifail == 0);
    assert(map.nsite == 0);
  }

  /* With data */
  {
    int ndata = 2;
    map_options_t opts = map_options_ndata(ndata);
    map_t map = {0};

    ifail = map_initialise(pe, cs, &opts, &map);

    assert(map.ndata == ndata);
    assert(map.data  != NULL);

    assert(map.ascii.datatype  == MPI_CHAR);
    assert(map.ascii.datasize  == sizeof(char));
    assert(map.ascii.count     == 3 + ndata*MAP_DATA_RECORD_LENGTH_ASCII + 1);
    assert(map.ascii.endian    == io_endianness());

    assert(map.binary.datatype == MPI_CHAR);
    assert(map.binary.datasize == sizeof(char));
    assert(map.binary.count    == 1 + ndata*(int)sizeof(double));
    assert(map.binary.endian   == io_endianness());

    assert(strcmp(map.filestub, "map") == 0);

    ifail = map_finalise(&map);
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_status
 *
 *****************************************************************************/

int test_map_status(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  int nlocal[3] = {0};
  map_options_t opts = map_options_default();
  map_t * map = NULL;

  /* The fluid setting must be zero at the moment */
  assert(MAP_FLUID == 0);

  cs_nlocal(cs, nlocal);
  map_create(pe, cs, &opts, &map);

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {

	int index = cs_index(cs, ic, jc, kc);
	int status = -1;
	map_status(map, index, &status);
	assert(status == MAP_FLUID);
	if (status != MAP_FLUID) ifail = -1;
      }
    }
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_status_set
 *
 *****************************************************************************/

int test_map_status_set(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  map_options_t opts = map_options_default();
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);

  {
    int index = cs_index(cs, 1, 1, 1);
    int status = -1;
    map_status_set(map, index, MAP_BOUNDARY);
    map_status(map, index, &status);
    assert(status == MAP_BOUNDARY);
  }

  {
    int index = cs_index(cs, 2, 2, 2);
    int status = -1;
    map_status_set(map, index, MAP_COLLOID);
    map_status(map, index, &status);
    assert(status == MAP_COLLOID);
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_data
 *
 *****************************************************************************/

int test_map_data(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  int ndata = 1;
  map_options_t opts = map_options_ndata(ndata);
  map_t map = {0};

  map_initialise(pe, cs, &opts, &map);
  assert(map.ndata == ndata);
  assert(map.data);

  /* A smoke test more than anything ... */
  {
    int index = 0;
    double datum = 0.0;
    map.data[index] = 1.0;
    map_data(&map, index, &datum);
    if (datum != 1.0) ifail = -1;
    assert(ifail == 0);
  }

  map_finalise(&map);

  return ifail;
}

/****************************************************************************
 *
 * test_map_data_set
 *
 *****************************************************************************/

int test_map_data_set(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  int ndata = 2;
  map_options_t opts = map_options_ndata(ndata);
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);

  /* Set some values */
  {
    int index = cs_index(cs, 1, 1, 1);
    double data[2] = {1.0, 2.0};
    double test[2] = {0};
    ifail = map_data_set(map, index, data);
    assert(ifail == 0);
    ifail = map_data(map, index, test);
    assert(ifail == 0);
    assert(fabs(data[0] - test[0]) < DBL_EPSILON);
    assert(fabs(data[1] - test[1]) < DBL_EPSILON);
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_porous_media
 *
 *****************************************************************************/

int test_map_porous_media(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  map_options_t opts = map_options_default();
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);
  assert(map->is_porous_media == 0);

  {
    int pm = -1;
    map_pm(map, &pm);
    assert(pm == 0);
    map_pm_set(map, 1);
    assert(map->is_porous_media == 1);
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_volume_local
 *
 *****************************************************************************/

int test_map_volume_local(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  map_options_t opts = map_options_default();
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);

  {
    int ivol = 0;
    int nlocal[3] = {0};

    cs_nlocal(cs, nlocal);

    map_volume_local(map, MAP_FLUID, &ivol);
    if (ivol != nlocal[X]*nlocal[Y]*nlocal[Z]) ifail = -1;
    assert(ifail == 0);
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_volume_allreduce
 *
 *****************************************************************************/

int test_map_volume_allreduce(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  map_options_t opts = map_options_default();
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);

  /* Check volume (all ranks) */
  {
    int ivol = 0;
    int ntotal[3] = {0};
    cs_ntotal(cs, ntotal);
    map_volume_allreduce(map, MAP_FLUID, &ivol);
    if (ivol != ntotal[X]*ntotal[Y]*ntotal[Z]) ifail = -1;
    assert(ifail == 0);
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_halo
 *
 *****************************************************************************/

int test_map_halo(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* Standard check */
  {
    map_options_t opts = map_options_ndata(2);
    map_t * map = NULL;

    map_create(pe, cs, &opts, &map);

    util_map_data_check_set(map);
    map_halo(map);

    ifail = util_map_data_check(map, map->cs->param->nhalo);
    assert(ifail == 0);

    map_free(&map);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_read_buf
 *
 *  It is appropriate to test map_read_buf() at the same time.
 *
 *****************************************************************************/

int test_map_read_buf(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* ndata = 0 */
  {
    map_options_t opts = map_options_default();
    char buf[BUFSIZ] = {0};
    map_t * map = NULL;

    map_create(pe, cs, &opts, &map);
    {
      int indexw = cs_index(cs, 3, 4, 5);
      int status = MAP_BOUNDARY;
      map_status_set(map, indexw, status);
      ifail = map_write_buf(map, indexw, buf);
      assert(ifail == 0);
    }
    {
      int indexr = cs_index(cs, 1, 1, 1);
      int status = -1;
      ifail = map_read_buf(map, indexr, buf);
      assert(ifail == 0);
      map_status(map, indexr, &status);
      assert(status == MAP_BOUNDARY);
    }
    map_free(&map);
  }

  /* ndata = 2 */
  {
    map_options_t opts = map_options_ndata(2);
    char buf[BUFSIZ] = {0};
    double data[2] = {2.0, 3.0};
    map_t * map = NULL;

    map_create(pe, cs, &opts, &map);
    {
      int indexw = cs_index(cs, 3, 4, 5);
      map_data_set(map, indexw, data);
      ifail = map_write_buf(map, indexw, buf);
      assert(ifail == 0);
    }
    {
      int indexr = cs_index(cs, 1, 1, 1);
      double dataread[2] = {0};
      ifail = map_read_buf(map, indexr, buf);
      assert(ifail == 0);
      map_data(map, indexr, dataread);
      assert(fabs(data[0] - dataread[0]) < DBL_EPSILON);
      assert(fabs(data[1] - dataread[1]) < DBL_EPSILON);
    }

    map_free(&map);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_read_buf_ascii
 *
 *****************************************************************************/

int test_map_read_buf_ascii(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  int status = MAP_COLLOID;
  double data[2] = {3.0, 5.0};

  /* Go straight to ndata = 2 */

  map_options_t opts = map_options_ndata(2);
  char buf[BUFSIZ] = {0};
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);
  {
    int indexw = cs_index(cs, 2, 3, 4);
    map_status_set(map, indexw, status);
    map_data_set(map, indexw, data);
    ifail = map_write_buf_ascii(map, indexw, buf);
    assert(ifail == 0);
  }

  {
    int indexr = cs_index(cs, 1, 1, 1);
    int statusread = MAP_FLUID;
    double dataread[2] = {0};
    ifail = map_read_buf_ascii(map, indexr, buf);
    assert(ifail == 0);
    map_status(map, indexr, &statusread);
    map_data(map, indexr, dataread);
    assert(statusread == status);
    assert(fabs(data[0] - dataread[0]) < DBL_EPSILON);
    assert(fabs(data[1] - dataread[1]) < DBL_EPSILON);
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_write_buf
 *
 *****************************************************************************/

int test_map_write_buf(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* ndata = 0 */
  {
    map_options_t opts = map_options_default();
    char buf[BUFSIZ] = {0};
    map_t * map = NULL;

    map_create(pe, cs, &opts, &map);
    {
      int indexw = cs_index(cs, 0, 0, 0); /* Start of buffer */
      int status = MAP_BOUNDARY;
      char s = {0};
      map_status_set(map, indexw, status);
      ifail = map_write_buf(map, indexw, buf);
      assert(ifail == 0);
      memcpy(&s, buf, sizeof(char));
      assert(s == MAP_BOUNDARY);
    }

    map_free(&map);
  }

  /* ndata = 2 */
  {
    map_options_t opts = map_options_ndata(2);
    char buf[BUFSIZ] = {0};
    double data[2] = {2.0, 3.0};
    map_t * map = NULL;

    map_create(pe, cs, &opts, &map);
    {
      int indexw = cs_index(cs, 0, 0, 0);
      int status = MAP_BOUNDARY;
      map_status_set(map, indexw, status);
      map_data_set(map, indexw, data);
      ifail = map_write_buf(map, indexw, buf);
      assert(ifail == 0);

    }

    map_free(&map);
  }


  return ifail;
}

/*****************************************************************************
 *
 *  test_map_io_aggr_pack
 *
 *****************************************************************************/

int test_map_io_aggr_pack(pe_t * pe, cs_t * cs) {

  int ifail = 0;
  int ndata = 2;
  map_options_t opts = map_options_ndata(ndata);
  map_t * map = NULL;

  map_create(pe, cs, &opts, &map);

  /* Default option is binary */
  {
    const io_metadata_t * meta = &map->output;
    io_aggregator_t buf = {0};

    assert(meta->cs     == cs);
    assert(meta->parent != MPI_COMM_NULL);
    assert(meta->comm   != MPI_COMM_NULL);

    /* Set some data values and check */
    io_aggregator_initialise(meta->element, meta->limits, &buf);

    util_map_data_check_set(map);
    ifail = map_io_aggr_pack(map, &buf);
    assert(ifail == 0);

    /* Read back the same values after clearing the originals */
    memset(map->status, 0, map->nsite*sizeof(char));
    memset(map->data, 0, (size_t) map->nsite*map->ndata*sizeof(double));

    ifail = map_io_aggr_unpack(map, &buf);
    assert(ifail == 0);
    ifail = util_map_data_check(map, 0);
    assert(ifail == 0);

    io_aggregator_finalise(&buf);
  }

  map_free(&map);

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_io_write
 *
 *****************************************************************************/

int test_map_io_write(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* ASCII */
  {
    map_options_t opts = map_options_ndata(1);
    map_t * map = NULL;

    opts.filestub                = "map-ascii";
    opts.iodata.output.mode      = IO_MODE_MPIIO;
    opts.iodata.output.iorformat = IO_RECORD_ASCII;

    map_create(pe, cs, &opts, &map);

    util_map_data_check_set(map);
    ifail = map_io_write(map, 0);
    assert(ifail == 0);

    map_free(&map);
  }

  /* BINARY */
  {
    map_options_t opts = map_options_default();
    map_t * map = NULL;

    opts.filestub                = "map-binary";
    opts.iodata.output.mode      = IO_MODE_MPIIO;
    opts.iodata.output.iorformat = IO_RECORD_BINARY;

    map_create(pe, cs, &opts, &map);

    util_map_data_check_set(map);
    ifail = map_io_write(map, 0);
    assert(ifail == 0);

    map_free(&map);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_map_io_read
 *
 *****************************************************************************/

int test_map_io_read(pe_t * pe, cs_t * cs) {

  int ifail = 0;

  /* ASCII; consistent with test_map_io_write() */
  {
    map_options_t opts = map_options_ndata(1);
    map_t * map = NULL;

    opts.filestub               = "map-ascii";
    opts.iodata.input.mode      = IO_MODE_MPIIO;
    opts.iodata.input.iorformat = IO_RECORD_ASCII;

    map_create(pe, cs, &opts, &map);

    ifail = map_io_read(map, 0);
    assert(ifail == 0);
    ifail = util_map_data_check(map, 0);
    assert(ifail == 0);

    map_free(&map);
  }

  /* BINARY; again, as per test_map_io_write() */
  {
    map_options_t opts = map_options_default();
    map_t * map = NULL;

    opts.filestub               = "map-binary";
    opts.iodata.input.mode      = IO_MODE_MPIIO;
    opts.iodata.input.iorformat = IO_RECORD_BINARY;

    map_create(pe, cs, &opts, &map);

    ifail = map_io_read(map, 0);
    assert(ifail == 0);
    ifail = util_map_data_check(map, 0);
    assert(ifail == 0);

    map_free(&map);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_map_unique_status
 *
 *  Return a unique value for status based on global (ic, jc, kc).
 *
 *  As status is currently implemented as "char", there is limited
 *  scope for a truly unique value here...
 *
 *  Returned as int and not valid map_status enum!
 *
 *****************************************************************************/

static int util_map_unique_status(const map_t * map, int ic, int jc, int kc) {

  int status = -1;
  int ntotal[3] = {0};
  int offset[3] = {0};

  assert(map);

  cs_ntotal(map->cs, ntotal);
  cs_nlocal_offset(map->cs, offset);

  {
    int strz = 1;
    int stry = strz*ntotal[Z];
    int strx = stry*ntotal[Y];
    int ix   = offset[X] + ic;
    int iy   = offset[Y] + jc;
    int iz   = offset[Z] + kc;
    if (ix <= 0)         ix += ntotal[X];
    if (ix >  ntotal[X]) ix -= ntotal[X];
    if (iy <= 0)         iy += ntotal[Y];
    if (iy >  ntotal[Y]) iy -= ntotal[Y];
    if (iz <= 0)         iz += ntotal[Z];
    if (iz >  ntotal[Z]) iz -= ntotal[Z];
    status = strx*ix + stry*iy + strz*iz;
    status = status % CHAR_MAX;
  }

  return status;
}

/*****************************************************************************
 *
 *  util_map_unique_data
 *
 *****************************************************************************/

static int util_map_unique_data(const map_t * map, int ic, int jc, int kc,
				double * data) {
  int ntotal[3] = {0};
  int offset[3] = {0};

  assert(map);
  assert(data);

  cs_ntotal(map->cs, ntotal);
  cs_nlocal_offset(map->cs, offset);

  {
    int strz = 1;
    int stry = strz*ntotal[Z];
    int strx = stry*ntotal[Y];
    int sdat = strx*(map->ndata - 1);
    int ix   = offset[X] + ic;
    int iy   = offset[Y] + jc;
    int iz   = offset[Z] + kc;
    if (ix <= 0)         ix += ntotal[X];
    if (ix >  ntotal[X]) ix -= ntotal[X];
    if (iy <= 0)         iy += ntotal[Y];
    if (iy >  ntotal[Y]) iy -= ntotal[Y];
    if (iz <= 0)         iz += ntotal[Z];
    if (iz >  ntotal[Z]) iz -= ntotal[Z];
    for (int n = 0; n < map->ndata; n++) {
      data[n] = sdat*n + strx*ix + stry*iy + strz*iz;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_map_data_check_set
 *
 *****************************************************************************/

static int util_map_data_check_set(map_t * map) {

  int nlocal[3] = {0};

  assert(map);
  assert(map->ndata <= 2);

  cs_nlocal(map->cs, nlocal);

  /* Set some values ...  */

  for (int ic = 1; ic <= nlocal[X]; ic++) {
    for (int jc = 1; jc <= nlocal[Y]; jc++) {
      for (int kc = 1; kc <= nlocal[Z]; kc++) {
	int index = cs_index(map->cs, ic, jc, kc);
	double data[2] = {0};

	/* Bypass map_status_set() as it has assertion on value .. */
	map->status[index] = util_map_unique_status(map, ic, jc, kc);

	util_map_unique_data(map, ic, jc, kc, data);
	map_data_set(map, index, data);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_map_data_check
 *
 *****************************************************************************/

static int util_map_data_check(const map_t * map, int nhalo) {

  int ifail = 0;
  int nlocal[3] = {0};

  cs_nlocal(map->cs, nlocal);

  /* Check consistent with the values set in util_map_data_check_set() */

  for (int ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (int jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (int kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	int status = -1;
	int index = cs_index(map->cs, ic, jc, kc);
	double data[2] = {0};
	double ref[2] = {0};

	map_status(map, index, &status);
	map_data(map, index, data);

	if (status != util_map_unique_status(map, ic, jc, kc)) ifail = -1;

	util_map_unique_data(map, ic, jc, kc, ref);

	for (int n = 0; n < map->ndata; n++) {
	  if (fabs(ref[n] - data[n]) > DBL_EPSILON) ifail = -1;
	}
      }
    }
  }

  return ifail;
}
