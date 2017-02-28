/*****************************************************************************
 *
 *  test_map.c
 *
 *  Unit test for map object.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "coords_field.h"
#include "map_s.h"

#include "test_coords_field.h"
#include "tests.h"

static int do_test1(pe_t * pe);
static int do_test2(pe_t * pe);
static int do_test_halo(pe_t * pe, int ndata);
static int do_test_io(pe_t * pe, int ndata, int io_format);

/*****************************************************************************
 *
 *  test_map_suite
 *
 *****************************************************************************/

int test_map_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  /*info("Map tests\n\n");*/

  do_test1(pe);
  do_test2(pe);
  do_test_halo(pe, 1);
  do_test_halo(pe, 2);

  do_test_io(pe, 0, IO_FORMAT_BINARY);
  do_test_io(pe, 0, IO_FORMAT_ASCII);
  do_test_io(pe, 2, IO_FORMAT_BINARY);
  do_test_io(pe, 2, IO_FORMAT_ASCII);

  pe_info(pe, "PASS     ./unit/test_map\n");
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  do_test1
 *
 *  Test the basic status functionality.
 *
 *****************************************************************************/

int do_test1(pe_t * pe) {

  int ndataref = 0;
  int ndata;
  int pm;

  int ic, jc, kc, index;
  int nlocal[3];
  int ntotal[3];

  int status;
  int vol;
  cs_t * cs = NULL;
  map_t * map = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);
  cs_nlocal(cs, nlocal);
  cs_ntotal(cs, ntotal);

  map_create(pe, cs, ndataref, &map);
  assert(map);

  map_ndata(map, &ndata);
  assert(ndata == ndataref);

  /* Porous media flag; not present by default. */
  map_pm(map, &pm);
  assert(pm == 0);
  map_pm_set(map, 1);
  map_pm(map, &pm);
  assert(pm == 1);

  /* Check default status is fluid everywhere. */
  /* This assertion is not necessary, although it cold be useful
   * to have fluid points associated with zero. */

  assert(MAP_FLUID == 0);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	map_status(map, index, &status);
	assert(status == MAP_FLUID);
      }
    }
  }

  /* Test volume (local, global) */

  vol = 0;
  map_volume_local(map, MAP_FLUID, &vol);
  assert(vol == nlocal[X]*nlocal[Y]*nlocal[Z]);

  vol = 0;
  map_volume_allreduce(map, MAP_FLUID, &vol);
  assert(vol == ntotal[X]*ntotal[Y]*ntotal[Z]);

  /* Test status set */

  index = cs_index(cs, 1, 1, 1);
  map_status_set(map, index, MAP_BOUNDARY);
  map_status(map, index, &status);
  assert(status == MAP_BOUNDARY);

  map_free(map);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test2
 *
 *  Check the data interface.
 *
 *****************************************************************************/

int do_test2(pe_t * pe) {

  int ndataref = 2;
  int ndata;
  int nlocal[3];
  int index = 0;

  double dataref[2] = {1.0, 2.0};
  double data[2];

  cs_t * cs = NULL;
  map_t * map = NULL;

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);
  cs_nlocal(cs, nlocal);

  map_create(pe, cs, ndataref, &map);
  assert(map);

  map_ndata(map, &ndata);
  assert(ndata == ndataref);

  map_data_set(map, index, dataref);
  map_data(map, index, data);
  assert(fabs(data[0] - dataref[0]) < DBL_EPSILON);
  assert(fabs(data[1] - dataref[1]) < DBL_EPSILON);

  map_free(map);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo
 *
 *****************************************************************************/

int do_test_halo(pe_t * pe, int ndata) {

  int nhalo;
  cs_t * cs = NULL;
  map_t * map  = NULL;

  assert(pe);
  assert(ndata > 0);

  cs_create(pe, &cs);
  cs_init(cs);
  cs_nhalo(cs, &nhalo);

  map_create(pe, cs, ndata, &map);
  assert(map);

  test_coords_field_set(cs, 1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_set(cs, map->ndata, map->data, MPI_DOUBLE, test_ref_double1);

  map_halo(map);

  test_coords_field_check(cs, nhalo, 1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_check(cs, nhalo, map->ndata, map->data, MPI_DOUBLE,
			  test_ref_double1);

  map_free(map);
  cs_free(cs);

  return 0;
}

/*****************************************************************************
 *
 *  do_test_io
 *
 *****************************************************************************/

static int do_test_io(pe_t * pe, int ndata, int io_format) {

  int grid[3];
  const char * filename = "map-io-test";
  MPI_Comm comm;

  cs_t * cs = NULL;
  map_t * map = NULL;
  io_info_t * iohandler = NULL;

  assert(pe);

  pe_mpi_comm(pe, &comm);

  grid[X] = 1;
  grid[Y] = 1;
  grid[Z] = 1;

  if (pe_mpi_size(pe) == 8) {
    grid[X] = 2;
    grid[Y] = 2;
    grid[Z] = 2;
  }

  cs_create(pe, &cs);
  cs_init(cs);
  map_create(pe, cs, ndata, &map);
  assert(map);

  map_init_io_info(map, grid, io_format, io_format);
  map_io_info(map, &iohandler);
  assert(iohandler);

  test_coords_field_set(cs, 1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_set(cs, ndata, map->data, MPI_DOUBLE, test_ref_double1);

  io_write_data(iohandler, filename, map);
  map_free(map);
  iohandler = NULL;
  map = NULL;
  MPI_Barrier(comm);

  /* Recreate, and read from file, and check. */

  map_create(pe, cs, ndata, &map);
  assert(map);

  map_init_io_info(map, grid, io_format, io_format);
  map_io_info(map, &iohandler);
  assert(iohandler);

  io_read_data(iohandler, filename, map);
  test_coords_field_check(cs, 0, 1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_check(cs, 0, ndata, map->data, MPI_DOUBLE, test_ref_double1);

  /* Wait before removing file(s) */
  MPI_Barrier(comm);
  io_remove(filename, iohandler);
  io_remove_metadata(iohandler, "map");

  map_free(map);
  cs_free(cs);

  return 0;
}
