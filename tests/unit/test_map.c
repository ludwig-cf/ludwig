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
 *  (c) 2012 The University of Edinburgh
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

static int do_test1(void);
static int do_test2(void);
static int do_test_halo(int ndata);
static int do_test_io(int ndata, int io_format);

/*****************************************************************************
 *
 *  test_map_suite
 *
 *****************************************************************************/

int test_map_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /*info("Map tests\n\n");*/

  do_test1();
  do_test2();
  do_test_halo(1);
  do_test_halo(2);

  do_test_io(0, IO_FORMAT_BINARY);
  do_test_io(0, IO_FORMAT_ASCII);
  do_test_io(2, IO_FORMAT_BINARY);
  do_test_io(2, IO_FORMAT_ASCII);

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

int do_test1(void) {

  int ndataref = 0;
  int ndata;
  int pm;

  int ic, jc, kc, index;
  int nlocal[3];
  int ntotal[3];

  int status;
  int vol;
  map_t * map = NULL;

  coords_init();
  coords_nlocal(nlocal);
  coords_ntotal(ntotal);

  map_create(ndataref, &map);
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

	index = coords_index(ic, jc, kc);
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

  index = coords_index(1, 1, 1);
  map_status_set(map, index, MAP_BOUNDARY);
  map_status(map, index, &status);
  assert(status == MAP_BOUNDARY);

  map_free(map);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test2
 *
 *  Check the data interface.
 *
 *****************************************************************************/

int do_test2(void) {

  int ndataref = 2;
  int ndata;
  int nlocal[3];
  int index = 0;

  double dataref[2] = {1.0, 2.0};
  double data[2];

  map_t * map = NULL;

  coords_init();
  coords_nlocal(nlocal);

  map_create(ndataref, &map);
  assert(map);

  map_ndata(map, &ndata);
  assert(ndata == ndataref);

  map_data_set(map, index, dataref);
  map_data(map, index, data);
  assert(fabs(data[0] - dataref[0]) < DBL_EPSILON);
  assert(fabs(data[1] - dataref[1]) < DBL_EPSILON);

  map_free(map);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_halo
 *
 *****************************************************************************/

int do_test_halo(int ndata) {

  int nhalo;
  map_t * map  = NULL;

  assert(ndata > 0);

  coords_init();
  nhalo = coords_nhalo();

  map_create(ndata, &map);
  assert(map);

  test_coords_field_set(1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_set(map->ndata, map->data, MPI_DOUBLE, test_ref_double1);

  map_halo(map);

  test_coords_field_check(nhalo, 1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_check(nhalo, map->ndata, map->data, MPI_DOUBLE,
			  test_ref_double1);

  map_free(map);
  coords_finish();

  return 0;
}

/*****************************************************************************
 *
 *  do_test_io
 *
 *****************************************************************************/

static int do_test_io(int ndata, int io_format) {

  int grid[3] = {1, 1, 1};
  const char * filename = "map-io-test";

  map_t * map = NULL;
  io_info_t * iohandler = NULL;

  if (pe_size() == 8) {
    grid[X] = 2;
    grid[Y] = 2;
    grid[Z] = 2;
  }

  coords_init();
  map_create(ndata, &map);
  assert(map);

  map_init_io_info(map, grid, io_format, io_format);
  map_io_info(map, &iohandler);
  assert(iohandler);

  test_coords_field_set(1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_set(ndata, map->data, MPI_DOUBLE, test_ref_double1);

  io_write_data(iohandler, filename, map);
  map_free(map);
  iohandler = NULL;
  map = NULL;
  MPI_Barrier(pe_comm());

  /* Recreate, and read from file, and check. */

  map_create(ndata, &map);
  assert(map);

  map_init_io_info(map, grid, io_format, io_format);
  map_io_info(map, &iohandler);
  assert(iohandler);

  io_read_data(iohandler, filename, map);
  test_coords_field_check(0, 1, map->status, MPI_CHAR, test_ref_char1);
  test_coords_field_check(0, ndata, map->data, MPI_DOUBLE, test_ref_double1);

  /* Wait before removing file(s) */
  MPI_Barrier(pe_comm());
  io_remove(filename, iohandler);
  io_remove_metadata(iohandler, "map");

  map_free(map);
  coords_finish();

  return 0;
}
