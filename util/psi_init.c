/*****************************************************************************
 *
 *  psi_init.c
 *
 *  Initialise the electrokinetic quantities and write to file
 *  suitable for reading at time t = 0.
 *
 *  The user must ensure that the initial charges are conistent
 *  with the values for unit charge, valency, and so on, set in
 *  the input file at run time.
 *
 *  Overall electroneutrality (solid plus fluid) must also be
 *  ensured by the user.
 *
 *  Compilation:
 *  Compile the main code for serial execuation.
 *  Compile this file via
 *
 *    $(CC) -I../mpi_s -I../targetDP psi_init.c ../src/libludwig.a \
 *          ../mpi_s/libmpi.a ../targetDP/libtargetDP_C.a
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich (o.henrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
 
#include "../src/pe.h"
#include "../src/coords.h"
#include "../src/psi.h"
#include "../src/psi_stats.h"

int psi_init_gc_problem(psi_t * psi, map_t * map);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  /* Set system size parameters here */
  
  int ntotal[3] = {64, 4, 4};
  int grid[3] = {1, 1, 1};


  io_info_t * iohandler = NULL;
  psi_t * psi = NULL;
  map_t * map = NULL;

  MPI_Init(&argc, &argv);
  pe_init();
  coords_ntotal_set(ntotal);
  coords_init();

  psi_create(2, &psi);
  map_create(0, &map);
  assert(psi);
  assert(map);

  /* Create initial conditions */

  psi_init_gc_problem(psi, map);

  /* Print out the total charge of each species. The user must
   * ensure overall electroneutrality */

  psi_stats_info(psi);

  /* Write files: the output will be psi-00000000.001-001
   * and psi-porous.001-001  */

  psi_init_io_info(psi, grid, IO_FORMAT_BINARY, IO_FORMAT_BINARY);
  map_init_io_info(map, grid, IO_FORMAT_BINARY, IO_FORMAT_BINARY);

  psi_io_info(psi, &iohandler);
  io_write_data(iohandler, "psi-00000000", psi);
  map_io_info(map, &iohandler);
  io_write_data(iohandler, "psi-porous", map);

  map_free(map);
  psi_free(psi);
  coords_finish();
  pe_finalise();

  MPI_Finalize();

  return 0;
}

/*****************************************************************************
 *
 *  psi_init_gc_problem
 *
 *  This is a copy of the Gouy-Chapman problem intended for a 64x4x4
 *  system where we have walls at x = 1 and x = 64.
 *
 *  sigma is the surface charge density, and rho_el is the charge
 *  density in the fluid for one charge species. The other charge
 *  species is adjusted to ensure overall electroneutrality.
 *
 *****************************************************************************/

int psi_init_gc_problem(psi_t * obj, map_t * map) {

  int ic, jc, kc, index;
  int nlocal[3];
  double rho_w, rho_i;

  double rho_el = 0.001;
  double sigma  = 0.03125;

  assert(obj);
  assert(map);

  coords_nlocal(nlocal);

  /* wall surface charge density */
  rho_w = sigma;

  /* counter charge density */
  rho_i = rho_w * 2.0 *L(Y)*L(Z) / (L(Y)*L(Z)*(L(X) - 2.0));

  /* apply counter charges & electrolyte */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi_set(obj, index, 0.0);
	psi_rho_set(obj, index, 0, rho_el);
	psi_rho_set(obj, index, 1, rho_el + rho_i);

      }
    }
  }

  /* apply wall charges */
  if (cart_coords(X) == 0) {
    ic = 1;
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {
    ic = nlocal[X];
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);
	map_status_set(map, index, MAP_BOUNDARY);

	psi_rho_set(obj, index, 0, rho_w);
	psi_rho_set(obj, index, 1, 0.0);

      }
    }
  }

  return 0;
}
