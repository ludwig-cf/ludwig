/*****************************************************************************
 *
 *  coords.c
 *
 *  The physical coordinate system.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"

static int    n_total[3] = {64, 64, 64};
static int    periodic[3] = {1, 1, 1};

static double length[3];
static double lmin[3];

/*****************************************************************************
 *
 *  coords_init
 *
 *  Set the physical lattice size, and periodic flags (always periodic
 *  by default.
 *
 *****************************************************************************/

void coords_init() {

  int n;

  /* Look for "size" in the user input and set the lattice size. */

  n = RUN_get_int_parameter_vector("size", n_total);

  if (n == 0) info("(Default) ");
  info("Lattice size is (%d, %d, %d)\n", n_total[X], n_total[Y], n_total[Z]);

  for (n = 0; n < 3; n++) {
    length[n] = (double) n_total[n];
    lmin[n] = 0.5;
  }

  /* Look for the "periodicity" in the user input. (This is
   * not reported at the moment.) */

  n = RUN_get_int_parameter_vector("periodicity", periodic);

  return;
}

/*****************************************************************************
 *
 *  N_total, is_periodic, L, Lmin
 *
 *  "getter" functions
 *
 *****************************************************************************/

int N_total(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return n_total[dim];
}

int is_periodic(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return periodic[dim];
}

double L(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return length[dim];
}

double Lmin(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return lmin[dim];
}
