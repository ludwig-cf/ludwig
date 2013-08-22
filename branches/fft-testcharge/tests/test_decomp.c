/*****************************************************************************
 *
 *  test_decomp.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Ruairi Short (R.Short@sms.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "psi_s.h"
#include "psi.h"
#include "coords.h"
#include "decomp.h"


/*****************************************************************************
 *
 *  hash
 *
 *  Generate a unique number from the global coordinates.
 *
 *****************************************************************************/

static double hash(double global_coord[]) {
  return N_total(Y)*N_total(Z)*global_coord[0] + N_total(Z)*global_coord[1] + global_coord[2];
}


/*****************************************************************************
 *
 *  main
 *
 *  Test the decomposition switching algorithm
 * 
 *  Give each lattice point a unique value from its global coordinate.
 *  Transform to the pencil grid and test this value.
 *  Transform back and test again
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  char inputfile[FILENAME_MAX] = "input";
  MPI_Comm ludwig_comm       = cart_comm();
  double global_coord[3]     = {0, 0, 0};
  int nlocal_offset[3]       = {0, 0, 0};
  int nlocal[3]              = {0, 0, 0};
  int isize[3]               = {0, 0, 0};
  int istart[3]              = {0, 0, 0};
  int rank                   = cart_rank();
  int num_procs              = pe_size();
  int ip                     = 1;
  int i, j, k;

  MPI_Init(&argc, &argv);

  pe_init();

  info("Testing decomposition switching\n");

  coords_init();
  decomp_init();


  /*create two psi_t objects to compare at the end*/
  psi_t *start_psi = NULL;
  psi_t *end_psi = NULL;

  psi_create(2, &start_psi);
  psi_create(2, &end_psi);
  assert(start_psi);
  assert(end_psi);

  coords_nlocal(nlocal);
  coords_nlocal_offset(nlocal_offset);

  /* initialise elements of send_array to hash evaluation of global coordinates */
  for(i=1; i<=nlocal[X]; i++) {
    global_coord[X] = (nlocal_offset[X] + i - 1); 
    for(j=1; j<=nlocal[Y]; j++) {
      global_coord[Y] = (nlocal_offset[Y] + j - 1); 
      for(k=1; k<=nlocal[Z]; k++) {
        global_coord[Z] = (nlocal_offset[Z] + k - 1); 
        start_psi->psi[coords_index(i,j,k)] = hash(global_coord);
      }
    }
  }

  ip = 1;
  decomp_pencil_starts(istart, ip);
  decomp_pencil_sizes(isize, ip);

  double *recv_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));

  decomp_cart_to_pencil(start_psi->psi, recv_array);


  /*test global coordinates of all points are correct */
  for(i=0; i<isize[Z]; i++) {
    global_coord[X] = istart[2-X]-1 + i; 
    for(j=0; j<isize[Y]; j++) {
      global_coord[Y] = istart[2-Y]-1 + j; 
      for(k=0; k<isize[X]; k++) {
        global_coord[Z] = k; 
        assert(recv_array[index_3d_f(i,j,k,isize)] - hash(global_coord) == 0);
      }
    }
  }

  decomp_pencil_to_cart(recv_array, end_psi->psi);

  /*test final array is the same as the original*/
  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        assert(start_psi->psi[coords_index(i,j,k)] == end_psi->psi[coords_index(i,j,k)]);
      }
    }
  }

  free(recv_array);

  psi_free(start_psi);
  psi_free(end_psi);

  decomp_finish();
  coords_finish();
  pe_finalise();

  MPI_Finalize();

  return 0;
}
