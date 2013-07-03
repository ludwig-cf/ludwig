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

static int num_procs          = 0;

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
 *****************************************************************************/

int main() {

  MPI_Init(NULL, NULL);
  pe_init();
  coords_init();

  struct holder *data        = malloc(sizeof(struct holder));
  MPI_Comm ludwig_comm       = cart_comm();
  double global_coord[3]     = {0, 0, 0};
  int nlocal[3]              = {0, 0, 0};
  int isize[3]               = {0, 0, 0};
  int istart[3]              = {0, 0, 0};
  int proc_dims[2]           = {0, 0};
  int i, j, k;
  int rank                   = cart_rank();

  num_procs = pe_size();
  coords_nlocal(nlocal);


  psi_t *start_psi = NULL;
  psi_t *end_psi = NULL;

  psi_create(2, &start_psi);
  psi_create(2, &end_psi);
  assert(start_psi);
  assert(end_psi);
  

  /* initialise elements of send_array to binary evaluation of global coordinates */
  for(i=0; i<nlocal[X]; i++) {
    global_coord[X] = (cart_coords(X)*nlocal[X] + i); 
    for(j=0; j<nlocal[Y]; j++) {
      global_coord[Y] = (cart_coords(Y)*nlocal[Y] + j); 
      for(k=0; k<nlocal[Z]; k++) {
        global_coord[Z] = (cart_coords(Z)*nlocal[Z] + k); 
        start_psi->psi[index_3d_c(i,j,k,nlocal)] = hash(global_coord);
      }
    }
  }

  initialise_decomposition_swap(proc_dims);

  pencil_starts(istart);
  pencil_sizes(isize);

  double *recv_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));

  cart_to_pencil(start_psi->psi, recv_array);


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

  pencil_to_cart(recv_array, end_psi->psi);

  /*test final array is the same as the original*/
  for(i=0; i<nlocal[X]; i++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(k=0; k<nlocal[Z]; k++) {
        assert(start_psi->psi[index_3d_c(i,j,k,nlocal)] == end_psi->psi[index_3d_c(i,j,k,nlocal)]);
      }
    }
  }

  free(recv_array);

  psi_free(start_psi);
  psi_free(end_psi);

  coords_finish();
  pe_finalise();
  p3dfft_clean();

  MPI_Finalize();

  return 0;
}
