/*****************************************************************************
 *
 *  decomp.c
 *
 *  Testing how the decomposition of Ludwig and P3DFFT works
 *
 *  $Id: decomp.c,v 0.2 2013-06-13 12:40:02 ruairi Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics and
 *  Edinburgh Parallel Computing Centre
 *
 *  Ruairi Short (R.Short@sms.ed.ac.uk)
 *  (c) 2013 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <mpi.h>
#include <assert.h>

#include "coords.h"
#include "config.h"
#include "p3dfft.h"

enum {X,Y,Z};

int main() {
  
  MPI_Init(NULL, NULL);

  int num_procs, rank;
  MPI_Comm p3dfft_comm;
  MPI_Comm ludwig_comm;


  int nlocal[3];
  int coords[3];
  int i, j, k;

/* ludwig decomposition */

  printf("initialise coords\n");
  pe_init();
  coords_init();

  ludwig_comm = cart_comm();
  MPI_Comm_size(ludwig_comm, &num_procs);
  MPI_Comm_rank(ludwig_comm, &rank);

  printf("set nlocal\n");
  coords_nlocal(nlocal);
  for(i=0; i<3; i++) {
    coords[i] = cart_coords(i);
  }

  double test_array[nlocal[X], nlocal[Y], nlocal[Z]];
  for(k=0; k<nlocal[X]; k++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(i=0; i<nlocal[Z]; i++) {
        test_array[k,j,i] = rank;
      }
    }
  }

  printf("rank %d, nlocals %d %d %d\n coords: (%d,%d,%d)\n", rank, nlocal[X], nlocal[Y], nlocal[Z], coords[0], coords[1], coords[2]);
/* p3dfft decomposition */
  p3dfft_comm = MPI_COMM_WORLD;

  int nx,ny,nz;
  int istart[3], iend[3], isize[3], ip;
  int overwrite, memsize[3];

  overwrite = 0;

  MPI_Comm_size(p3dfft_comm, &num_procs);
  MPI_Comm_rank(p3dfft_comm, &rank);

  int proc_dims[2];
  nx = ny = nz = 64;



  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
    proc_dims[0] = cart_size(0);
    proc_dims[1] = cart_size(1); 

    p3dfft_setup(proc_dims, nx, ny, nz, overwrite, memsize);

    p3dfft_get_dims(istart, iend, isize, 1);

  //  printf("rank: %d, istart %d %d %d, iend %d %d %d\n", rank, istart[0], istart[1], istart[2], iend[0], iend[1], iend[2]);
    printf("sizes %d %d %d \n", isize[0], isize[1], isize[2]); 
  }
  else { /*we need to do some work */
    proc_dims[0] = num_procs/4;
    proc_dims[1] = 4;

    p3dfft_setup(proc_dims, nx, ny, nz, overwrite, memsize);

    p3dfft_get_dims(istart, iend, isize, 1);

  //  printf("rank: %d, istart %d %d %d, iend %d %d %d\n", rank, istart[0], istart[1], istart[2], iend[0], iend[1], iend[2]);
    printf("sizes %d %d %d \n", isize[0], isize[1], isize[2]); 

  /* now will create the subarrays for the mapping of data */
    int local_coord[2] = {0, 0};
    int global_coord[2];
    int fft_proc_coord[2];
    int array_of_sizes[3], array_of_subsizes[3], array_of_starts[3];
    int dest_proc;
    MPI_Request request;
    MPI_Datatype subarray;

    while(local_coord[0] < nlocal[0]) {
      while(local_coord[1] < nlocal[1]) {
        /* compute destination proc */
        for(i=0; i<3; i++) {
          coords[i] = cart_coords(i);
          array_of_sizes[i] = nlocal[i];
        }
        for(i=0; i<2; i++) {
          global_coord[i] = (coords[i]+local_coord[i])*nlocal[i];
          fft_proc_coord[i] = global_coord[i]/isize[2-i]; 
          array_of_subsizes[i] = isize[2-i] - (global_coord[i]%isize[2-i]);
          if(array_of_subsizes[i] > nlocal[i]) {
            array_of_subsizes[i] = nlocal[i];
          }
          array_of_starts[i] = local_coord[i];
        assert(array_of_starts[i] >= 0);
        if(0 == rank  || array_of_starts[i] > (array_of_sizes[i] - array_of_subsizes[i]) ) {
            printf("rank %d, i %d, row %d, col %d, start %d, size %d, subsize %d\n", rank, i, local_coord[1], local_coord[0], array_of_starts[i], array_of_sizes[i], array_of_subsizes[i]);
          }
        assert(array_of_starts[i] <= (array_of_sizes[i] - array_of_subsizes[i]) );
        }

        dest_proc = fft_proc_coord[0]*proc_dims[0] + fft_proc_coord[1];

        /* set up subarray to be sent, mostly already done*/
        array_of_subsizes[2] = nlocal[2];
        array_of_starts[2] = 0;
        assert(array_of_starts[2] >= 0);
        assert(array_of_starts[2] <= (array_of_sizes[2] - array_of_subsizes[2]) );

        MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &subarray) ;
        MPI_Type_commit(&subarray);

  //      MPI_Isend(test_array, 1, subarray, dest_proc, 1, ludwig_comm, &request);
        printf("array of starts %d %d %d \n", array_of_starts[0], array_of_starts[1], array_of_starts[2]);

        /* clean up */
        /* Do I need to? */
        MPI_Type_free(&subarray);
        /* increment to the next row that needs to be checked */
        local_coord[1] += array_of_subsizes[1];
      }
      /* all rows on this column accounted for, move to next column that needs checking */
      local_coord[1] = 0;
      local_coord[0] += array_of_subsizes[0];
    }
  }


  MPI_Finalize();

  return 0;
}
