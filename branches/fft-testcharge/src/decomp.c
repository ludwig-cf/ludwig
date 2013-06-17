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

int find_recv_proc (int local_coord[3], int nlocal[3]);

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

  double test_array[nlocal[X]] [nlocal[Y]] [nlocal[Z]];
  for(k=0; k<nlocal[X]; k++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(i=0; i<nlocal[Z]; i++) {
        test_array[k][j][i] = rank;
      }
    }
  }

//  printf("rank %d, nlocals %d %d %d\n coords: (%d,%d,%d)\n", rank, nlocal[X], nlocal[Y], nlocal[Z], coords[0], coords[1], coords[2]);
/* p3dfft decomposition */
  p3dfft_comm = MPI_COMM_WORLD;

  int nx,ny,nz;
  int istart[3], iend[3], isize[3], ip;
  int overwrite, memsize[3];

  overwrite = 0;

  MPI_Comm_size(p3dfft_comm, &num_procs);
  MPI_Comm_rank(p3dfft_comm, &rank);

  int proc_dims[2];
  nx = ny = nz = 16;


/* transferring data from cartesian to pencil */
  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
    if(rank == 0) { printf("already in pencil decomposition!\n"); }
    proc_dims[0] = cart_size(0);
    proc_dims[1] = cart_size(1); 

    p3dfft_setup(proc_dims, nx, ny, nz, overwrite, memsize);

    p3dfft_get_dims(istart, iend, isize, 1);

  //  printf("rank: %d, istart %d %d %d, iend %d %d %d\n", rank, istart[0], istart[1], istart[2], iend[0], iend[1], iend[2]);
    printf("sizes %d %d %d \n", isize[0], isize[1], isize[2]); 
    double recv_array[isize[0]] [isize[1]] [isize[2]];
    for(k=0; k<nlocal[X]; k++) {
      for(j=0; j<nlocal[Y]; j++) {
        for(i=0; i<nlocal[Z]; i++) {
          recv_array[k][j][i] = test_array[k][j][i];
        }
      }
    }
    if(rank == 0) {
      for(k=0; k<isize[X]; k++) {
        for(j=0; j<isize[Y]; j++) {
          for(i=0; i<isize[Z]; i++) {
            printf("%f ", recv_array[k][j][i]);
          }
          printf("\t");
        }
        printf("\n");
      }
    }

  }
  else { /*we need to do some work */
    if(rank == 0) { printf("Moving data from cartesian to pencil decomposition.\n"); }
    proc_dims[0] = num_procs/4;
    proc_dims[1] = 4;

    
    if(rank == 0) { printf("before proc_dims %d %d\n", proc_dims[0], proc_dims[1]); }
/* for now will only consider good decompositions of new fft grid */
    assert(ny%proc_dims[1] == 0);
    assert(nz%proc_dims[0] == 0);

    /* note that this routine swaps the elements of proc_dims */
    p3dfft_setup(proc_dims, nx, ny, nz, overwrite, memsize);

    p3dfft_get_dims(istart, iend, isize, 1);


  //  printf("rank: %d, istart %d %d %d, iend %d %d %d\n", rank, istart[0], istart[1], istart[2], iend[0], iend[1], iend[2]);
    printf("sizes %d %d %d \n", isize[0], isize[1], isize[2]); 

  /* now will create the subarrays for the mapping of data */
    double recv_array[isize[2]] [isize[1]] [isize[0]];
    int local_coord[3] = {0, 0, 0};
    int global_coord[2];
    int fft_proc_coord[2];
    int array_of_sizes[3], array_of_subsizes[3], array_of_starts[3];
    int dest_proc;
    int send_count, recv_count;
    MPI_Request *request = malloc(sizeof(MPI_Request));
    MPI_Datatype *send_subarray = malloc(sizeof(MPI_Datatype));
    MPI_Datatype *recv_subarray = malloc(sizeof(MPI_Datatype));
  
    recv_count = send_count = 1;

    while(local_coord[0] < nlocal[0]) {
      while(local_coord[1] < nlocal[1]) {
        if(local_coord[0] != 0 || local_coord[1] !=0) {
          send_count ++;
          realloc(send_subarray, send_count*sizeof(MPI_Datatype));
          realloc(request, send_count*sizeof(MPI_Request));
        }
        /* compute destination proc */
        for(i=0; i<3; i++) {
          coords[i] = cart_coords(i);
          array_of_sizes[i] = nlocal[i];
        }

        for(i=0; i<2; i++) {
          global_coord[i] = (coords[i]*nlocal[i]) + local_coord[i];
          fft_proc_coord[i] = global_coord[i]/isize[2-i]; 
          assert(fft_proc_coord[i] < proc_dims[i] && fft_proc_coord >= 0);
          array_of_subsizes[i] = isize[2-i] - (global_coord[i]%isize[2-i]);
          if(array_of_subsizes[i] > nlocal[i]) {
            array_of_subsizes[i] = nlocal[i];
          }
          printf("subsize[%d] %d\n", i, array_of_subsizes[i]);
          array_of_starts[i] = local_coord[i];

          assert(array_of_starts[i] >= 0);
          assert(array_of_starts[i] <= (array_of_sizes[i] - array_of_subsizes[i]) );
        }

        dest_proc = fft_proc_coord[0]*proc_dims[1] + fft_proc_coord[1];
        assert(dest_proc < num_procs);

        /* set up subarray to be sent, mostly already done*/
        array_of_subsizes[2] = nlocal[2];
        array_of_starts[2] = 0;

        assert(array_of_starts[2] >= 0);
        assert(array_of_starts[2] <= (array_of_sizes[2] - array_of_subsizes[2]) );

        MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &send_subarray[send_count-1]) ;
        MPI_Type_commit(&send_subarray[send_count-1]);
        
        /* send the data */
        MPI_Isend(array_of_subsizes, 3, MPI_INT, dest_proc, 1, ludwig_comm, &request[send_count-1]);
        MPI_Isend(test_array, 1, send_subarray[send_count-1], dest_proc, 2, ludwig_comm, &request[send_count-1]);
        printf("array of starts %d %d %d \n", array_of_starts[0], array_of_starts[1], array_of_starts[2]);

        /* clean up */
        /* Do I need to? - No want to keep this structure for sending back to original decomposition*/
//        MPI_Type_free(send_subarray[send_count-1]);
        /* increment to the next row that needs to be checked */
        local_coord[1] += array_of_subsizes[1];
      }
      /* all rows on this column accounted for, move to next column that needs checking */
      local_coord[1] = 0;
      local_coord[0] += array_of_subsizes[0];
    }

    /* prepare to receive */
    int recv_size[3];
    int *recv_proc = malloc(sizeof(int));;
    MPI_Status status;

    /* first receive the messages detailing sizes */
    for(i=0; i<3; i++) {
      local_coord[i] = 0;
    }

/* processors currently try to receive from the same remote proc more than once, need to fix this */
    while(local_coord[0] < isize[2]) {
      while(local_coord[1] < isize[1]) {
        while(local_coord[2] < isize[0]) {
          if(local_coord[2] != 0 || local_coord[1] !=0 || local_coord[0] !=0) {
            recv_count++;
            realloc(recv_subarray, recv_count*sizeof(MPI_Datatype));
            realloc(recv_proc, recv_count*sizeof(int));
  //          realloc(request, recv_count*sizeof(MPI_Request));
          }
          for(i=0; i<2; i++) {
            global_coord[i] = istart[2-i] + local_coord[i];
          }
/* look at why this doesn't give correct values */
          recv_proc[recv_count-1] = find_recv_proc(global_coord, nlocal);

          printf("proc %d receiving from %d, count %d\n", rank, recv_proc[recv_count-1], recv_count);
          MPI_Recv(recv_size, 3, MPI_INT, recv_proc[recv_count-1], 1, ludwig_comm, &status);
          if(rank == 0) { printf("recv_size %d %d %d, isize %d %d %d \n", recv_size[0], recv_size[1], recv_size[2], isize[0], isize[1], isize[2]); }
          assert(recv_size[2] <= isize[0]);
          assert(recv_size[1] <= isize[1]);
          assert(recv_size[0] <= isize[2]);

          /* create subarray for receiving */
          for(i=0; i<3; i++) {
            array_of_starts[i] = local_coord[i];
            array_of_subsizes[i] = recv_size[i];
            array_of_sizes[i] = isize[2-i];
            assert(array_of_starts[i] >= 0);
            assert(array_of_starts[i] <= (array_of_sizes[i] - array_of_subsizes[i]) );
          }

    /* can probably be more clever about this in situations where each proc will have the same subarray to recv */
          MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &recv_subarray[recv_count-1]) ;
          MPI_Type_commit(&recv_subarray[recv_count-1]);

          MPI_Recv(recv_array, 1, recv_subarray[recv_count-1], recv_proc[recv_count-1], 2, ludwig_comm, &status);

//        MPI_Type_free(&recv_subarray[recv_count-1]);
          local_coord[2] += recv_size[2];
        } 
        local_coord[2] = 0;
        local_coord[1] += recv_size[1];
      }
      local_coord[1] = 0;
      local_coord[0] += recv_size[0];
    }

    if(rank == 0) {
      for(k=0; k<isize[X]; k++) {
        for(j=0; j<isize[Y]; j++) {
          for(i=0; i<isize[Z]; i++) {
            printf("%.1f ", recv_array[k][j][i]);
          }
          printf("\t");
        }
        printf("\n");
      }
    }

  }



  MPI_Finalize();

  return 0;
}


int find_recv_proc (int global_coord[3], int nlocal[3]) {
 
  int i;
  int coords[3];
  int recv_proc;

  for(i=0; i<2; i++) {
/*    assert(global_coord[i] < ntotal_[i]); */
    coords[i] = global_coord[i]/nlocal[i];
  }

  recv_proc = ( coords[2]*cart_size(2) + coords[1]*cart_size(1) ) + coords[0] * (cart_size(1)*cart_size(2));

  return recv_proc;
}
