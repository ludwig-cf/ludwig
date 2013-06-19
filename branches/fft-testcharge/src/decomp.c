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

/* In theory should be able to adapt this into a function that will change the decompositions.
 * However it may be necessary to change the actual MPI sends/recvs to a seperate routine as this will allow them to be done seperately in future 
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#include "coords.h"
#include "config.h"
#include "p3dfft.h"

//enum {X,Y,Z};

int find_recv_proc (int global_coord[], int nlocal[]);
int ar_index (int x, int y, int z, int isize[]);

int main() {
  
  MPI_Init(NULL, NULL);

  int num_procs, rank;
  MPI_Comm ludwig_comm;
  MPI_Comm p3dfft_comm;


  int nlocal[3];
  int coords[3];
  int i, j, k;
  int print = 0;

/* ludwig decomposition setup*/

  pe_init();
  coords_init();

  ludwig_comm = cart_comm();
  MPI_Comm_size(ludwig_comm, &num_procs);
  MPI_Comm_rank(ludwig_comm, &rank);
  MPI_Status status;

  coords_nlocal(nlocal);
  for(i=0; i<3; i++) {
    coords[i] = cart_coords(i);
  }

  double send_array[nlocal[X]] [nlocal[Y]] [nlocal[Z]];
  for(k=0; k<nlocal[X]; k++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(i=0; i<nlocal[Z]; i++) {
        send_array[k][j][i] = rank;
      }
    }
  }

/* p3dfft decomposition setup */
  p3dfft_comm = MPI_COMM_WORLD;

  int nx,ny,nz;
  int istart[3], iend[3], isize[3], ip;
  int overwrite, memsize[3];
  

  overwrite = 0; /* don't allow overwriting input of btran */

  MPI_Comm_size(p3dfft_comm, &num_procs);
  MPI_Comm_rank(p3dfft_comm, &rank);

  int proc_dims[2];
/* these are in Fortran ordering
 * as such x is the direction of stride-1 memory
 */
  nx = 16;
  ny = 8;
  nz = 4;

  int *dest_proc = malloc(sizeof(int));;
  int *recv_proc = malloc(sizeof(int));;
  int send_count, recv_count;
  MPI_Request *request = malloc(sizeof(MPI_Request));
  MPI_Datatype *send_subarray = malloc(sizeof(MPI_Datatype));
  MPI_Datatype *recv_subarray = malloc(sizeof(MPI_Datatype));

  double *recv_array; /* The size of this is unknown until p3dfft_get_dims is called */

/* transferring data from cartesian to pencil */
  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
    if(rank == 0) { printf("already in pencil decomposition!\n"); }
    proc_dims[0] = cart_size(0);
    proc_dims[1] = cart_size(1); 

    p3dfft_setup(proc_dims, nx, ny, nz, overwrite, memsize);

    ip = 1;
    p3dfft_get_dims(istart, iend, isize, ip);

    recv_array = malloc(isize[0]*isize[1]*isize[2]*sizeof(double));
    assert(isize[0] == nlocal[Z]);
    for(k=0; k<nlocal[X]; k++) {
      for(j=0; j<nlocal[Y]; j++) {
        for(i=0; i<nlocal[Z]; i++) {
          recv_array[ar_index(k,j,i,isize)] = send_array[k][j][i];
        }
      }
    }

  }
  else { /*we need to do some work */
    if(rank == 0) { printf("Moving data from cartesian to pencil decomposition.\n"); }
/* It will be necessary to allow the user specify the grid they want */
    proc_dims[0] = num_procs/4;
    proc_dims[1] = 4;

    
/* for now will only consider pencil decompositions that divide evenly into the lattice */
    assert(ny%proc_dims[1] == 0);
    assert(nz%proc_dims[0] == 0);

    /* note that this routine swaps the elements of proc_dims */
    p3dfft_setup(proc_dims, nx, ny, nz, overwrite, memsize);

    ip = 1;
    p3dfft_get_dims(istart, iend, isize, ip);

  /* now will create the subarrays for the mapping of data */
    int local_coord[3] = {0, 0, 0};
    int global_coord[3];
    int fft_proc_coord[2];
    int array_of_sizes[3], array_of_subsizes[3], array_of_starts[3];
    recv_count = send_count = 1;

    recv_array = malloc(isize[0]*isize[1]*isize[2]*sizeof(double));

    while(local_coord[0] < nlocal[0]) {
      while(local_coord[1] < nlocal[1]) {
        if(local_coord[0] != 0 || local_coord[1] !=0) {
          send_count ++;
          dest_proc = realloc(dest_proc, send_count*sizeof(int));
          send_subarray = realloc(send_subarray, send_count*sizeof(MPI_Datatype));
          request = realloc(request, send_count*sizeof(MPI_Request));
        }
        /* compute destination proc (put this in fn) */
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
          array_of_starts[i] = local_coord[i];

          assert(array_of_starts[i] >= 0);
          assert(array_of_starts[i] <= (array_of_sizes[i] - array_of_subsizes[i]) );
        }

        dest_proc[send_count-1] = fft_proc_coord[0]*proc_dims[1] + fft_proc_coord[1];
        assert(dest_proc[send_count-1] < num_procs);

        /* set up subarray to be sent, mostly already done*/
        array_of_subsizes[2] = nlocal[2];
        array_of_starts[2] = 0;

        assert(array_of_starts[2] >= 0);
        assert(array_of_starts[2] <= (array_of_sizes[2] - array_of_subsizes[2]) );

        MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &send_subarray[send_count-1]) ;
        MPI_Type_commit(&send_subarray[send_count-1]);
        
        /* send the data */
        MPI_Isend(array_of_subsizes, 3, MPI_INT, dest_proc[send_count-1], 1, ludwig_comm, &request[send_count-1]);
        MPI_Isend(send_array, 1, send_subarray[send_count-1], dest_proc[send_count-1], 2, ludwig_comm, &request[send_count-1]);

        /* increment to the next row that needs to be checked */
        local_coord[1] += array_of_subsizes[1];
      }
      /* all rows on this column accounted for, move to next column that needs checking */
      local_coord[1] = 0;
      local_coord[0] += array_of_subsizes[0];
    }

    /* prepare to receive */
    int recv_size[3];

    /* first receive the messages detailing sizes */
    for(i=0; i<3; i++) {
      local_coord[i] = 0;
    }

    while(local_coord[0] < isize[2]) {
      while(local_coord[1] < isize[1]) {
        while(local_coord[2] < isize[0]) {
          if(local_coord[2] != 0 || local_coord[1] !=0 || local_coord[0] !=0) {
            recv_count++;
            recv_subarray = realloc(recv_subarray, recv_count*sizeof(MPI_Datatype));
            recv_proc = realloc(recv_proc, recv_count*sizeof(int));
          }
          for(i=0; i<3; i++) {
            /* istart is global coord in fortran code (ie starts at 1*/
            global_coord[i] = istart[2-i] - 1 + local_coord[i];
          }
          recv_proc[recv_count-1] = find_recv_proc(global_coord, nlocal);
          MPI_Recv(recv_size, 3, MPI_INT, recv_proc[recv_count-1], 1, ludwig_comm, &status);
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

/*wait on sends*/

  }
  if(rank == 0) { printf("Data now in pencil decomposition\n"); }

/* this print is for testing purposes. 
    if(rank == print) { 
      for(k=0; k<isize[Z]; k++) {
        for(j=0; j<isize[Y]; j++) {
          for(i=0; i<isize[X]; i++) {
            printf("%.0f ", recv_array[ar_index(k,j,i,isize)]);
          }
          printf("\t");
        }
        printf("\n");
      }
    }
*/

/* do FFT here */
  /* need dimensions of complex array */
  int fstart[3], fend[3], fsize[3];
  unsigned char op_f[3]="fft", op_b[3]="tff";


  ip = 2;
  p3dfft_get_dims(fstart, fend, fsize, ip);
  if(rank == print) { printf("fsize %d %d %d\n", fsize[0], fsize[1], fsize[2]); }

  double *transf_array = malloc(2*fsize[0]*fsize[1]*fsize[2] * sizeof(double)); /* x2 since they will be complex numbers */
  double *end_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));

  if(rank == 0) { printf("Performing forward fft\n"); }
  p3dfft_ftran_r2c(recv_array, transf_array, op_f);

/*
  if(rank == print) {
    for(k=0; k<fsize[Z]; k++) {
      for(j=0; j<fsize[Y]; j++) {
        for(i=0; i<fsize[X]; i+=2) { 
          printf("%.0f %.0f  ", transf_array[ar_index(k,j,i,fsize)], transf_array[ar_index(k,j,i+1,fsize)]);}
        }
        printf("\t\t\t");
      }
      printf("\n");
    }
  }
*/
  
  if(rank == 0) { printf("Performing backward fft\n"); }
  p3dfft_btran_c2r(transf_array, end_array, op_b);

/* checking the array is the same after transforming back also dividing by (nx*ny*nz)*/
      for(k=0; k<isize[Z]; k++) {
        for(j=0; j<isize[Y]; j++) {
          for(i=0; i<isize[X]; i++) {
            end_array[ar_index(k,j,i,isize)] = end_array[ar_index(k,j,i,isize)]/(nx*ny*nz);
            if(abs(recv_array[ar_index(k,j,i,isize)] - end_array[ar_index(k,j,i,isize)] ) > 0.1e-5) {
              printf("error: rank %d, end array[%d][%d][%d] is wrong, %f and should be %f\n", rank, k, j, i, end_array[ar_index(k,j,i,isize)], recv_array[ar_index(k,j,i,isize)]);
            }
          }
        }
      }

  /*send the data back to cartesian decomp */
  if(cart_size(2) == 1) { /* already in a pencil decomposition! */

  }
  else {
    if(rank == 0) { printf("Transferring data back to cartesian processor decomposition\n"); }
    MPI_Request pencil_cart_request[recv_count];

    for(i=0; i<recv_count; i++) {
      MPI_Isend(recv_array, 1, recv_subarray[i], recv_proc[i], 2, ludwig_comm, &pencil_cart_request[i]);
    }
    for(i=0; i<send_count; i++) {
      MPI_Recv(send_array, 1, send_subarray[i], dest_proc[i], 2, ludwig_comm, &status);
    }
  }

    for(k=0; k<nlocal[X]; k++) {
      for(j=0; j<nlocal[Y]; j++) {
        for(i=0; i<nlocal[Z]; i++) {
          if(send_array[k][j][i] != rank) {
//            printf("error: send array[%d][%d][%d] is wrong, %d and should be %d\n", k, j, i, send_array[k][j][i], rank);
          }
        }
      }
    }

  if(rank == 0) { printf("Data now in cartesian processor decomposition\n"); }


/* careful about this freeing, ideally we would have these be static and when the decomposition is next to be changed the messages can simply be sent, without the need for calculating the destinations again*/

  free(dest_proc);
  free(recv_proc);
  free(request);
  free(recv_array);

  MPI_Finalize();

  return 0;
}


int find_recv_proc (int global_coord[], int nlocal[]) {
 
  int i;
  int coords[3] = {0,0,0};
  int recv_proc;

  for(i=0; i<3; i++) {
/*    assert(global_coord[i] < ntotal_[i]); */
    coords[i] = global_coord[i]/nlocal[i];
  }

  recv_proc = coords[0]*cart_size(1)*cart_size(2) + coords[1]*cart_size(2) + coords[2];  


  return recv_proc;
}

int ar_index (int x, int y, int z, int isize[]) {

  return x*isize[1]*isize[0] + y*isize[0] + z; 
}
