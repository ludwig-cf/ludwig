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


struct holder {
  double *recv_array;

  int *dest_proc;
  int  send_count;
  int *send_subsizes;
  MPI_Datatype *send_subarray;

  int *recv_proc;
  int recv_count;
  int *recv_subsizes;
  MPI_Datatype *recv_subarray;

};

static int initialised_                = 0;

int find_recv_proc (int global_coord[], int nlocal[]);
int index_3d (int x, int y, int z, int isize[]);
void find_cart_global_coord(int global_coord[], int local_coord[], int nlocal[], int coords[]);
void find_cart_subarray_subsizes(int array_of_subsizes[], int global_coord[], int isize[], int nlocal[]);
void find_cart_subarray_starts(int array_of_starts[], int local_coord[]);
int find_cart_dest_proc(int global_coord[], int isize[], int proc_dims[]);
int find_number_cart_dest_procs(int isize[], int proc_dims[], int nlocal[], int num_procs, struct holder *data);
int find_number_pencil_recv_procs(int isize[], int istart[], int nlocal[], int num_procs, struct holder *data, MPI_Comm ludwig_comm);
void initialise_decomposition_swap(struct holder *data, int nlocal[], int num_procs, MPI_Comm ludwig_comm, int isize[]);

int main() {
  
  MPI_Init(NULL, NULL);

  int num_procs, rank;
  MPI_Comm ludwig_comm;
  MPI_Comm p3dfft_comm;

  int nlocal[3];
  int coords[3];
  int i, j, k;
  int isize[3];
  int ip;

  struct holder *data = malloc(sizeof(struct holder));

  double send_array[nlocal[X]] [nlocal[Y]] [nlocal[Z]];
  double final_array[nlocal[X]] [nlocal[Y]] [nlocal[Z]];

/* ludwig decomposition setup*/

  pe_init();
  coords_init();

  ludwig_comm = cart_comm();
  MPI_Comm_size(ludwig_comm, &num_procs);
  MPI_Comm_rank(ludwig_comm, &rank);

  coords_nlocal(nlocal);
  for(i=0; i<3; i++) {
    coords[i] = cart_coords(i);
  }

  for(k=0; k<nlocal[X]; k++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(i=0; i<nlocal[Z]; i++) {
        send_array[k][j][i] = rank;
        final_array[k][j][i] = rank;
      }
    }
  }

/* p3dfft decomposition setup */
  p3dfft_comm = MPI_COMM_WORLD;

  MPI_Comm_size(p3dfft_comm, &num_procs);
  MPI_Comm_rank(p3dfft_comm, &rank);


  if(initialised_ == 0) {
    initialise_decomposition_swap(data, nlocal, num_procs, ludwig_comm, isize);
    initialised_ = 1;
  }

  MPI_Request send_request[data->send_count];
  MPI_Status send_status[data->send_count];
  MPI_Status recv_status;

  if(rank == 0) { printf("Executing swap\n"); } 
  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
/* copy data or move pointer */
    for(k=0; k<nlocal[X]; k++) {
      for(j=0; j<nlocal[Y]; j++) {
        for(i=0; i<nlocal[Z]; i++) {
          data->recv_array[index_3d(k,j,i,isize)] = send_array[k][j][i];
        }
      }
    }
  }
  else { /* swap decompositions */
    for(i=0; i<data->send_count; i++) {
      MPI_Isend(send_array, 1, data->send_subarray[i], data->dest_proc[i], 2, ludwig_comm, &send_request[i]);
    }

    if(rank == 0) { printf("Posting receives\n"); }
    for(i=0; i<data->recv_count; i++) {
      MPI_Recv(data->recv_array, 1, data->recv_subarray[i], data->recv_proc[i], 2, ludwig_comm, &recv_status);
    }
  }

  MPI_Waitall(data->send_count, send_request, send_status);
/*wait on sends*/

  
  if(rank == 0) { printf("Data now in pencil decomposition\n"); }


/* do FFT here */
  /* need dimensions of complex array */
  int fstart[3], fend[3], fsize[3];
  unsigned char op_f[3]="fft", op_b[3]="tff";


  ip = 2;
  p3dfft_get_dims(fstart, fend, fsize, ip);

  double *transf_array = malloc(2*fsize[0]*fsize[1]*fsize[2] * sizeof(double)); /* x2 since they will be complex numbers */
  double *end_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));

  if(rank == 0) { printf("Performing forward fft\n"); }
  p3dfft_ftran_r2c(data->recv_array, transf_array, op_f);

  
  if(rank == 0) { printf("Performing backward fft\n"); }
  p3dfft_btran_c2r(transf_array, end_array, op_b);

/* checking the array is the same after transforming back also dividing by (nx*ny*nz)*/
      for(k=0; k<isize[Z]; k++) {
        for(j=0; j<isize[Y]; j++) {
          for(i=0; i<isize[X]; i++) {
            end_array[index_3d(k,j,i,isize)] = end_array[index_3d(k,j,i,isize)]/(N_total(0)*N_total(1)*N_total(2));
            if(abs(data->recv_array[index_3d(k,j,i,isize)] - end_array[index_3d(k,j,i,isize)] ) > 0.1e-10) {
              printf("error: rank %d, end array[%d][%d][%d] is wrong, %f and should be %f\n", rank, k, j, i, end_array[index_3d(k,j,i,isize)], data->recv_array[index_3d(k,j,i,isize)]);
            }
          }
        }
      }


  /*send the data back to cartesian decomp */
  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
/* copy data back (or move pointer) */
    for(k=0; k<nlocal[X]; k++) {
      for(j=0; j<nlocal[Y]; j++) {
        for(i=0; i<nlocal[Z]; i++) {
          final_array[k][j][i] = end_array[index_3d(k,j,i,isize)];
        }
      }
    }
  }
  else {
    if(rank == 0) { printf("Transferring data back to cartesian processor decomposition\n"); }
    MPI_Request pencil_cart_request[data->recv_count];
    MPI_Status pencil_cart_status[data->recv_count];

    for(i=0; i<data->recv_count; i++) {
      MPI_Isend(end_array, 1, data->recv_subarray[i], data->recv_proc[i], 2, ludwig_comm, &pencil_cart_request[i]);
    }
    for(i=0; i<data->send_count; i++) {
      MPI_Recv(final_array, 1, data->send_subarray[i], data->dest_proc[i], 2, ludwig_comm, &send_status[i]);
    }

    MPI_Waitall(data->send_count, pencil_cart_request, pencil_cart_status);
  }


  for(k=0; k<nlocal[X]; k++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(i=0; i<nlocal[Z]; i++) {
        if(abs(send_array[k][j][i] -  final_array[k][j][i]) > 1e-10) {
          printf("rank %d, error: final array[%d][%d][%d] is wrong, %f and should be %f\n", rank, k, j, i, final_array[k][j][i],  send_array[k][j][i]);
        }
      }
    }
  }

  if(rank == 0) { printf("Data now in cartesian processor decomposition\n"); }


/* careful about this freeing, ideally we would have these be static and when the decomposition is next to be changed the messages can simply be sent, without the need for calculating the destinations again*/

//  free(dest_proc);
//  free(recv_proc);
//  free(request);
//  free(recv_array);

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

/* find the index of a 2d malloc'd array, where size is the extent in the x dimension */
int index_2d (int x, int y, int size) {

  return x*size + y; 
}

int index_3d (int x, int y, int z, int isize[]) {

  return x*isize[1]*isize[0] + y*isize[0] + z; 
}


void find_cart_global_coord(int global_coord[], int local_coord[], int nlocal[], int coords[]) {
  int i;
  for(i=0; i<2; i++) { 
    global_coord[i] = (coords[i]*nlocal[i]) + local_coord[i];
  }
}

void find_cart_subarray_subsizes(int array_of_subsizes[], int global_coord[], int isize[], int nlocal[]) {
  int i;
  for(i=0; i<3; i++) {
    if(i != 3) {
      array_of_subsizes[i] = isize[2-i] - (global_coord[i]%isize[2-i]);
      if(array_of_subsizes[i] > nlocal[i]) {
        array_of_subsizes[i] = nlocal[i];
      }
    }
    else {
      array_of_subsizes[i] = nlocal[i];
    }
  }
}

void find_cart_subarray_starts(int array_of_starts[], int local_coord[]) {
  int i;
  for(i=0; i<3; i++) {
    if(i != 3) {
      array_of_starts[i] = local_coord[i];
    }
    else {
      array_of_starts[i] = 0;
    }
  }
}

int find_cart_dest_proc(int global_coord[], int isize[], int proc_dims[]) {
  int i;
  int fft_proc_coord[2];
  for(i=0; i<2; i++) {
    fft_proc_coord[i] = global_coord[i]/isize[2-i]; 
    assert(fft_proc_coord[i] < proc_dims[i] && fft_proc_coord >= 0);
  }
  return fft_proc_coord[0]*proc_dims[1] + fft_proc_coord[1];
}

/*returns numnber of destination processors */
int find_number_cart_dest_procs(int isize[], int proc_dims[], int nlocal[], int num_procs, struct holder *data) {

  int global_coord[3] = {0, 0, 0};
  int local_coord[3] = {0, 0, 0};
  int coords[3];
  int array_of_subsizes[3], array_of_starts[3];
  int send_count = 0;
  int iter = 0;
  int i;

  for(i=0; i<3; i++) {
    coords[i] = cart_coords(i);
  }

/* first find how many messages need to be sent */
  while(local_coord[0] < nlocal[0]) {
    while(local_coord[1] < nlocal[1]) {
      send_count ++;
      find_cart_global_coord(global_coord, local_coord, nlocal, coords);
      find_cart_subarray_subsizes(array_of_subsizes, global_coord, isize, nlocal);


      local_coord[1] += array_of_subsizes[1];
    }
    local_coord[1] = 0;
    local_coord[0] += array_of_subsizes[0];
  }

/* can then assign memory for dest_proc[] and send_subarray[] */
  data->dest_proc = malloc(send_count*sizeof(int));
  data->send_subarray = malloc(send_count*sizeof(MPI_Datatype));
  data->send_subsizes = malloc(3*send_count*sizeof(int));

  for(i=0; i<3; i++) {
    local_coord[i] = 0;
    global_coord[i] = 0;
  }

  iter = 0;
/* finally compute dest_proc[] and subarray[] elements */
  while(local_coord[0] < nlocal[0]) {
    while(local_coord[1] < nlocal[1]) {

      find_cart_global_coord(global_coord, local_coord, nlocal, coords);
      find_cart_subarray_subsizes(array_of_subsizes, global_coord, isize, nlocal);
      find_cart_subarray_starts(array_of_starts, local_coord);

      for(i=0; i<3; i++) {
        assert(array_of_starts[i] >= 0);
        assert(array_of_starts[i] <= (nlocal[i] - array_of_subsizes[i]) );
      }

      data->dest_proc[iter] = find_cart_dest_proc(global_coord, isize, proc_dims);
      assert(data->dest_proc[iter] < num_procs);

      /* set up subarray to be sent */
      MPI_Type_create_subarray(3, nlocal, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &data->send_subarray[iter]);
      MPI_Type_commit(&data->send_subarray[iter]);

      assert(&data->send_subarray[send_count] != NULL);
      
      for(i=0; i<3; i++) {
        data->send_subsizes[index_2d(iter,i,3)] = array_of_subsizes[i];
      }

      /* increment to the next row that needs to be checked */
      local_coord[1] += array_of_subsizes[1];
      iter++;
      assert(iter <= send_count);
    }
    /* all rows on this column accounted for, move to next column that needs checking */
    local_coord[1] = 0;
    local_coord[0] += array_of_subsizes[0];
  }

  return send_count;
}

int find_number_pencil_recv_procs(int isize[], int istart[], int nlocal[], int num_procs, struct holder *data, MPI_Comm ludwig_comm) {

    int i, recv_count;
    int array_size;
    int local_coord[3] = {0,0,0};
    int global_coord[3] = {0,0,0};
    int array_of_starts[3], array_of_subsizes[3], array_of_sizes[3];
    MPI_Status status;

/*
 * Will be using realloc. It is necessary to make a guess at how many procs need to be received from. The smallest this will
 * be is the number of procs in the z dimension in the cartesian decomposition. Use this as the guess and resize to be twice
 * as big each resize.
 */

  data->recv_subsizes = malloc(3*cart_size(2)*sizeof(int));
  data->recv_subarray = malloc(cart_size(2)*sizeof(MPI_Datatype));
  data->recv_proc = malloc(cart_size(2)*sizeof(int));
  array_size = cart_size(2);

  recv_count = 0;
  while(local_coord[0] < isize[2]) {
    while(local_coord[1] < isize[1]) {
      while(local_coord[2] < isize[0]) {
        if(recv_count == array_size) {
          printf("Resizing\n");
          data->recv_subsizes = realloc(data->recv_subsizes, 3*2*array_size*sizeof(int));
          data->recv_subarray = realloc(data->recv_subarray, 2*array_size*sizeof(MPI_Datatype));
          data->recv_proc = realloc(data->recv_proc, 2*array_size*sizeof(int));
          array_size = array_size * 2;
        }
        for(i=0; i<3; i++) {
          /* istart is global coord in fortran code (ie starts at 1)*/
          global_coord[i] = istart[2-i] - 1 + local_coord[i];
        }
        data->recv_proc[recv_count] = find_recv_proc(global_coord, nlocal);
        assert(data->recv_proc[recv_count] < num_procs);

        /* receive the messages detailing sizes */
        MPI_Recv(&data->recv_subsizes[index_2d(recv_count,0,3)], 3, MPI_INT, data->recv_proc[recv_count], 1, ludwig_comm, &status);
        assert(data->recv_subsizes[index_2d(recv_count,2,3)] <= isize[0] && data->recv_subsizes[index_2d(recv_count,1,3)] <= isize[1] && data->recv_subsizes[index_2d(recv_count,0,3)] <= isize[2]);

        /* create subarray for receiving */
        for(i=0; i<3; i++) {
          array_of_starts[i] = local_coord[i];
          array_of_subsizes[i] = data->recv_subsizes[index_2d(recv_count,i,3)];
          array_of_sizes[i] = isize[2-i];
          assert(array_of_starts[i] >= 0);
          assert(array_of_starts[i] <= (array_of_sizes[i] - array_of_subsizes[i]) );
        }

  /* can probably be more clever about this in situations where each proc will have the same subarray to recv */
        MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &data->recv_subarray[recv_count]) ;
        MPI_Type_commit(&data->recv_subarray[recv_count]);

        assert(&data->recv_subarray[recv_count] != NULL);

        recv_count++;
        local_coord[2] += data->recv_subsizes[index_2d(recv_count-1,2,3)];
      } 
      local_coord[2] = 0;
      local_coord[1] += data->recv_subsizes[index_2d(recv_count-1,1,3)];
    }
    local_coord[1] = 0;
    local_coord[0] += data->recv_subsizes[index_2d(recv_count-1,0,3)];
  }

  return recv_count;
}

void initialise_decomposition_swap(struct holder *data, int nlocal[], int num_procs, MPI_Comm ludwig_comm, int isize[]) {

  int nsize[3];
  int proc_dims[2];
  int rank;

  int overwrite, memsize[3];
  int istart[3], iend[3];
  int ip;
  int i;
 
  overwrite = 0; /* don't allow overwriting input of btran */

  MPI_Comm_rank(ludwig_comm, &rank);

  for(i=0; i<3; i++) {
    nsize[i] = N_total(2-i);
  }   

/* transferring data from cartesian to pencil */
  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
    if(rank == 0) { printf("already in pencil decomposition!\n"); }
    proc_dims[0] = cart_size(0);
    proc_dims[1] = cart_size(1); 

    p3dfft_setup(proc_dims, nsize[0], nsize[1], nsize[2], overwrite, memsize);

    ip = 1;
    p3dfft_get_dims(istart, iend, isize, ip);

    data->recv_array = malloc(isize[0]*isize[1]*isize[2]*sizeof(double));
    assert(isize[0] == nlocal[Z] && isize[1] == nlocal[Y] && isize[2] == nlocal[X]);

  }
  else { /*we need to do some work */
    if(rank == 0) { printf("Moving data from cartesian to pencil decomposition.\n"); }
/* It will be necessary to allow the user specify the grid they want */
    proc_dims[0] = num_procs/4;
    proc_dims[1] = 4;

    
/* for now will only consider pencil decompositions that divide evenly into the lattice */
    assert(nsize[1]%proc_dims[1] == 0);
    assert(nsize[2]%proc_dims[0] == 0);

    /* note that this routine swaps the elements of proc_dims */
    p3dfft_setup(proc_dims, nsize[0], nsize[1], nsize[2], overwrite, memsize);

    ip = 1;
    p3dfft_get_dims(istart, iend, isize, ip);

  /* now will create the subarrays for the mapping of data */
    int array_of_sizes[3], array_of_subsizes[3], array_of_starts[3];

    data->recv_array = malloc(isize[0]*isize[1]*isize[2]*sizeof(double));
/* finds the number of destination processors, and also sets subarray and find subsizes to be sent */
    data->send_count = find_number_cart_dest_procs(isize, proc_dims, nlocal, num_procs, data);
    
/* send the data */
    MPI_Request send_request[data->send_count];
    MPI_Status send_status[data->send_count];
    for(i=0; i<data->send_count; i++) {
      MPI_Isend(&data->send_subsizes[i*3], 3, MPI_INT, data->dest_proc[i], 1, ludwig_comm, &send_request[i]);
    }

/* finds the number of receiving processors, and also sets subarray and find subsizes to be received */
    data->recv_count = find_number_pencil_recv_procs(isize, istart, nlocal, num_procs, data, ludwig_comm);

  MPI_Waitall(data->send_count, send_request, send_status);

  }
}
