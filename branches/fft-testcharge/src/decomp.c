/*****************************************************************************
 *
 *  decomp.c
 *
 *  Swapping between cartesian and pencil processor decompositions
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
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#include "coords.h"
#include "config.h"
#include "p3dfft.h"
#include "decomp.h"



static int initialised_                = 0;
static MPI_Comm ludwig_comm            = MPI_COMM_NULL;
static struct holder *data             = NULL;
static int nlocal[3]                   = {0, 0, 0};

static int pe_rank                     = 0;
static int num_procs                   = 0;
static int proc_dims[2]                = {0, 0};
static int isize[3]                    = {0, 0, 0};
static int istart[3]                   = {0, 0, 0};


static int find_recv_proc (int global_coord[]);
static int index_2d (int x, int y, int size);
static void find_cart_global_coord(int global_coord[], int local_coord[], int coords[]);
static void find_cart_subarray_subsizes(int array_of_subsizes[], int global_coord[]);
static void find_cart_subarray_starts(int array_of_starts[], int local_coord[]);
static int find_cart_dest_proc(int global_coord[]);
static int find_number_cart_dest_procs();
static int find_number_pencil_recv_procs();

void initialise_decomposition_swap(int input_proc_dims[]) {

  if(initialised_ == 0) {
    int nsize[3];

    int overwrite, memsize[3];
    int iend[3];
    int ip;
    int i;
   
    overwrite = 0; /* don't allow overwriting input of btran */
/* initialise the necessary static variables */
    ludwig_comm = cart_comm();
    pe_rank = cart_rank();
    num_procs = pe_size();
    coords_nlocal(nlocal);
    data = malloc(sizeof(struct holder));

/*have not allowed for user to change anything by passing input_proc_dims yet*/

    for(i=0; i<3; i++) {
      nsize[i] = N_total(2-i);
    }   

  /* transferring data from cartesian to pencil */
    if(cart_size(2) == 1) { /* already in a pencil decomposition! */
      if(pe_rank == 0) { printf("already in pencil decomposition!\n"); }
      proc_dims[1] = cart_size(0);
      proc_dims[0] = cart_size(1); 

      p3dfft_setup(proc_dims, nsize[0], nsize[1], nsize[2], overwrite, memsize);

      ip = 1;
      p3dfft_get_dims(istart, iend, isize, ip);

    }
    else { /*we need to do some work */
      if(pe_rank == 0) { printf("Moving data from cartesian to pencil decomposition.\n"); }
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

  /* finds the number of destination processors, and also sets subarray and find subsizes to be sent */
      data->send_count = find_number_cart_dest_procs();
      
  /* send the data */
      MPI_Request send_request[data->send_count];
      MPI_Status send_status[data->send_count];
      for(i=0; i<data->send_count; i++) {
        MPI_Isend(&data->send_subsizes[i*3], 3, MPI_INT, data->dest_proc[i], 1, ludwig_comm, &send_request[i]);
      }

  /* finds the number of receiving processors, and also sets subarray and find subsizes to be received */
      data->recv_count = find_number_pencil_recv_procs();

    MPI_Waitall(data->send_count, send_request, send_status);

    }
    initialised_ = 1;
  }

/*temporary measure until this takes into account input_proc_dims*/
  input_proc_dims[0] = proc_dims[0];
  input_proc_dims[1] = proc_dims[1];
}

void cart_to_pencil(double *send_array, double *recv_array) {

  int i,j,k;
  int num_procs;

  num_procs = pe_size();
  
  assert(initialised_);
  if(initialised_ == 0) {
    if(pe_rank == 0) { printf("error: decomposition switching not initialised; call decomp_init first\n"); }
      MPI_Abort(ludwig_comm, 1);
  }

  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
/* copy data or move pointer */
    for(i=0; i<nlocal[X]; i++) {
      for(j=0; j<nlocal[Y]; j++) {
        for(k=0; k<nlocal[Z]; k++) {
          recv_array[index_3d_f(i,j,k,isize)] = send_array[index_3d_c(i,j,k,nlocal)];
        }
      }
    }
  }
  else { /* swap decompositions */
    MPI_Request send_request[data->send_count];
    MPI_Status send_status[data->send_count];
    MPI_Status recv_status;
    for(i=0; i<data->send_count; i++) {
      MPI_Isend(send_array, 1, data->send_subarray[i], data->dest_proc[i], 2, ludwig_comm, &send_request[i]);
    }

    for(i=0; i<data->recv_count; i++) {
      MPI_Recv(recv_array, 1, data->recv_subarray[i], data->recv_proc[i], 2, ludwig_comm, &recv_status);
    }

    MPI_Waitall(data->send_count, send_request, send_status);
  }

}


  /*send the data back to cartesian decomp */
void pencil_to_cart(double *end_array, double *final_array) {

  int i,j,k;
      
  if(initialised_ == 0) {
    if(pe_rank == 0) { printf("error: decomposition switching not initialised; call cart_to_pencil before pencil_to_cart\n"); }
    MPI_Abort(ludwig_comm, 1);
  }

  if(cart_size(2) == 1) { /* already in a pencil decomposition! */
/* copy data back (or move pointer) */
    for(i=0; i<nlocal[X]; i++) {
      for(j=0; j<nlocal[Y]; j++) {
        for(k=0; k<nlocal[Z]; k++) {
          final_array[index_3d_c(i,j,k,nlocal)] = end_array[index_3d_f(i,j,k,isize)];
        }
      }
    }
  }
  else {
    if(pe_rank == 0) { printf("Transferring data back to cartesian processor decomposition\n"); }
    MPI_Request pencil_cart_request[data->recv_count];
    MPI_Status pencil_cart_status[data->recv_count];
    MPI_Status status;

    for(i=0; i<data->recv_count; i++) {
      MPI_Isend(end_array, 1, data->recv_subarray[i], data->recv_proc[i], 2, ludwig_comm, &pencil_cart_request[i]);
    }
    for(i=0; i<data->send_count; i++) {
      MPI_Recv(final_array, 1, data->send_subarray[i], data->dest_proc[i], 2, ludwig_comm, &status);
    }

    printf("hi\n");
    MPI_Waitall(data->recv_count, pencil_cart_request, pencil_cart_status);
  }
}


int find_recv_proc (int global_coord[]) {
 
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

/* find the index of a 2d malloc'd array, where size is the extent in the y dimension */
int index_2d (int x, int y, int size) {

  return x*size + y; 
}

/*
 *  returns the index of an array with direction z contiguous in memory and size[2] corrsponding the the extent of z
 */
int index_3d_f (int x, int y, int z, int size[]) {

  return (x*size[1] + y)*size[0] + z; 
}

/*
 *  returns the index of an array with direction z contiguous in memory and size[0] corrsponding the the extent of z
 */
int index_3d_c (int x, int y, int z, int size[]) {

  return (x*size[1] + y)*size[2] + z; 
}


void find_cart_global_coord(int global_coord[], int local_coord[], int coords[]) {
  int i;
  for(i=0; i<2; i++) { 
    global_coord[i] = (coords[i]*nlocal[i]) + local_coord[i];
  }
}

void find_cart_subarray_subsizes(int array_of_subsizes[], int global_coord[]) {
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

int find_cart_dest_proc(int global_coord[]) {
  int i;
  int fft_proc_coord[2];
  for(i=0; i<2; i++) {
    fft_proc_coord[i] = global_coord[i]/isize[2-i]; 
    assert(fft_proc_coord[i] < proc_dims[i] && fft_proc_coord >= 0);
  }
  return fft_proc_coord[0]*proc_dims[1] + fft_proc_coord[1];
}

/*returns numnber of destination processors */
int find_number_cart_dest_procs() {

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
      find_cart_global_coord(global_coord, local_coord, coords);
      find_cart_subarray_subsizes(array_of_subsizes, global_coord);


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

      find_cart_global_coord(global_coord, local_coord, coords);
      find_cart_subarray_subsizes(array_of_subsizes, global_coord);
      find_cart_subarray_starts(array_of_starts, local_coord);

      for(i=0; i<3; i++) {
        assert(array_of_starts[i] >= 0);
        assert(array_of_starts[i] <= (nlocal[i] - array_of_subsizes[i]) );
      }

      data->dest_proc[iter] = find_cart_dest_proc(global_coord);
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

int find_number_pencil_recv_procs() {

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
        data->recv_proc[recv_count] = find_recv_proc(global_coord);
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

/***************************************************************************************
 * pencil_sizes
 *
 * gives the extents of the local arrays on each processor
 * important to note that the ordering is fortran and this does not attempt to change it
 * That is: isize[0] is the size of the direction that is contiguous in memory
 *
 ****************************************************************************************/
void pencil_sizes(int size[3]) {
  int i;
  for(i=0; i<3; i++) {
    size[i] = isize[i];
  }
}

/***************************************************************************************
 * pencil_starts
 *
 * gives the global starts of the local arrays on each processor, 0 indexed
 * important to note that the ordering is fortran and this does not attempt to change it
 * That is: istart0] is the start of the direction that is contiguous in memory
 *
 ****************************************************************************************/
void pencil_starts(int start[3]) {
  int i;
  for(i=0; i<3; i++) {
    start[i] = istart[i];
  }
}

