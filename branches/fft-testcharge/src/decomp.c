/*****************************************************************************
 *
 *  decomp.c
 *
 *  Swapping between cartesian and pencil processor decompositions
 *  Also controls inititialisation of P3DFFT.
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

#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "runtime.h"

#include "config.h"
#include "p3dfft.h"
#include "decomp.h"


typedef struct decomp_s decomp_t;
struct decomp_s {
  int *dest_proc;
  int  send_count;
  int *send_subsizes;
  MPI_Datatype *send_subarray;

  int *recv_proc;
  int recv_count;
  int *recv_subsizes;
  MPI_Datatype *recv_subarray;

};

/* No default state is given as it is highly dependent on grid size
 * and number of MPI tasks being used. */
static int initialised_                = 0;
static MPI_Comm comm                   = MPI_COMM_NULL;
static decomp_t *data                  = NULL;
static int nlocal[3]                   = {0, 0, 0};

static int pencil                      = 0;
static int proc_dims[3]                = {0, 0, 0};
static int isize[3]                    = {0, 0, 0};
static int istart[3]                   = {0, 0, 0};
static int fsize[3]                    = {0, 0, 0};
static int fstart[3]                   = {0, 0, 0};


static int index_2d (int x, int y, int size);
static void find_cart_global_coord(int global_coord[], int local_coord[]);
static void find_cart_subarray_subsizes(int array_of_subsizes[], int global_coord[]);
static void find_cart_subarray_starts(int array_of_starts[], int local_coord[]);
static void find_cart_subarray_sizes(int array_of_sizes[], int local_coord[]);
static int find_pencil_recv_proc (int global_coord[]);
static int find_cart_dest_proc(int global_coord[]);
static int find_number_cart_dest_procs();
static int find_number_pencil_recv_procs();
static int decomp_is_pencil(int num_procs, int n); /* determine if the decomposition is already pencil */
static void decomp_check(int nsize[]); /* Check the decomposition is acceptable */

/*****************************************************************************
 *
 *  initialise_decomposition_swap
 *
 *  Set up the Pencil decomposition for the current state.
 *  Also initialise P3DFFT, and create the data structures
 *  necessary to swap between cartesian and pencil
 *
 *****************************************************************************/

/* This routine could be improved by havung no communications, as performance testing indicates it would be faster.
 * Most of the framework to do this is already set up in this code and only changes to find_number_pencil_recv_procs and
 * decomp_init along with the contents on decomp_t would be necesssary. */
void decomp_init() {

  if(initialised_ == 0) { /* Avoid calling initialisation twice.  */
    int nsize[3];

    int iend[3], fend[3];
    int ip;
    int ia, ja;
    int tmp; /* To swap ordering of pencil grid back after call to p3dfft_init. */
    int n;
    int num_procs;

    int overwrite, memsize[3];
   
    overwrite = 1; /* Allow overwriting input of btran. This is more efficient.*/

    /* initialise the necessary static variables */
    comm = cart_comm();
    coords_nlocal(nlocal);
    data = malloc(sizeof(struct decomp_s));

    num_procs = pe_size();

    for (ia = 0; ia < 3; ia++) {
      nsize[ia] = N_total(2-ia);
    }   

    n = RUN_get_int_parameter_vector("pencil_grid", proc_dims);

    /* initialise transferring data from cartesian to pencil */

    /* if already in a pencil decomposition, the user has not specified an acceptable grid, or has specified the same grid */
    if(decomp_is_pencil(num_procs, n)) {

      pencil = 1;

      proc_dims[1] = cart_size(0);
      proc_dims[0] = cart_size(1); 

      decomp_check(nsize);

      /* note that this routine swaps the elements of proc_dims. It is possible to 
       * change this behaviour when compiling P3DFFT. */
      p3dfft_setup(proc_dims, nsize[0], nsize[1], nsize[2], overwrite, memsize);
      info("\nPencil decomposition:      %2d %2d\n", proc_dims[0], proc_dims[1]);

      ip = 1;
      p3dfft_get_dims(istart, iend, isize, ip);

      ip = 2;
      p3dfft_get_dims(fstart, fend, fsize, ip);


    }
    else {
      /* If the user has specified empty proc_dims in either direction, or the proc_dims do not 
       * match the number of MPI Tasks try and find the best one.
       * Otherwise will use the users proc_dims. */

      if(proc_dims[0] == 0 || proc_dims[1] == 0 || n == 0 || proc_dims[0]*proc_dims[1] != num_procs) {
        proc_dims[0] = proc_dims[1] = 0;
        MPI_Dims_create(num_procs, 2, proc_dims);
      }

      /* p3dfft_setup swaps the elements of proc_dims. We swap them first so the result is
       * is what we want.
       * It is possible to change this behaviour when compiling P3DFFT though it is not recommended. */
      tmp = proc_dims[0];
      proc_dims[0] = proc_dims[1];
      proc_dims[1] = tmp;
      
      decomp_check(nsize);

      p3dfft_setup(proc_dims, nsize[0], nsize[1], nsize[2], overwrite, memsize);
      info("\nPencil decomposition:      %2d %2d\n", proc_dims[0], proc_dims[1]);

      /* Determine local array sizes.
       * ip = 1 corresponds to non transformed array
       * ip = 2 is for the transformed array. */
      ip = 1;
      p3dfft_get_dims(istart, iend, isize, ip);

      ip = 2;
      p3dfft_get_dims(fstart, fend, fsize, ip);

      /* Compute the number of destination processors while in Cartesian decomposition.
       * This routine also creates an array of the destination processors and the subarrays
       * required for switching decompositions. */
      data->send_count = find_number_cart_dest_procs();
      
      /* Send the subsizes. */
      MPI_Request send_request[data->send_count];
      MPI_Status send_status[data->send_count];
      for (ia = 0; ia < data->send_count; ia++) {
        MPI_Isend(&data->send_subsizes[ia*3], 3, MPI_INT, data->dest_proc[ia], 1, comm, &send_request[ia]);
      }

      /* Compute the number of receiving processors while in Pencil decomposition.
       * This routine also creates an array of the source processors and the subarrays
       * required for switching decompositions.
       * MPI_Recv calls are included to receive the subsizes. */
      data->recv_count = find_number_pencil_recv_procs();

    MPI_Waitall(data->send_count, send_request, send_status);

    }
    initialised_ = 1;
  }

  return;
}

/*****************************************************************************
 *
 *  decomp_cart_to_pencil
 *
 *  Swap from cartesian to pencil decomposition.
 *  Note: arrays must be allocated before this call
 *
 *****************************************************************************/

void decomp_cart_to_pencil(double *in_array, double *out_array) {

  int ia, ja, ka;
  
  assert(initialised_);
  /* Production runs will have assert turned off. */
  if(initialised_ == 0) {
    fatal("error: decomposition switching not initialised; call decomp_init first\n");
  }

  
  if(pencil) { /* already in a pencil decomposition! */
  /* copy data, remember in_array has halos and isize is Fortran ordered! */
    for (ia = 1; ia <= nlocal[X]; ia++) {
      for (ja = 1; ja <= nlocal[Y]; ja++) {
        for (ka = 1; ka <= nlocal[Z]; ka++) {
          out_array[index_3d_f(ia-1,ja-1,ka-1,isize)] = in_array[coords_index(ia,ja,ka)];
        }
      }
    }
  }
  else { /* swap decompositions, send and receive from the list of processors calculated in initialisation. */
    MPI_Request send_request[data->send_count];
    MPI_Status send_status[data->send_count];
    MPI_Status recv_status;
    for (ia = 0; ia < data->send_count; ia++) {
      MPI_Isend(in_array, 1, data->send_subarray[ia], data->dest_proc[ia], 2, comm, &send_request[ia]);
    }

    for (ia = 0; ia < data->recv_count; ia++) {
      MPI_Recv(out_array, 1, data->recv_subarray[ia], data->recv_proc[ia], 2, comm, &recv_status);
    }

    MPI_Waitall(data->send_count, send_request, send_status);
  }

  return;
}


/*****************************************************************************
 *
 *  decomp_pencil_to_cart
 *
 *  Swap from pencil to cartesian decomposition.
 *  Note: arrays must be allocated before this call
 *
 *****************************************************************************/

void decomp_pencil_to_cart(double *in_array, double *out_array) {

  int ia, ja, ka;
      
  assert(initialised_);
  /* Production runs will have assert turned off. */
  if(initialised_ == 0) {
    fatal("error: decomposition switching not initialised; call decomp_init first\n");
  }

  if(pencil) { /* already in a pencil decomposition! */
/* copy data back, remember out_array has halos! */
    for (ia = 1; ia <= nlocal[X]; ia++) {
      for (ja = 1; ja <= nlocal[Y]; ja++) {
        for (ka = 1; ka <= nlocal[Z]; ka++) {
          out_array[coords_index(ia,ja,ka)] = in_array[index_3d_f(ia-1,ja-1,ka-1,isize)];
        }
      }
    }

  }
  else { 
    /* swap decompositions, send and receive from the list of processors calculated in initialisation. 
     * swap is in opposite direction so receives and sends are switched from before. */
    MPI_Request pencil_cart_request[data->recv_count];
    MPI_Status pencil_cart_status[data->recv_count];
    MPI_Status status;

    for (ia = 0; ia < data->recv_count; ia++) {
      MPI_Isend(in_array, 1, data->recv_subarray[ia], data->recv_proc[ia], 2, comm, &pencil_cart_request[ia]);
    }
    for (ia = 0; ia < data->send_count; ia++) {
      MPI_Recv(out_array, 1, data->send_subarray[ia], data->dest_proc[ia], 2, comm, &status);
    }

    MPI_Waitall(data->recv_count, pencil_cart_request, pencil_cart_status);
  }

  return;
}

/*****************************************************************************
 *
 *  find_pencil_recv_proc
 *
 *  returns the processor to receive from, given a global coordinate
 *
 *  The coordinates of the destination processor are given by the number of 
 *  full local Cartesian arrays that fit in the global array before
 *  the current point in a given direction. 
 *
 *  The MPI rank is then computed from this and cart_size. 
 *
 *****************************************************************************/

int find_pencil_recv_proc (int global_coord[]) {
 
  int ia;
  int coords[3] = {0,0,0};
  int recv_proc;

  for (ia = 0; ia < 3; ia++) {
    coords[ia] = global_coord[ia]/nlocal[ia];
  }

  recv_proc = (coords[0]*cart_size(1) + coords[1])*cart_size(2) + coords[2];  

  return recv_proc;
}

/*****************************************************************************
 *
 *  index_2d
 *
 *  returns the index of a 2 dimensional malloc'd array.
 *  size should be the extent in the y dimension.
 *
 *****************************************************************************/

int index_2d (int x, int y, int size) {

  return x*size + y; 
}

/*****************************************************************************
 *
 *  index_3d_f
 *
 *  returns the index of a 3 dimensional malloc'd array.
 *  size[] has fortran ordering. That is size[0] is the size of the
 *  contiguous direction.
 *
 *****************************************************************************/

int index_3d_f (int x, int y, int z, int size[]) {

  return (x*size[1] + y)*size[0] + z; 
}

/*****************************************************************************
 *
 *  index_3d_c
 *
 *  returns the index of a 3 dimensional malloc'd array.
 *  size[] has c ordering. That is size[2] is the size of the
 *  contiguous direction.
 *
 *****************************************************************************/

int index_3d_c (int x, int y, int z, int size[]) {

  return (x*size[1] + y)*size[2] + z; 
}

/*****************************************************************************
 *
 *  find_cart_global_coord
 *
 *  finds the global coordinates of a given point in the local array in the
 *  cartesian decomposition
 *
 *****************************************************************************/

void find_cart_global_coord(int global_coord[], int local_coord[]) {
  int ia;
  int n[3];
  coords_nlocal_offset(n);

  for (ia = 0; ia < 3; ia++) {
    global_coord[ia] = n[ia] + local_coord[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  find_cart_subarray_subsizes
 *
 *  finds the subarray subsizes for sending and receiving in the cartesian 
 *  decomposition
 *
 * The subarray sizes are:
 * the size of the local pencil array minus the offset of the current point
 * from the edge of the local pencil array. If this is bigger than the local
 * array, send all local array elements. 
 *
 *****************************************************************************/

void find_cart_subarray_subsizes(int array_of_subsizes[], int global_coord[]) {

  int ia;
  for (ia = 0; ia < 3; ia++) {
    /* In the non contiguous direction need to compute the sizes. */
    if(ia != 2) {
      array_of_subsizes[ia] = isize[2-ia] - (global_coord[ia]%isize[2-ia]);
      if(array_of_subsizes[ia] > nlocal[ia]) {
        array_of_subsizes[ia] = nlocal[ia];
      }
    }
    /* In the contiguous direction, the entire set of local elements
     * will need to be sent. */
    else { 
      array_of_subsizes[ia] = nlocal[ia];
    }
  }
  
  return;
}

/*****************************************************************************
 *
 *  find_cart_subarray_starts
 *
 *  finds the subarray starts for sending and receiving in the cartesian 
 *  decomposition
 *
 *  The starts are the local coordinate plus the size of the halo in the non
 *  contiguous memory direction, and simply after the halo in the contiguous.
 *
 *****************************************************************************/

void find_cart_subarray_starts(int array_of_starts[], int local_coord[]) {

  int ia;
  for (ia = 0; ia < 3; ia++) {
    if(ia != 2) {
      array_of_starts[ia] = local_coord[ia] + coords_nhalo();
    }
    else {
      array_of_starts[ia] = coords_nhalo();
    }
  }

  return;
}

/*****************************************************************************
 *
 *  find_cart_subarray_sizes
 *
 *  finds the subarray sizes for sending and receiving in the cartesian 
 *  decomposition, i.e. the size of the local Cartesian array.
 *
 *****************************************************************************/

static void find_cart_subarray_sizes(int array_of_sizes[], int local_coord[]) {

  int ia;
  for (ia = 0; ia < 3; ia++) {
    array_of_sizes[ia] = nlocal[ia] + 2;
  }

  return;
}

/*****************************************************************************
 *
 *  find_cart_dest_proc
 *
 *  Returns the processor to send to, given a global coordinate in
 *  the cartesian decomposition
 *
 *  The coordinates of the destination processor are given by the number of 
 *  full local pencil arrays that fit in the global array before
 *  the current point in a given direction. 
 *
 *  The MPI rank is then computed from this and proc_dims. 
 *
 *****************************************************************************/

int find_cart_dest_proc(int global_coord[]) {

  int ia;
  int fft_proc_coord[2];

  for ( ia = 0; ia < 2; ia++ ) {
    fft_proc_coord[ia] = global_coord[ia]/isize[2-ia]; 
    assert(fft_proc_coord[ia] < proc_dims[ia] && fft_proc_coord >= 0);
  }

  return fft_proc_coord[0]*proc_dims[1] + fft_proc_coord[1];
}

/*****************************************************************************
 *
 *  find_number_cart_dest_procs
 *
 *  Returns the number of processes that need to have messages sent to them.
 *  While doing this it creates the subarrays for sending to each of these
 *
 *****************************************************************************/

int find_number_cart_dest_procs() {

  int global_coord[3] = {0, 0, 0};
  int local_coord[3] = {0, 0, 0};
  int coords[3];
  int array_of_subsizes[3], array_of_starts[3], array_of_sizes[3];
  int send_count = 0;
  int iter = 0;
  int ia;

  for (ia = 0; ia < 3; ia++) {
    coords[ia] = cart_coords(ia);
  }

/* first find how many messages need to be sent */
  while(local_coord[0] < nlocal[0]) {
    while(local_coord[1] < nlocal[1]) {
      send_count ++;
      find_cart_global_coord(global_coord, local_coord);
      find_cart_subarray_subsizes(array_of_subsizes, global_coord);

      /* Skip forward the size of the subarray to avoid processing every point. */
      local_coord[1] += array_of_subsizes[1];
    }
    local_coord[1] = 0;
    local_coord[0] += array_of_subsizes[0];
  }

  /* can then assign memory for dest_proc[] and send_subarray[] */
  data->dest_proc = malloc(send_count*sizeof(int));
  data->send_subarray = malloc(send_count*sizeof(MPI_Datatype));
  data->send_subsizes = malloc(3*send_count*sizeof(int));

  for (ia = 0; ia < 3; ia++) {
    local_coord[ia] = 0;
    global_coord[ia] = 0;
  }

  iter = 0;
  /* Compute dest_proc[] and subarray sizes.
   * Also set up the subarrays for each destination processor. 
   * Again we jump in subsizes. */
  while(local_coord[0] < nlocal[0]) {
    while(local_coord[1] < nlocal[1]) {

      find_cart_global_coord(global_coord, local_coord);
      find_cart_subarray_subsizes(array_of_subsizes, global_coord);
      find_cart_subarray_starts(array_of_starts, local_coord);
      find_cart_subarray_sizes(array_of_sizes, nlocal);

      for(ia = 0; ia < 3; ia++) {
        assert(array_of_subsizes[ia] >= 1);
        assert(array_of_subsizes[ia] <= array_of_sizes[ia]);
        assert(array_of_starts[ia] >= 0);
        assert(array_of_starts[ia] <= (array_of_sizes[ia] - array_of_subsizes[ia]) );
      }

      data->dest_proc[iter] = find_cart_dest_proc(global_coord);
      assert(data->dest_proc[iter] < num_procs);

      /* set up subarray to be sent */
      MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &data->send_subarray[iter]);
      MPI_Type_commit(&data->send_subarray[iter]);

      assert(&data->send_subarray[send_count] != NULL);
      
      for (ia = 0; ia < 3; ia++) {
        data->send_subsizes[index_2d(iter,ia,3)] = array_of_subsizes[ia];
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

/*****************************************************************************
 *
 *  find_number_pencil_recv_procs
 *
 *  Returns the number of processes that will send messages to the calling
 *  process.
 *  While doing this it creates the subarrays for receiving from each of these
 *  by receiving messages from each of them detailing the subsizes
 *
 *****************************************************************************/

int find_number_pencil_recv_procs() {

    int ia, recv_count;
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
        /* If arrays are full, reallocate */
        if(recv_count == array_size) {
          data->recv_subsizes = realloc(data->recv_subsizes, 3*2*array_size*sizeof(int));
          data->recv_subarray = realloc(data->recv_subarray, 2*array_size*sizeof(MPI_Datatype));
          data->recv_proc = realloc(data->recv_proc, 2*array_size*sizeof(int));
          array_size = array_size * 2;
        }
        for (ia = 0; ia < 3; ia++) {
          /* istart is global coord in fortran code (ie starts at 1)*/
          global_coord[ia] = istart[2-ia] - 1 + local_coord[ia];
        }
        data->recv_proc[recv_count] = find_pencil_recv_proc(global_coord);
        assert(data->recv_proc[recv_count] < num_procs);

        /* receive the messages detailing sizes */
        MPI_Recv(&data->recv_subsizes[index_2d(recv_count,0,3)], 3, MPI_INT, data->recv_proc[recv_count], 1, comm, &status);
        assert(data->recv_subsizes[index_2d(recv_count,2,3)] <= isize[0] && data->recv_subsizes[index_2d(recv_count,1,3)] <= isize[1] && data->recv_subsizes[index_2d(recv_count,0,3)] <= isize[2]);

        /* create subarray for receiving */
        for (ia = 0; ia < 3; ia++) {
          array_of_starts[ia] = local_coord[ia];
          array_of_subsizes[ia] = data->recv_subsizes[index_2d(recv_count,ia,3)];
          array_of_sizes[ia] = isize[2-ia];
          assert(array_of_starts[ia] >= 0);
          assert(array_of_starts[ia] <= (array_of_sizes[ia] - array_of_subsizes[ia]) );
        }

  /* can probably be more clever about this in situations where each proc will have the same subarray to recv */
        MPI_Type_create_subarray(3, array_of_sizes, array_of_subsizes, array_of_starts, MPI_ORDER_C, MPI_DOUBLE, &data->recv_subarray[recv_count]) ;
        MPI_Type_commit(&data->recv_subarray[recv_count]);

        assert(&data->recv_subarray[recv_count] != NULL);

        recv_count++;
        local_coord[2] += array_of_subsizes[2];
      } 
      local_coord[2] = 0;
      local_coord[1] += array_of_subsizes[1];
    }
    local_coord[1] = 0;
    local_coord[0] += array_of_subsizes[0];
  }

  return recv_count;
}

/***************************************************************************************
 *
 * decomp_pencil_sizes
 *
 * gives the extents of the local arrays on each processor
 * important to note that the ordering is fortran and this does not attempt to change it
 * That is: isize[0] is the size of the direction that is contiguous in memory
 *
 * ip == 1 corresponds to non-transformed arrays
 * ip == 2 corresponds to transformed arrays
 *
 ****************************************************************************************/
void decomp_pencil_sizes(int size[3], int ip) {

  int ia;

  assert(ip == 1 || ip == 2);
  if(ip == 1) {
    for (ia = 0; ia < 3; ia++) {
      size[ia] = isize[ia];
    }
  }
  else if(ip == 2) {
    for (ia = 0; ia < 3; ia++) {
      size[ia] = fsize[ia];
    }
  }

  return;
}

/***************************************************************************************
 * pencil_starts
 *
 * gives the global starts of the local arrays on each processor, 0 indexed
 * important to note that the ordering is fortran and this does not attempt to change it
 * That is: istart[0] is the start of the direction that is contiguous in memory
 *
 * ip == 1 corresponds to non-transformed arrays
 * ip == 2 corresponds to transformed arrays
 *
 ****************************************************************************************/
void decomp_pencil_starts(int start[3], int ip) {

  int ia;

  if(ip == 1) {
    for (ia = 0; ia < 3; ia++) {
      start[ia] = istart[ia];
    }
  }
  else if(ip == 2) {
    for (ia = 0; ia < 3; ia++) {
      start[ia] = fstart[ia];
    }
  }

  return;
}

/***************************************************************************************
 *
 * decomp_fftarr_size
 *
 * returns the size of the larger of the arrays involved in the fft routines
 * this allows for only one array to be allocated.
 * f is multiplied by 2 as it is an array of complex numbers and this is not taken into
 * account by the library returning the sizes.
 *
 ****************************************************************************************/

int decomp_fftarr_size() {

  int i,f;

  i = isize[0]*isize[1]*isize[2];
  f = fsize[0]*fsize[1]*fsize[2]*2;
  
  if(f >= i) {
    return f;
  }
  else {
    return i;
  }

}

/***************************************************************************************
 * decomp_is_pencil
 *
 * Checks if decomposition switching is necessary.
 * It will not be necessary if the Cartesian is 2d and no pencil grid has been 
 * correctly specified by the user, or if the pencil grid exactly matches
 * the Cartesian one.
 *
 ****************************************************************************************/

int decomp_is_pencil(int num_procs, int n) {

  /* If the original decomposition is 2 dimensional. */
  if(cart_size(2) == 1) {
    /* If the pencil dimensions have not been set or set incorrectly. */
    if(proc_dims[0] == 0 || proc_dims[1] == 0 || n == 0 || proc_dims[0]*proc_dims[1] != num_procs) {
      return 1;
    }
    /* or if the users pencil dimensions exactly match the cartesian ones. */
    else if(cart_size(1) == proc_dims[1] && cart_size(0) == proc_dims[0]) {
      return 1;
    }
  }
  else {
    return 0;
  }

}

/***************************************************************************************
 *
 * decomp_check
 *
 * Determines whether the decomposition is suitable for the grid.
 *
 * Computing the FFTs incolves rotating the decomposition through all
 * 3 possibilities. Thus is must be suitable for all of these.
 * e.g. a 64x64x8 grid. We could decompose the 64x64 face onto 4096 MPI Tasks.
 * The 64x8 face can only be decomposed onto 512 MPI Tasks.
 *
 ****************************************************************************************/

void decomp_check(int nsize[]) {
  
  int ia, ja;

  /* Will only consider pencil decompositions that divide evenly into the lattice */
  assert(nsize[1]%proc_dims[0] == 0);
  assert(nsize[2]%proc_dims[1] == 0);
  assert(nsize[2]%proc_dims[0] == 0);

  for (ia = 0; ia < 3; ia++) {
    for (ja = 0; ja < 2 ; ja++) {
      if(nsize[ia]/proc_dims[ja] < 1) {
        fatal("error: nsites too small for chosen number of processors. Please use a bigger grid or less processors\n");
      }
    }
  }

  return;
}

/***************************************************************************************
 *
 * decomp_finish
 *
 * cleans up memory. Should be called after all other decomp calls
 *
 ****************************************************************************************/

void decomp_finish() {

  assert(initialised_);
  int ia;
  
  if(!pencil) {
    free(data->dest_proc);
    free(data->send_subsizes);
    free(data->send_subarray);
    free(data->recv_proc);
    free(data->recv_subsizes);
    free(data->recv_subarray);
  }

  free(data);

  p3dfft_clean();

  /* reset static file scoped variables */
  comm            = MPI_COMM_NULL;
  data            = NULL;
  pencil          = 0;
    
  for (ia = 0; ia < 3; ia++) {
    nlocal[ia]    = 0;
    proc_dims[ia] = 0;
    isize[ia]     = 0;
    istart[ia]    = 0;
    fsize[ia]     = 0;
    fstart[ia]    = 0;
  }

  initialised_   = 0;

  return;
}

