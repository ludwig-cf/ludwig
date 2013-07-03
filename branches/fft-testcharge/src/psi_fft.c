/*****************************************************************************
 *
 *  psi_fft.c
 *
 *  Use P3DFFT to conduct ffts and decomp.c to transfer between cartesian and
 *  pencil decompositions.
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

 /*
 * want to have a psi_fft function which can call the decomposition swap
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>

#include "coords.h"
#include "decomp.h"

#include "psi_s.h"
#include "psi.h"


static void multiply_by_k (double *array, int fstart[], int fsize[]);

int psi_fft_poisson(psi_t *obj){
  
//  MPI_Init(NULL, NULL);

  int num_procs, rank;
  MPI_Comm ludwig_comm;
  MPI_Comm p3dfft_comm;

  int nlocal[3];
  int coords[3];
  int proc_dims[2];
  int i, j, k;
  int istart[3], iend[3], isize[3];
  int ip;

  int fstart[3], fend[3], fsize[3];
  unsigned char op_f[3]="fft", op_b[3]="tff";

/* ludwig decomposition setup*/
//  pe_init();
//  coords_init();

  ludwig_comm = cart_comm();
  num_procs = pe_size();
  rank = cart_rank();

  coords_nlocal(nlocal);

  double *final_array;
  double *recv_array;
  double *end_array;
  double *transf_array;

/*
  psi_t *obj;
  obj = malloc(sizeof(psi_t));

  struct holder *data = malloc(sizeof(struct holder));

  obj->psi = malloc(nlocal[0]*nlocal[1]*nlocal[2] * sizeof(double));
  final_array = malloc(nlocal[0]*nlocal[1]*nlocal[2] * sizeof(double));

  for(k=0; k<nlocal[X]; k++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(i=0; i<nlocal[Z]; i++) {
        obj->psi[index_3d_c(k,j,i,nlocal)] = rank;
      }
    }
  }*/
  
  initialise_decomposition_swap(proc_dims);

  if(rank == 0) { printf("Proccessors now in pencil decomposition\n"); }

  pencil_sizes(isize); 
  pencil_starts(istart);

  recv_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));
  end_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));

  ip = 2;
  p3dfft_get_dims(fstart, fend, fsize, ip);

  transf_array = malloc(2*fsize[0]*fsize[1]*fsize[2] * sizeof(double)); /* x2 since they will be complex numbers */

/*arrays need to be malloc'd before this call*/
  cart_to_pencil(obj->psi, recv_array);

  if(rank == 0) { printf("Performing forward fft\n"); }
  p3dfft_ftran_r2c(recv_array, transf_array, op_f);

  multiply_by_k(transf_array, fstart, fsize);
  
  if(rank == 0) { printf("Performing backward fft\n"); }
  p3dfft_btran_c2r(transf_array, end_array, op_b);

  printf("test print 1\n");

/* checking the array is the same after transforming back also dividing by (nx*ny*nz)*/
      for(k=0; k<isize[Z]; k++) {
        for(j=0; j<isize[Y]; j++) {
          for(i=0; i<isize[X]; i++) {
            end_array[index_3d_f(k,j,i,isize)] = end_array[index_3d_f(k,j,i,isize)]/(N_total(0)*N_total(1)*N_total(2));
//         if(abs(recv_array[index_3d_f(k,j,i,isize)] - end_array[index_3d_f(k,j,i,isize)] ) > 0.1e-10) {
//              printf("error: rank %d, end array[%d][%d][%d] is wrong, %f and should be %f\n", rank, k, j, i, end_array[index_3d_f(k,j,i,isize)], recv_array[index_3d_f(k,j,i,isize)]);
//            }
          }
        }
      }

  printf("test print 2\n");

  pencil_to_cart(obj->psi, final_array);


/*  for(k=0; k<nlocal[X]; k++) {
    for(j=0; j<nlocal[Y]; j++) {
      for(i=0; i<nlocal[Z]; i++) {
        if(abs(obj->psi[index_3d_c(k,j,i,nlocal)] - final_array[index_3d_c(k,j,i,nlocal)]) > 1e-10) {
          printf("rank %d, error: final array[%d][%d][%d] is wrong, %f and should be %f\n", rank, k, j, i, final_array[index_3d_c(k,j,i,nlocal)], obj->psi[index_3d_c(k,j,i,nlocal)]);
        }
      }
    }
  }*/

  if(rank == 0) { printf("Data now in cartesian processor decomposition\n"); }


/* need to free memory */

//  MPI_Finalize();

  return 0;
}

void multiply_by_k (double *array, int fstart[], int fsize[]) {

  int i,j,k;
  int ix, iy, iz;
  double kx, ky, kz;
  double k_square;
  double pi;
  int global_coord[3];
  int local_coord[3] = {0, 0, 0};
  int n_total[3] = {0, 0, 0};

  pi = 4.0*atan(1.0);

/*first find global address of first element*/
  for(i=0; i<3; i++) {
    global_coord[i] = fstart[2-i] - 1 + local_coord[i];
    n_total[i] = N_total(i);
  }

/*loop over elements, computing and multiplying by 1/k at each*/
/* the ordering here is important, depending on whether P3DFFT was built
 * with stride 1 defined or not will change the sizes of each array dimension.
 * stride 1 allows the transposed data to be contiguous in the Z direction
 * (from Fortran's perspective)
 */
  for(i=0; i<fsize[Z]; i++) {
    global_coord[X] ++; 
    ix = i - n_total[X]*(2*global_coord[X]/n_total[X]);
    kx = (2.0*pi/n_total[X])*ix;
    for(j=0; j<fsize[Y]; j++) {
      global_coord[Y] ++;
      iy = j - n_total[Y]*(2*global_coord[Y]/n_total[Y]);
      ky = (2.0*pi/n_total[Y])*iy;
      for(k=0; k<fsize[X]*2; k+=2) {
        global_coord[Z] ++;
        iz = i - n_total[Z]*(2*global_coord[Z]/n_total[Z]);
        kz = (2.0*pi/n_total[Z])*iz;
        k_square = 1/(kx*kx + ky*ky + kz*kz);
        if(kx == 0 && ky == 0 && kz == 0) {
          k_square = 0;
        }
        array[index_3d_f(i,j,k,fsize)] = array[index_3d_f(i,j,k,fsize)]*k_square;
        /* complex numbers so need to multiply both parts of the number */
        array[index_3d_f(i,j,k+1,fsize)] = array[index_3d_f(i,j,k+1,fsize)]*k_square;
      }
    }
  } 

}
