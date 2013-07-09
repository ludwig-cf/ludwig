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
#include <math.h>

#include "coords.h"
#include "decomp.h"

#include "psi_s.h"
#include "psi.h"


static void solve_poisson(double *transf_array, double epsilon, int fstart[], int fsize[]);

int psi_fft_poisson(psi_t *obj){
  

  int num_procs, rank;
  MPI_Comm ludwig_comm;
  MPI_Comm p3dfft_comm;

  int nlocal[3];
  int coords[3];
  int proc_dims[2];
  int i, j, k;
  int istart[3], iend[3], isize[3];
  int ip;
  int index;

  int fstart[3], fend[3], fsize[3];
  unsigned char op_f[3]="fft", op_b[3]="tff";

  ludwig_comm = cart_comm();
  num_procs = pe_size();
  rank = cart_rank();

  coords_nlocal(nlocal);

  double *recv_array;
  double *end_array;
  double *transf_array;

/* initial data to be solved, with halos
 * (could strip them off here and make the subarrays easier)
 */
  double *rho_elec = malloc(coords_nsites()*sizeof(double));
  double rho;

  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        index = coords_index(i, j, k);
        psi_rho_elec(obj, index, &rho);
        rho_elec[index] = rho;
      }
    }
  }

  decomp_initialise(proc_dims);

/*  if(pe_rank() == 0) { printf("printing array in psi_fft!\n"); }
  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        if(pe_rank() == 0) {
          printf("%8.5f\n", obj->psi[coords_index(i,j,k)]);
        }
//        assert(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)] < 1e-5);
      }
    }
  }*/

  decomp_pencil_sizes(isize); 
  decomp_pencil_starts(istart);

  recv_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));
  end_array = malloc(isize[0]*isize[1]*isize[2] * sizeof(double));

  ip = 2;
  p3dfft_get_dims(fstart, fend, fsize, ip);
  if(pe_rank() == 0) { printf("fsizes %d %d %d, isizes %d %d %d\n", fsize[0], fsize[1], fsize[2], isize[0], isize[1], isize[2]); }

  transf_array = malloc(2*fsize[0]*fsize[1]*fsize[2] * sizeof(double)); /* x2 since they will be complex numbers */

/*arrays need to be malloc'd before this call*/
  decomp_cart_to_pencil(rho_elec, recv_array);
  if(rank == 0) { printf("Proccessors now in pencil decomposition\n"); }

    for(k=0; k<isize[Z]; k++) {
      for(j=0; j<isize[Y]; j++) {
        for(i=0; i<isize[X]; i++) {
//          printf("%8.5f\n", recv_array[index_3d_f(k,j,i,isize)]);
        }
      }
    }

  if(rank == 0) { printf("Performing forward fft\n"); }
  p3dfft_ftran_r2c(recv_array, transf_array, op_f);

//  double *p = transf_array;
  int farray_size[3];

  for(i=0; i<3; i++) {
    if(i != X) {
      farray_size[i] = fsize[i];
    }
    else {
      farray_size[i] = 2*fsize[i];
    }
  }


  for(i=0; i<farray_size[Z]; i++) {
    for(j=0; j<farray_size[Y]; j++) {
      for(k=0; k<farray_size[X]; k+=2) {
        if(pe_rank() == 0) {
//          printf("%d %d %d %f %f\n", i,j,k,transf_array[index_3d_f(i,j,k,farray_size)], transf_array[index_3d_f(i,j,k+1,farray_size)]);
        }
//        assert(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)] < 1e-5);
      }
    }
  }

  solve_poisson(transf_array, obj->epsilon, fstart, fsize);


  if(pe_rank() == 0) printf("after k\n");
  for(i=0; i<farray_size[Z]; i++) {
    for(j=0; j<farray_size[Y]; j++) {
      for(k=0; k<farray_size[X]; k+=2) {
        if(pe_rank() == 0) {
//          printf("%d %d %d %f %f\n", i,j,k,transf_array[index_3d_f(i,j,k,farray_size)], transf_array[index_3d_f(i,j,k+1,farray_size)]);
        }
//        assert(psi_sor->psi[coords_index(i,j,k)] - psi_fft->psi[coords_index(i,j,k)] < 1e-5);
      }
    }
  }

  
  if(rank == 0) { printf("Performing backward fft\n"); }
  p3dfft_btran_c2r(transf_array, end_array, op_b);

/* checking the array is the same after transforming back also dividing by (nx*ny*nz)*/
      for(k=0; k<isize[Z]; k++) {
        for(j=0; j<isize[Y]; j++) {
          for(i=0; i<isize[X]; i++) {
            end_array[index_3d_f(k,j,i,isize)] = end_array[index_3d_f(k,j,i,isize)]/(N_total(0)*N_total(1)*N_total(2));
//        if(fabs(recv_array[index_3d_f(k,j,i,isize)] - end_array[index_3d_f(k,j,i,isize)] ) > 0.1e-10) {
//              printf("error: rank %d, end array[%d][%d][%d] is wrong, %f and should be %f\n", rank, k, j, i, end_array[index_3d_f(k,j,i,isize)], recv_array[index_3d_f(k,j,i,isize)]);
//            }
          }
        }
      }

  decomp_pencil_to_cart(end_array, obj->psi);


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
  free(rho_elec);
  free(recv_array);
  free(transf_array);
  free(end_array);

  return 0;
}

void solve_poisson(double *transf_array, double epsilon, int fstart[], int fsize[]) {

  int i,j,k;
  int ix, iy, iz;
  double kx, ky, kz;
  double k_square;
  double pi;
  int global_coord[3];
  int local_coord[3] = {0, 0, 0};
  int n_total[3] = {0, 0, 0};
  int farray_size[3] = {0, 0, 0};

  pi = 4.0*atan(1.0);

/*#ifdef STRIDE1*/
/*stride 1 enabled, thus fsize array ordering is {X,Y,Z}*/
/*first find global address of first element*/
  for(i=0; i<3; i++) {
    global_coord[i] = fstart[i] - 1;
    n_total[i] = N_total(i);
    if(i == X) {
      farray_size[i] = 2*fsize[i];
    }
    else {
      farray_size[i] = fsize[i];
    }
  }

/*loop over elements, computing and multiplying by 1/k at each*/
/* the ordering here is important, depending on whether P3DFFT was built
 * with stride 1 defined or not will change the sizes of each array dimension.
 * stride 1 allows the transposed data to be contiguous in the Z direction
 * (from Fortran's perspective)
 */
  for(i=0; i<farray_size[Z]; i++) {
    iz = global_coord[Z] - n_total[Z]*(2*global_coord[Z]/n_total[Z]);
    kz = (2.0*pi/n_total[Z])*iz;
    for(j=0; j<farray_size[Y]; j++) {
      iy = global_coord[Y] - n_total[Y]*(2*global_coord[Y]/n_total[Y]);
      ky = (2.0*pi/n_total[Y])*iy;
      for(k=0; k<farray_size[X]; k+=2) {
        ix = global_coord[X] - n_total[X]*(2*global_coord[X]/n_total[X]);
        kx = (2.0*pi/n_total[X])*ix;
        if(kz == 0 && ky == 0 && kx == 0) {
          k_square = 0;
        }
        else {
          k_square = 1/((kx*kx + ky*ky + kz*kz)*epsilon);
        }
        transf_array[index_3d_f(i,j,k,farray_size)] = -transf_array[index_3d_f(i,j,k,farray_size)]*k_square;
        /* complex numbers so need to multiply both parts of the number */
        transf_array[index_3d_f(i,j,k+1,farray_size)] = -transf_array[index_3d_f(i,j,k+1,farray_size)]*k_square;
        if(pe_rank() == 1) {
//          printf("global %d %d %d\n", global_coord[X], global_coord[Y], global_coord[Z]); 
//          printf("x:%2d %2d %8.5f y:%2d %2d %8.5f z:%2d %2d %8.5f %8.5f\n", k, ix, kx, j, iy, ky, i, iz, kz, k_square);
        }
        global_coord[X] ++;
      }
      global_coord[X] = fstart[X] - 1;
      global_coord[Y] ++;
    }
    global_coord[Y] = fstart[Y] - 1;
    global_coord[Z] ++; 
  } 

//#else
/*stride 1 not enabled, thus fsize array ordering is {Z,Y,X}*/
/*first find global address of first element*/
/*  for(i=0; i<3; i++) {
    global_coord[i] = fstart[i] - 1;
    n_total[i] = N_total(i);
    if(i == X) {
      farray_size[i] = 2*fsize[i];
    }
    else {
      farray_size[i] = fsize[i];
    }
  }
stride 1 not enabled so ordering of fsize array is now {Z,Y,X}.
  for(i=0; i<farray_size[Z]; i++) {
    ix = global_coord[X] - n_total[X]*(2*global_coord[X]/n_total[X]);
    kx = (2.0*pi/n_total[X])*ix;
    for(j=0; j<farray_size[Y]; j++) {
      iy = global_coord[Y] - n_total[Y]*(2*global_coord[Y]/n_total[Y]);
      ky = (2.0*pi/n_total[Y])*iy;
      for(k=0; k<farray_size[X]; k+=2) {
        iz = global_coord[Z] - n_total[Z]*(2*global_coord[Z]/n_total[Z]);
        kz = (2.0*pi/n_total[Z])*iz;

        if(kz == 0 && ky == 0 && kx == 0) {
          k_square = 0;
        }
        else {
          k_square = 1/((kx*kx + ky*ky + kz*kz)*epsilon);
        }
        transf_array[index_3d_f(i,j,k,farray_size)] = -transf_array[index_3d_f(i,j,k,farray_size)]*k_square;
        // complex numbers so need to multiply both parts of the number 
        transf_array[index_3d_f(i,j,k+1,farray_size)] = -transf_array[index_3d_f(i,j,k+1,farray_size)]*k_square;
        if(pe_rank() == 1) {
          printf("test global %d %d %d\n", global_coord[X], global_coord[Y], global_coord[Z]); 
          printf("x:%2d %2d %8.5f y:%2d %2d %8.5f z:%2d %2d %8.5f %8.5f\n", k, ix, kx, j, iy, ky, i, iz, kz, k_square);
        }
        global_coord[Z] ++;
      }
      global_coord[Z] = fstart[Z] - 1;
      global_coord[Y] ++;
    }
    global_coord[Y] = fstart[Y] - 1;
    global_coord[X] ++; 
  } 
#endif*/

}

void psi_fft_clean() {

  decomp_finish();
  p3dfft_clean();

}

