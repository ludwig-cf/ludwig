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


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "config.h"
#include "decomp.h"

#include "psi_s.h"
#include "psi.h"


static void solve_poisson(double *transf_array, double epsilon, int fstart[], int fsize[]);

/*****************************************************************************
 *
 *  psi_fft_poisson
 *
 *
 *  First attempt.
 *
 *  Requires best decomp_init to have been called before this function is called
 *  Transfers the data from cartesian to pencil decompositions, and then performs
 *  FFTs. The solve_poisson function does the actual solve and then the data 
 *  is transformed back to the original layout
 *
 *****************************************************************************/

int psi_fft_poisson(psi_t *obj){
  

  int num_procs;
  MPI_Comm ludwig_comm;

  int nlocal[3];
  int i, j, k;
  int isize[3];
  int ip;
  int index;

  int fstart[3], fsize[3];
  unsigned char op_f[3]="fft", op_b[3]="tff";

  coords_nlocal(nlocal);

  double *pencil_array;

/* initial data to be solved, with halos
 * (could strip them off here and make the subarrays easier)
 */
  double *rho_elec;

  rho_elec = malloc(coords_nsites()*sizeof(double));

  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        index = coords_index(i, j, k);
        psi_rho_elec(obj, index, &rho_elec[index]);
//        rho_elec[index] = rho;

      }
    }
  }


  ip = 1;
  decomp_pencil_sizes(isize, ip); 

  ip = 2;
  decomp_pencil_sizes(fsize, ip);
  decomp_pencil_starts(fstart, ip);

  pencil_array = malloc(decomp_fftarr_size() * sizeof(double));

/*arrays need to be malloc'd before this call*/
  decomp_cart_to_pencil(rho_elec, pencil_array);

  p3dfft_ftran_r2c(pencil_array, pencil_array, op_f);

  solve_poisson(pencil_array, obj->epsilon, fstart, fsize);
  
  p3dfft_btran_c2r(pencil_array, pencil_array, op_b);

  decomp_pencil_to_cart(pencil_array, obj->psi);

  free(rho_elec);
  free(pencil_array);

  return 0;
}

/*****************************************************************************
 *
 *  solve_poisson
 *
 *  In Fourier Space, Poisson's equation looks like
 *
 *  Theta(k) = Sigma(k)/(k^2 epsilon)
 *
 *  Where Theta(k) = F(psi) and Sigma(k) = F(rho).
 *  Thus multiplying the transform of rho by 1/(k^2 epsilon) before transforming
 *  back to real space solves Poisson's equation
 *
 *
 *  Multiply the transformed array by 1/(k^2)epsilon in order to 
 *  solve Poisson's equation
 *
 *****************************************************************************/

void solve_poisson(double *transf_array, double epsilon, int fstart[], int fsize[]) {

  int i,j,k;
  int ix, iy, iz;
  double kx, ky, kz;
  double k_square;
  double pi;
  int global_coord[3];
  int n_total[3] = {0, 0, 0};
  int farray_size[3] = {0, 0, 0};

  pi = 4.0*atan(1.0);


//printf("rank %d, fstart: %d %d %d\n", pe_rank(), fstart[0], fstart[1], fstart[2]);

#ifdef STRIDE1
/*stride 1 enabled, thus fsize array ordering is {X,Y,Z}
 * => fsize[0] still corresponds to the contiguous memory, but the x direction*/

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

/*loop over elements, computing and multiplying by 1/k at each lattice point*/
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
          k_square = 1/((kx*kx + ky*ky + kz*kz)*epsilon*(N_total(0)*N_total(1)*N_total(2)));
        }
        transf_array[index_3d_f(i,j,k,farray_size)] = transf_array[index_3d_f(i,j,k,farray_size)]*k_square;
        /* complex numbers so need to multiply both parts of the number */
        transf_array[index_3d_f(i,j,k+1,farray_size)] = transf_array[index_3d_f(i,j,k+1,farray_size)]*k_square;

        global_coord[X] ++;
      }
      global_coord[X] = fstart[X] - 1;
      global_coord[Y] ++;
    }
    global_coord[Y] = fstart[Y] - 1;
    global_coord[Z] ++; 
  } 

#else
/*stride 1 not enabled, thus fsize array ordering is {Z,Y,X}
 * => fsize[0] still corresponds to the contiguous memory, but the z direction*/

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
        transf_array[index_3d_f(i,j,k,farray_size)] = transf_array[index_3d_f(i,j,k,farray_size)]*k_square;
        // complex numbers so need to multiply both parts of the number 
        transf_array[index_3d_f(i,j,k+1,farray_size)] = transf_array[index_3d_f(i,j,k+1,farray_size)]*k_square;

        global_coord[Z] ++;
      }
      global_coord[Z] = fstart[Z] - 1;
      global_coord[Y] ++;
    }
    global_coord[Y] = fstart[Y] - 1;
    global_coord[X] ++; 
  } 
#endif /*STRIDE1*/

}

