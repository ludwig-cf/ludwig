/*****************************************************************************
 *
 *  psi_fft.c
 *
 *  A solution of the Poisson equation for the potenial and
 *  charge densities stored in the psi_t object using the P3DFFT library
 *  and the decomp routines.
 *
 *  The Poisson equation looks like
 *
 *    nabla^2 \psi = - rho_elec / epsilon
 *
 *  where psi is the potential, rho_elec is the free charge density, and
 *  epsilon is a permeability.
 *
 *  $Id: psi_fft.c,v 0.2 2013-06-13 12:40:02 ruairi Exp $
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
#include "psi_s.h"
#include "psi_fft.h"

/*decomposition switching*/
#include "decomp.h"

/*needed to determine whether the stride1 flag is set*/
#include "config.h"


static void solve_poisson(double *transf_array, double epsilon, int fstart[], int fsize[]);
static void solve_poisson_alt(double *transf_array, double epsilon, int fstart[], int fsize[]);

/*****************************************************************************
 *
 *  psi_fft_poisson
 *
 *
 *  First attempt.
 *
 *  Requires decomp_init to have been called before this function is called
 *  Transfers the data from cartesian to pencil decompositions, and then performs
 *  FFTs. The solve_poisson function does the actual solve and then the data 
 *  is transformed back to the original layout.
 *
 *  Original implementation was found to be lacking accuracy in some cases.
 *  Though physical simulation results have been correct.
 *
 *  A new algorithm is included in solve_poisson_alt. This shows very close
 *  agreement with the SOR solver but has not been rigorously tested.
 *
 *****************************************************************************/

int psi_fft_poisson(psi_t *obj) {
  
  int i, j, k;
  int index;
  int nlocal[3];

  int ip; /* Used to request pencil array sizes from decomp module. */
  int fstart[3], fsize[3]; /* Arrays to store sizes of Fourier transformed array. */
  /* These tell P3DFFT what kind of Fourier Transforms to do */
  unsigned char op_f[3]="fft", op_b[3]="tff";

  /* Could declare these as static to improve efficiency */
  double *pencil_array;
  double *rho_elec;

  pencil_array = malloc(decomp_fftarr_size() * sizeof(double));
  rho_elec = malloc(coords_nsites()*sizeof(double));
  /* set up initial data to be solved, with halos*/
  /* could strip them off here and make the subarrays easier
   * Alternatively, could do set this up on a pencil array instead
   * of a cartesian one.*/

  coords_nlocal(nlocal);

  for(i=1; i<=nlocal[X]; i++) {
    for(j=1; j<=nlocal[Y]; j++) {
      for(k=1; k<=nlocal[Z]; k++) {
        index = coords_index(i, j, k);
        psi_rho_elec(obj, index, &rho_elec[index]);
      }
    }
  }

  /* Find the sizes of the arrays, needed in order to
   * solve the equation. */
  ip = 2;
  decomp_pencil_sizes(fsize, ip);
  decomp_pencil_starts(fstart, ip);


  /* Switch to pencil decomposition*/
  decomp_cart_to_pencil(rho_elec, pencil_array);

  /* Perform forward FFT. The same array is used for input and output
   * to save memory. In both calls the first argument is the input
   * and the second is the output. */
  p3dfft_ftran_r2c(pencil_array, pencil_array, op_f);

  solve_poisson(pencil_array, obj->epsilon, fstart, fsize);
  /*solve_poisson_alt(pencil_array, obj->epsilon, fstart, fsize);*/
  
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
 *  Note that we use the following:
 *  kx=(2*pi*x/Nx)
 *  k^2 = kx^2 + ky^2 + kz^2
 *
 *  following Press et al., which is the convention used by FFTW, and thus P3DFFT,
 *  if we have i = 0, .., nx-1 then we get:
 *  0 frequency at i = 0,
 *  positive frequencies at            1 ... (nx/2) - 1
 *  negative fequencies at             (nx/2) + 1 ... n-1
 *  and
 *  frequency at nx/2 is periodic = +/- pi, below -pi.
 *  
 *****************************************************************************/

static void solve_poisson(double *transf_array, double epsilon, int fstart[], int fsize[]) {

  int i,j,k;
  int ix, iy, iz;
  int index;
  double kx, ky, kz;
  double k_square;
  double pi;
  int global_coord[3];
  int n_total[3] = {0, 0, 0};
  int farray_size[3] = {0, 0, 0};

  pi = 4.0*atan(1.0);


#ifdef STRIDE1
  /* stride 1 enabled, thus fsize and fstart array ordering is {x,y,z}
   * e.g. fsize[0] corresponds to the contiguous memory, and the x direction.
   * Would be nice to reduce the amount of code duplication but it is difficult
   * to maintain efficient ordering of the loops if that is done. */

  /* Find global address of first element. fsize[] does not account for the
   * fact that the array contains copmlex numbers, so the size corresponding
   * to the contiguous memory direction must be doubled. */
  for(i=0; i<3; i++) {
    global_coord[i] = fstart[i] - 1;
    n_total[i] = N_total(i);
    if(i == 0) {
      farray_size[i] = 2*fsize[i];
    }
    else {
      farray_size[i] = fsize[i];
    }
  }

  /* Loop over elements, computing and multiplying by 1/k at each lattice point*/
  for(i=0; i<farray_size[2]; i++) {
    iz = global_coord[Z] - n_total[Z]*((2*global_coord[Z])/n_total[Z]);
    kz = (2.0*pi/n_total[Z])*iz;
    for(j=0; j<farray_size[1]; j++) {
      iy = global_coord[Y] - n_total[Y]*((2*global_coord[Y])/n_total[Y]);
      ky = (2.0*pi/n_total[Y])*iy;
      for(k=0; k<farray_size[0]; k+=2) {
        ix = global_coord[X] - n_total[X]*((2*global_coord[X])/n_total[X]);
        kx = (2.0*pi/n_total[X])*ix;
        if(kz == 0 && ky == 0 && kx == 0) {
          k_square = 0;
        }
        else {
          k_square = 1/((kx*kx + ky*ky + kz*kz)*epsilon*(n_total[X]*n_total[Y]*n_total[Z]));
        }
        /* farray_size is a fortran ordered array so call the fortran index function*/
        index = index_3d_f(i,j,k,farray_size); 
        transf_array[index] = transf_array[index]*k_square;     /* real part */
        transf_array[index+1] = transf_array[index+1]*k_square; /*complex part */

        /* track the global array coordinate.
         * As fstart is a Fortran array it is one indexed. */ 
        global_coord[X] ++;
      }
      global_coord[X] = fstart[X] - 1;
      global_coord[Y] ++;
    }
    global_coord[Y] = fstart[Y] - 1;
    global_coord[Z] ++; 
  } 

#else
/*stride 1 not enabled, thus fsize and fstart array ordering is {z,y,x}
 * e.g. fsize[0] corresponds to the contiguous memory, and the z direction*/

  /* Find global address of first element. fsize[] does not account for the
   * fact that the array contains complex numbers, so the size corresponding
   * to the contiguous memory direction must be doubled. The ordering is preserved to 
   * improve cache usage.
   * fstart must have the opposite index ordering in order to correspond
   * correctly to the global coordinate.*/
  for(i=0; i<3; i++) {
    global_coord[i] = fstart[2-i] - 1;
    n_total[i] = N_total(i);
    if(i == 0) {
      farray_size[i] = 2*fsize[i];
    }
    else {
      farray_size[i] = fsize[i];
    }
  }

  /* This differs from above in the fact that the elements of farray_size refer to different
   * refer to different physical direction. Thus the ordering of the computation is changed. */ 
  for(i=0; i<farray_size[2]; i++) {
    ix = global_coord[X] - n_total[X]*((2*global_coord[X])/n_total[X]);
    kx = (2.0*pi/n_total[X])*ix;
    for(j=0; j<farray_size[1]; j++) {
      iy = global_coord[Y] - n_total[Y]*((2*global_coord[Y])/n_total[Y]);
      ky = (2.0*pi/n_total[Y])*iy;
      for(k=0; k<farray_size[0]; k+=2) {
        iz = global_coord[Z] - n_total[Z]*((2*global_coord[Z])/n_total[Z]);
        kz = (2.0*pi/n_total[Z])*iz;

        if(kz == 0 && ky == 0 && kx == 0) {
          k_square = 0;
        }
        else {
          /* Use this line to normalise the Fourier Transform */
          k_square = 1/((kx*kx + ky*ky + kz*kz)*epsilon*(n_total[X]*n_total[Y]*n_total[Z]));
        }

        /* farray_size is a fortran ordered array so call the fortran index function*/
        index = index_3d_f(i,j,k,farray_size);
        transf_array[index] = transf_array[index]*k_square;     /* real part */
        transf_array[index+1] = transf_array[index+1]*k_square; /* complex part */

        /* track the global array coordinate.
         * As fstart is a Fortran array it is one indexed.
         * Its ordering is also the standard Fortran ordering, meaning the elements
         * must be swapped when reading them. */ 
        global_coord[Z] ++;
      }
      global_coord[Z] = fstart[2-Z] - 1;
      global_coord[Y] ++;
    }
    global_coord[Y] = fstart[2-Y] - 1;
    global_coord[X] ++; 
  } 
#endif /*stride1*/

}

/*****************************************************************************
 *
 *  solve_poisson_alt
 *  
 * Alternative algorithm, we use:
 * theta(k) = h^2*sigma(k)/6 - 2cos(2*pi*coord[X]/nx) - 2cos(2*pi*coord[Y]/ny)
 *                                                    - 2cos(2*pi*coord[Z]/nz)
 * This is derived by applying Fourier Transforms to the 6 point stencil,
 * as used in the SOR solver. The resultinf exponents are then expressed
 * as Cosines in order to aid the computation.
 * The h^2 coefficient is exactly the inverse of the scaling that needs to be
 * applied to agree numerically with the SOR solver and thus does not appear.
 *
 * This gives much closer agreement to the SOR solver in the single solve case
 * but has not been rigorously tested.
 * It is currently included for information purposes only.
 *
 * We refer to the entire coefficient as k_square to be in keeping with 
 * the first algorithm.
 *
 *
 *****************************************************************************/

static void solve_poisson_alt(double *transf_array, double epsilon, int fstart[], int fsize[]) {

  int i,j,k;
  int ix, iy, iz;
  int index;
  int global_coord[3];
  int n_total[3] = {0, 0, 0};
  int farray_size[3] = {0, 0, 0};

  double kx, ky, kz;
  double k_square;
  double pi;
  double div[3] = 2.0*pi/n_total[X];
  double denom[3];
  pi = 4.0*atan(1.0);


#ifdef STRIDE1
/*stride 1 enabled, thus fsize and fstart array ordering is {x,y,z}
 * e.g. fsize[0] corresponds to the contiguous memory, and the x direction.
 * would be nice to reduce the amount of code duplication but it is difficult
 * to maintain good ordering of the loops if that is done*/

  /* Find global address of first element. fsize[] does not account for the
   * fact that the array contains copmlex numbers, so the size corresponding
   * to the contiguous memory direction must be doubled. 
   * Also compute the divides necessary in the Cosines to reduce total number
   * of divides. */
  for(i=0; i<3; i++) {
    global_coord[i] = fstart[i] - 1;
    n_total[i] = N_total(i);
    div[i] = 2.0*pi/n_total[i];
    if(i == X) {
      farray_size[i] = 2*fsize[i];
    }
    else {
      farray_size[i] = fsize[i];
    }
  }

  for(i=0; i<farray_size[2]; i++) {
    denom[Z] = 6 - 2.0*cos(div[Z]*global_coord[Z]);
    for(j=0; j<farray_size[1]; j++) {
      denom[Y] = denom[Z] - 2.0*cos(div[Y]*global_coord[Y]);
      for(k=0; k<farray_size[0]; k+=2) {
        denom[X] = denom[Y] - 2.0*cos(div[X]*global_coord[X]);
        if(denom[X] <= 1e-5) { /* need to ensure k_square is 0 when the denominator is 0. */
          k_square = 0;
        }
        else {
          /* Use this line to normalise the Fourier Transform */
          k_square = 1/(denom[X]*epsilon*(n_total[X]*n_total[Y]*n_total[Z]));
        }
        
        /* farray_size is a fortran ordered array so call the fortran index function*/
        index = index_3d_f(i,j,k,farray_size);
        transf_array[index] = transf_array[index]*k_square;     /* real part */
        transf_array[index+1] = transf_array[index+1]*k_square; /* complex part */

        /* track the global array coordinate.
         * As fstart is a Fortran array it is one indexed. */ 
        global_coord[X] ++;
      }
      global_coord[X] = fstart[X] - 1;
      global_coord[Y] ++;
    }
    global_coord[Y] = fstart[Y] - 1;
    global_coord[Z] ++; 
  }


#else
/*stride 1 not enabled, thus fsize and fstart array ordering is {z,y,x}
 * e.g. fsize[0] corresponds to the contiguous memory, and the z direction*/

/*alternative implementation for no stride1 defined not yet written.*/

#endif /*stride1*/

}
