/*****************************************************************************
 *
 *  propagation_gpu.c
 *
 *  Propagation schemes for the different models.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *   Adapted to run on GPU: Alan Gray
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

/* below define needed to stop repeated declaration of distribution_ndist */
#define INCLUDING_FROM_GPU_SOURCE 

#include "model.h"
#include "propagation_gpu.h"

extern "C" void checkCUDAError(const char *msg);

/*****************************************************************************
 *
 *  propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

void propagation_gpu() {


  int ndist, nhalo;
  int N[3];
  static dim3 BlockDims;
  static dim3 GridDims;

  ndist = distribution_ndist();
  nhalo = coords_nhalo();
  coords_nlocal(N); 

  /* copy f to ftmp on accelerator */
  copy_f_to_ftmp_on_gpu();



  /* run the kernel */
  if (NVEL == 9){
    printf("propagate_d2q9 not yet supported in GPU mode\n");
    exit(1);
  }
  if (NVEL == 15){
    printf("propagate_d3q15 not yet supported in GPU mode\n");
    exit(1);
  }
  if (NVEL == 19){
  /* set up CUDA grid */
  #define BLOCKSIZE 256
  /* 1D decomposition - use x grid and block dimension only */ 
  BlockDims.x=BLOCKSIZE;
  GridDims.x=(N[X]*N[Y]*N[Z]+BlockDims.x-1)/BlockDims.x;
  propagate_d3q19_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo, 
  					      N_d,f_d,ftmp_d);
  cudaThreadSynchronize();

  }  
  return;
}

/*****************************************************************************
 *
 *  propagate_d3q19_gpu_d
 *
 *  Follows the velocities defined in d3q19.c
 *
 *  Adapted to run on GPU: Alan Gray
 *
 *****************************************************************************/


__global__ static void propagate_d3q19_gpu_d(int ndist, int nhalo, int N[3],
					     double* fnew_d, double* fold_d) {

  int ii, jj, kk, index, n, p, threadIndex, nsite, Nall[3];
  int xstr, ystr, zstr, pstr, xfac, yfac;


  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsite = Nall[X]*Nall[Y]*Nall[Z];

  zstr = 1;
  ystr = zstr*Nall[Z];
  xstr = ystr*Nall[Y];
  pstr = nsite*ndist; 


  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  /* Avoid going beyond problem domain */
   if (threadIndex < N[X]*N[Y]*N[Z])
     {

      /* calculate index from CUDA thread index */
      yfac = N[Z];
      xfac = N[Y]*yfac;
       
      ii = threadIndex/xfac;
      jj = ((threadIndex-xfac*ii)/yfac);
      kk = (threadIndex-ii*xfac-jj*yfac);
      
      index = get_linear_index_gpu_d(ii+1,jj+1,kk+1,Nall);
    
      for (n = 0; n < ndist; n++) {
	
	/* Distributions moving forward in memory. */

	fnew_d[9*pstr+n*nsite+index]=fold_d[9*pstr+n*nsite+index+                    (-1)];
	fnew_d[8*pstr+n*nsite+index]=fold_d[8*pstr+n*nsite+index+          (-1)*ystr+(+1)];
	fnew_d[7*pstr+n*nsite+index]=fold_d[7*pstr+n*nsite+index+         +(-1)*ystr     ];
	fnew_d[6*pstr+n*nsite+index]=fold_d[6*pstr+n*nsite+index+         +(-1)*ystr+(-1)];
	fnew_d[5*pstr+n*nsite+index]=fold_d[5*pstr+n*nsite+index+(-1)*xstr+(+1)*ystr     ];
	fnew_d[4*pstr+n*nsite+index]=fold_d[4*pstr+n*nsite+index+(-1)*xstr          +(+1)];
	fnew_d[3*pstr+n*nsite+index]=fold_d[3*pstr+n*nsite+index+(-1)*xstr               ];
	fnew_d[2*pstr+n*nsite+index]=fold_d[2*pstr+n*nsite+index+(-1)*xstr          +(-1)];
	fnew_d[1*pstr+n*nsite+index]=fold_d[1*pstr+n*nsite+index+(-1)*xstr+(-1)*ystr     ];
	
	/* Distributions moving backward in memory. */  
	
	fnew_d[10*pstr+n*nsite+index]=fold_d[10*pstr+n*nsite+index+                    (+1)];
	fnew_d[11*pstr+n*nsite+index]=fold_d[11*pstr+n*nsite+index+          (+1)*ystr+(-1)];
	fnew_d[12*pstr+n*nsite+index]=fold_d[12*pstr+n*nsite+index+         +(+1)*ystr     ];
	fnew_d[13*pstr+n*nsite+index]=fold_d[13*pstr+n*nsite+index+         +(+1)*ystr+(+1)];
	fnew_d[14*pstr+n*nsite+index]=fold_d[14*pstr+n*nsite+index+(+1)*xstr+(-1)*ystr     ];
	fnew_d[15*pstr+n*nsite+index]=fold_d[15*pstr+n*nsite+index+(+1)*xstr          +(-1)];
	fnew_d[16*pstr+n*nsite+index]=fold_d[16*pstr+n*nsite+index+(+1)*xstr               ];
	fnew_d[17*pstr+n*nsite+index]=fold_d[17*pstr+n*nsite+index+(+1)*xstr          +(+1)];
	fnew_d[18*pstr+n*nsite+index]=fold_d[18*pstr+n*nsite+index+(+1)*xstr+(+1)*ystr     ];

	
      } 


     }


   
   return;
}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])
{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}

/* get 3d coordinates from the index on the accelerator */
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}
