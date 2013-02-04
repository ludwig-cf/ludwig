/****************************************************************************
 *
 *  phi_lb_coupler_gpu.cu
 * 
 *  Alan Gray
 * 
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "phi_lb_coupler_gpu.h"
#include "phi_lb_coupler_internal_gpu.h"

/* below define needed to stop repeated declaration of distribution_ndist */
#define INCLUDING_FROM_GPU_SOURCE 

#include "model.h"
#include "site_map.h"


/*****************************************************************************
 *
 *  phi_compute_phi_site
 *
 *  Recompute the value of the order parameter at all the current
 *  fluid sites (domain proper).
 *
 *  This couples the scalar order parameter phi to the LB distribution
 *  in the case of binary LB.
 *
 *****************************************************************************/

void phi_compute_phi_site_gpu() {

  int N[3], ndist, nhalo;

  if (distribution_ndist() == 1) return;

  assert(distribution_ndist() == 2);

  ndist = distribution_ndist();
  nhalo = coords_nhalo();
  coords_nlocal(N);

/* set up CUDA grid */
  coords_nlocal(N); 
  int nblocks=(N[X]*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;

  /* run the kernel */
  phi_compute_phi_site_gpu_d<<<nblocks,DEFAULT_TPB>>>(ndist, N_d,nhalo, 
							 f_d,	 
							 site_map_status_d, 
							 phi_site_d);
  cudaThreadSynchronize();


  return;
}


/*****************************************************************************
 *
 *  phi_compute_phi_site_gpu_d
 *
 *  Adapted to run on GPU: Alan Gray
 *
 *****************************************************************************/

__global__ void phi_compute_phi_site_gpu_d(int ndist, int N[3], int nhalo, 
					   double* f_d, 
					   char* site_map_status_d, 
					   double* phi_site_d)
{


  int ii,jj,kk,index,p,xfac,yfac,nsite,threadIndex, Nall[3];
  
  
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsite = Nall[X]*Nall[Y]*Nall[Z];

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
      
      if (site_map_status_d[index] == FLUID)
	{
	  
	  double rho = 0.0;
	  
	  for (p = 0; p < NVEL; p++) {
	    rho += f_d[ndist*nsite*p + nsite + index];
	  }
	  
	  phi_site_d[index] = rho;
	}
            
    }
}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])
{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}
