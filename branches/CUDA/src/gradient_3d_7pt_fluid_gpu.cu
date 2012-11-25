/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_gpu.c
 *
 *  Gradient operations for 3D seven point stencil.
 *
 *                        (ic, jc+1, kc)
 *         (ic-1, jc, kc) (ic, jc  , kc) (ic+1, jc, kc)
 *                        (ic, jc-1, kc)
 *
 *  ...and so in z-direction
 *
 *  d_x phi = [phi(ic+1,jc,kc) - phi(ic-1,jc,kc)] / 2
 *  d_y phi = [phi(ic,jc+1,kc) - phi(ic,jc-1,kc)] / 2
 *  d_z phi = [phi(ic,jc,kc+1) - phi(ic,jc,kc-1)] / 2
 *
 *  nabla^2 phi = phi(ic+1,jc,kc) + phi(ic-1,jc,kc)
 *              + phi(ic,jc+1,kc) + phi(ic,jc-1,kc)
 *              + phi(ic,jc,kc+1) + phi(ic,jc,kc-1)
 *              - 6 phi(ic,jc,kc)
 *
 *  Corrections for Lees-Edwards planes and plane wall in X are included.
 *
 *  $Id: gradient_3d_7pt_fluid.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 * Adapted to run on GPU: Alan Gray
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "gradient_3d_7pt_fluid_gpu.h"


void phi_gradients_compute_gpu()
{

  int nop,N[3],Ngradcalc[3],nhalo;



  nhalo = coords_nhalo();
  int nextra=nhalo-1;
  coords_nlocal(N); 
  nop = phi_nop();
  

  /* set up CUDA grid */
  
  Ngradcalc[X]=N[X]+2*nextra;
  Ngradcalc[Y]=N[Y]+2*nextra;
  Ngradcalc[Z]=N[Z]+2*nextra;
  
  int nblocks=(Ngradcalc[X]*Ngradcalc[Y]*Ngradcalc[Z]+DEFAULT_TPB-1)
    /DEFAULT_TPB;
  
  /* run the kernel */
  gradient_3d_7pt_fluid_operator_gpu_d<<<nblocks,DEFAULT_TPB>>>
    (nop, nhalo, N_d, phi_site_d,grad_phi_site_d,delsq_phi_site_d,
     le_index_real_to_buffer_d,nextra); 
  
  cudaThreadSynchronize();
  
  return;
  
  }


/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_operator_gpu_d
 *
 *****************************************************************************/

__global__ void gradient_3d_7pt_fluid_operator_gpu_d(int nop, int nhalo, 
						     int N_d[3], 
						     const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     int * le_index_real_to_buffer_d,
						     int nextra) {
  int n;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  int threadIndex, Nall[3], Ngradcalc[3],ii, jj, kk;

  ys = N_d[Z] + 2*nhalo;

  Nall[X]=N_d[X]+2*nhalo;
  Nall[Y]=N_d[Y]+2*nhalo;
  Nall[Z]=N_d[Z]+2*nhalo;

  int nsites=Nall[X]*Nall[Y]*Nall[Z];

  Ngradcalc[X]=N_d[X]+2*nextra;
  Ngradcalc[Y]=N_d[Y]+2*nextra;
  Ngradcalc[Z]=N_d[Z]+2*nextra;

  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  /* Avoid going beyond problem domain */
  if (threadIndex < Ngradcalc[X]*Ngradcalc[Y]*Ngradcalc[Z])
    {

      /* calculate index from CUDA thread index */

      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Ngradcalc);
      index = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra,
				     kk+nhalo-nextra,Nall);      
      

      /* icm1 = le_index_real_to_buffer(ic, -1); */
      /* icp1 = le_index_real_to_buffer(ic, +1); */
      /*le_index_real_to_buffer_d holds -1 then +1 translation values */
      icm1=le_index_real_to_buffer_d[ii+nhalo-nextra];
      icp1=le_index_real_to_buffer_d[Nall[X]+ii+nhalo-nextra];      

      indexm1 = get_linear_index_gpu_d(icm1,jj+nhalo-nextra,kk+nhalo-nextra,Nall);
      indexp1 = get_linear_index_gpu_d(icp1,jj+nhalo-nextra,kk+nhalo-nextra,Nall);


      for (n = 0; n < nop; n++) { 

	  grad_d[X*nsites*nop+n*nsites+index]
	    = 0.5*(field_d[nsites*n+indexp1] - field_d[nsites*n+indexm1]);
	  grad_d[Y*nsites*nop+n*nsites+index]
	    = 0.5*(field_d[nsites*n+(index + ys)] - field_d[nsites*n+(index - ys)]);
	  grad_d[Z*nsites*nop+n*nsites+index]
	    = 0.5*(field_d[nsites*n+(index + 1)] - field_d[nsites*n+(index - 1)]);
	  del2_d[n*nsites + index]
	    = field_d[nsites*n+indexp1] + field_d[nsites*n+indexm1]
	    + field_d[nsites*n+(index + ys)] + field_d[nsites*n+(index - ys)]
	    + field_d[nsites*n+(index + 1)] + field_d[nsites*n+(index - 1)]
	    - 6.0*field_d[nsites*n+index];
		  } 


   } 
  return;
}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N_d[3])
{
  
  int yfac = N_d[Z];
  int xfac = N_d[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}

/* get 3d coordinates from the index on the accelerator */
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N_d[3])

{
  
  int yfac = N_d[Z];
  int xfac = N_d[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}
