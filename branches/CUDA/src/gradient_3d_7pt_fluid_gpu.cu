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

  static dim3 BlockDims;
  static dim3 GridDims;


  nhalo = coords_nhalo();
  int nextra=nhalo-1;
  coords_nlocal(N); 
  nop = phi_nop();
  

  /* set up CUDA grid */
  
  Ngradcalc[X]=N[X]+2*nextra;
  Ngradcalc[Y]=N[Y]+2*nextra;
  Ngradcalc[Z]=N[Z]+2*nextra;
  
#define BLOCKSIZE 256
  /* 1D decomposition - use x grid and block dimension only */ 
  BlockDims.x=BLOCKSIZE;
  GridDims.x=(Ngradcalc[X]*Ngradcalc[Y]*Ngradcalc[Z]+BlockDims.x-1)
    /BlockDims.x;
  
  /* run the kernel */
  gradient_3d_7pt_fluid_operator_gpu_d<<<GridDims.x,BlockDims.x>>>
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
      icm1=le_index_real_to_buffer_d[ii+1];
      icp1=le_index_real_to_buffer_d[Nall[X]+ii+1];      
      
      indexm1 = get_linear_index_gpu_d(icm1,jj+nhalo,kk+nhalo,Nall);
      indexp1 = get_linear_index_gpu_d(icp1,jj+nhalo,kk+nhalo,Nall);


      for (n = 0; n < nop; n++) { 
	  grad_d[3*(nop*index + n) + X]
	    = 0.5*(field_d[nop*indexp1 + n] - field_d[nop*indexm1 + n]);
	  grad_d[3*(nop*index + n) + Y]
	    = 0.5*(field_d[nop*(index + ys) + n] - field_d[nop*(index - ys) + n]);
	  grad_d[3*(nop*index + n) + Z]
	    = 0.5*(field_d[nop*(index + 1) + n] - field_d[nop*(index - 1) + n]);
	  del2_d[nop*index + n]
	    = field_d[nop*indexp1      + n] + field_d[nop*indexm1      + n]
	    + field_d[nop*(index + ys) + n] + field_d[nop*(index - ys) + n]
	    + field_d[nop*(index + 1)  + n] + field_d[nop*(index - 1)  + n]
	    - 6.0*field_d[nop*index + n];
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
