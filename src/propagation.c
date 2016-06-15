/*****************************************************************************
 *
 *  propagation.c
 *
 *  Propagation schemes for the different models.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "propagation.h"
#include "lb_model_s.h"
#include "timer.h"

__host__ int lb_propagation_driver(lb_t * lb);

static int lb_propagate_d2q9(lb_t * lb);
static int lb_propagate_d3q15(lb_t * lb);
static int lb_propagate_d3q19(lb_t * lb);
__host__ int lb_model_swapf(lb_t * lb);

__global__ void lb_propagation_kernel(lb_t * lb);
__global__ void lb_propagation_kernel_novector(lb_t * lb);

/*****************************************************************************
 *
 *  lb_propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

__host__ int lb_propagation(lb_t * lb) {

  assert(lb);

  if (NVEL ==  9) lb_propagate_d2q9(lb);
  if (NVEL == 15) lb_propagate_d3q15(lb);
  if (NVEL == 19) lb_propagate_d3q19(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagate_d3q19
 *
 *  Follows the velocities defined in d3q19.c
 *
 *****************************************************************************/

__target__  void lb_propagate_d3q19_site(const double* __restrict__ t_f, 
					 double* __restrict__ t_fprime, 
					 const int baseIndex){
  
  
  

  int coords[3];
  
  int i;
  int n,p;
  int iv=0;
  
  
#if VVL == 1    
  /*restrict operation to the interior lattice sites*/ 
  targetCoords3D(coords,tc_Nall,baseIndex); 
  if (coords[0] >= (tc_nhalo) && 
      coords[1] >= (tc_nhalo) && 
      coords[2] >= (tc_nhalo) &&
      coords[0] < tc_Nall[X]-(tc_nhalo) &&  
      coords[1] < tc_Nall[Y]-(tc_nhalo)  &&  
      coords[2] < tc_Nall[Z]-(tc_nhalo) )
#endif
    
    { 
      
      
      
      /* work out which sites in this chunk should be included */
      int includeSite[VVL];
      __targetILP__(iv) includeSite[iv]=0;
      
      int coordschunk[3][VVL];
      
      __targetILP__(iv){
	for(i=0;i<3;i++){
	  targetCoords3D(coords,tc_Nall,baseIndex+iv);
	  coordschunk[i][iv]=coords[i];
	}
      }
      
      __targetILP__(iv){
	
	if ((coordschunk[0][iv] >= (tc_nhalo) &&
	     coordschunk[1][iv] >= (tc_nhalo) &&
	     coordschunk[2][iv] >= (tc_nhalo) &&
	     coordschunk[0][iv] < tc_Nall[X]-(tc_nhalo) &&
	     coordschunk[1][iv] < tc_Nall[Y]-(tc_nhalo)  &&
	     coordschunk[2][iv] < tc_Nall[Z]-(tc_nhalo)))
	  
	  includeSite[iv]=1;
      }
      
      
      
      int index1[VVL];
      for (n = 0; n < tc_ndist; n++) {
	
	
	for (p=0;p<NVEL;p++){
	  
	  
	  /* get neighbour indices for this chunk */
	  __targetILP__(iv) index1[iv] = 
	    targetIndex3D(coordschunk[0][iv]-tc_cv[p][0],
			  coordschunk[1][iv]-tc_cv[p][1],
			  coordschunk[2][iv]-tc_cv[p][2],tc_Nall);
	  
	  
	  /* perform propagation for to all non-halo sites */
	  __targetILP__(iv){
	    
	    if(includeSite[iv])
	      {
		
		t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL,baseIndex+iv , n, p)] 
		  = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, index1[iv], n, p)];
		
	      }
	    
	  }
	}
	
	
	
      }
      
      
      
    }
  
  
  return;
  
  
}

__targetEntry__  void lb_propagate_d3q19_lattice(lb_t* t_lb) {

  int baseIndex=0;

  __targetTLP__(baseIndex,tc_nSites){
    lb_propagate_d3q19_site (t_lb->f,t_lb->fprime,baseIndex);
  }

  return;
}


static __host__ int lb_propagate_d3q19(lb_t * lb) {

  int nhalo;
  int nlocal[3];

  assert(lb);
  assert(NVEL == 19);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];

  int nDist;
  lb_ndist(lb, &nDist);

  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstToTarget(&tc_ndist,&nDist, sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(tc_cv,cv, NVEL*3*sizeof(int)); 

  TIMER_start(TIMER_PROP_KERNEL);
  lb_propagate_d3q19_lattice __targetLaunch__(nSites) (lb->target);
  targetSynchronize();
  TIMER_stop(TIMER_PROP_KERNEL);

  lb_model_swapf(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagate_d2q9
 *
 *  General implementation.
 *
 *****************************************************************************/

static int lb_propagate_d2q9(lb_t * lb) {

  int ic, jc, kc, index, index1, n, p;
  int xstr, ystr, zstr;
  int nlocal[3];

  assert(lb);
  assert(NVEL == 9);

  coords_nlocal(nlocal);

  /* Stride in memory for velocities, and space */

  p = lb->ndist*lb->nsite;
  coords_strides(&xstr, &ystr, &zstr);
  kc = 1;

  for (n = 0; n < lb->ndist; n++) {

    /* Distributions moving forward in memory. */
  
    for (ic = nlocal[X]; ic >= 1; ic--) {
      for (jc = nlocal[Y]; jc >= 1; jc--) {

	index = coords_index(ic, jc, kc);
	index1 = coords_index(ic, jc - 1, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 4)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 4)];

	index1 = coords_index(ic - 1, jc + 1, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 3)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 3)];

	index1 = coords_index(ic - 1, jc, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 2)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 2)];

	index1 = coords_index(ic - 1, jc - 1, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 1)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 1)];
      }
    }
  
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);
	index1 = coords_index(ic, jc + 1, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 5)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 5)];

	index1 = coords_index(ic + 1, jc - 1, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 6)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 6)];

	index1 = coords_index(ic + 1, jc, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 7)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 7)];

	index1 = coords_index(ic + 1, jc + 1, kc);
	lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 8)] =
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 8)];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagate_d3q15
 *
 *  General implementation
 *
 *****************************************************************************/

static int lb_propagate_d3q15(lb_t * lb) {

  int ic, jc, kc, index, index1, n, p;
  int xstr, ystr, zstr;
  int nlocal[3];

  assert(lb);
  assert(NVEL == 15);

  coords_nlocal(nlocal);

  /* Stride in memory for velocities, and space */

  p = lb->ndist*lb->nsite;
  coords_strides(&xstr, &ystr, &zstr);

  for (n = 0; n < lb->ndist; n++) {

    /* Distributions moving forward in memory. */
  
    for (ic = nlocal[X]; ic >= 1; ic--) {
      for (jc = nlocal[Y]; jc >= 1; jc--) {
	for (kc = nlocal[Z]; kc >= 1; kc--) {

	  index = coords_index(ic, jc, kc);
	  index1 = coords_index(ic, jc, kc - 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 7)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 7)];
	  index1 = coords_index(ic, jc - 1, kc);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 6)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 6)];
	  index1 = coords_index(ic - 1, jc + 1, kc + 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 5)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 5)];
	  index1 = coords_index(ic - 1, jc + 1, kc - 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 4)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 4)];
	  index1 = coords_index(ic - 1, jc, kc);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 3)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 3)];
	  index1 = coords_index(ic - 1, jc - 1, kc + 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 2)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 2)];
	  index1 = coords_index(ic - 1, jc - 1, kc - 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 1)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 1)];
	}
      }
    }

    /* Distributions mvoing backward in memory. */
  
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(ic, jc, kc);

	  index1 = coords_index(ic, jc, kc + 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 8)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 8)];
	  index1 = coords_index(ic, jc + 1, kc);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 9)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 9)];
	  index1 = coords_index(ic + 1, jc - 1, kc - 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 10)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 10)];
	  index1 = coords_index(ic + 1, jc - 1, kc + 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 11)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 11)];
	  index1 = coords_index(ic + 1, jc, kc);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 12)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 12)];
	  index1 = coords_index(ic + 1, jc + 1, kc - 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 13)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 13)];
	  index1 = coords_index(ic + 1, jc + 1, kc + 1);
	  lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, 14)] =
	    lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, index1, n, 14)];
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagation_driver
 *
 *****************************************************************************/

__host__ int lb_propagation_driver(lb_t * lb) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  __host__ int lb_model_swapf(lb_t * lb);

  assert(lb);

  coords_nlocal(nlocal);

  /* The kernel is local domain only */

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  /* Encapsulate. lb_kernel_commit(lb); */
  copyConstToTarget(tc_cv, cv, NVEL*3*sizeof(int)); 

  kernel_ctxt_create(NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  /* NEED TO EDIT CURRENTLY UNTIL ALIASING SORTED IN model.c */
  __host_launch_kernel(lb_propagation_kernel_novector, nblk, ntpb, lb);
  targetDeviceSynchronise();

  kernel_ctxt_free(ctxt);

  lb_model_swapf(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagation_kernel_novector
 *
 *  Non-vectorised version, just for testing.
 *
 *****************************************************************************/

__global__ void lb_propagation_kernel_novector(lb_t * lb) {

  int kindex;
  __shared__ int kiter;

  assert(lb);

  kiter = kernel_iterations();

  __target_simt_parallel_for(kindex, kiter, 1) {

    int n, p;
    int ic, jc, kc;
    int icp, jcp, kcp;
    int index, indexp;

    ic = kernel_coords_ic(kindex);
    jc = kernel_coords_jc(kindex);
    kc = kernel_coords_kc(kindex);
    index = kernel_coords_index(ic, jc, kc);

    for (n = 0; n < lb->ndist; n++) {
      for (p = 0; p < NVEL; p++) {

	/* Pull from neighbour */ 
	icp = ic - tc_cv[p][X];
	jcp = jc - tc_cv[p][Y];
	kcp = kc - tc_cv[p][Z];
	indexp = kernel_coords_index(icp, jcp, kcp);
	lb->fprime[LB_ADDR(lb->nsite, lb->ndist, NVEL, index, n, p)] 
	  = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, indexp, n, p)];
      }
    }
    /* Next site */
  }

  return;
}

/*****************************************************************************
 *
 *  lb_propagation_kernel
 *
 *  Ultimately an optimised version.
 *
 *****************************************************************************/

__global__ void lb_propagation_kernel(lb_t * lb) {

  int kindex;
  __shared__ int kiter;

  assert(lb);

  kiter = kernel_vector_iterations();

  __targetTLP__ (kindex, kiter) {

    int iv, indexp;
    int n, p;
    int icp, jcp, kcp;
    int icv[NSIMDVL];
    int jcv[NSIMDVL];
    int kcv[NSIMDVL];
    int maskv[NSIMDVL];
    int index[NSIMDVL];

    __targetILP__(iv) {

      icv[iv] = kernel_coords_icv(kindex, iv);
      jcv[iv] = kernel_coords_jcv(kindex, iv);
      kcv[iv] = kernel_coords_kcv(kindex, iv);

      index[iv] = kernel_coords_index(icv[iv], jcv[iv], kcv[iv]);
      maskv[iv] = kernel_mask(icv[iv], jcv[iv], kcv[iv]);
    }

    for (n = 0; n < lb->ndist; n++) {
      for (p = 0; p < NVEL; p++) {

	__targetILP__(iv) {
	  /* If this is a halo site, just copy, else pull from neighbour */ 
	  icp = icv[iv] - maskv[iv]*tc_cv[p][X];
	  jcp = jcv[iv] - maskv[iv]*tc_cv[p][Y];
	  kcp = kcv[iv] - maskv[iv]*tc_cv[p][Z];
	  indexp = kernel_coords_index(icp, jcp, kcp);
	  lb->fprime[LB_ADDR(lb->nsite, lb->ndist, NVEL, index[iv], n, p)] 
	    = lb->f[LB_ADDR(lb->nsite, lb->ndist, NVEL, indexp, n, p)];
	}
      }
    }
    /* Next sites */
  }

  return;
}

/*****************************************************************************
 *
 *  lb_model_swapf
 *
 *  Switch the "f" and "fprime" pointers.
 *  Intended for use after the propagation step.
 *
 *****************************************************************************/

__host__ int lb_model_swapf(lb_t * lb) {

  int ndevice;
  double * tmp1;
  double * tmp2;

  assert(lb);
  assert(lb->target);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    tmp1 = lb->f;
    lb->f = lb->fprime;
    lb->fprime = tmp1;
  }
  else {
    copyFromTarget(&tmp1, &lb->target->f, sizeof(double *)); 
    copyFromTarget(&tmp2, &lb->target->fprime, sizeof(double *)); 

    copyToTarget(&lb->target->f, &tmp2, sizeof(double *));
    copyToTarget(&lb->target->fprime, &tmp1, sizeof(double *));
  }

  return 0;
}
