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
#include "propagation.h"
#include "lb_model_s.h"
#include "timer.h"

static int lb_propagate_d2q9(lb_t * lb);
static int lb_propagate_d3q15(lb_t * lb);
static int lb_propagate_d3q19(lb_t * lb);

/*****************************************************************************
 *
 *  lb_propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

__targetHost__ int lb_propagation(lb_t * lb) {

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
					 double* t_fprime, 
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


static int lb_propagate_d3q19(lb_t * lb) {

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
  copyFromTarget(&nDist,&(lb->ndist),sizeof(int)); 


  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstToTarget(&tc_ndist,&nDist, sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(tc_cv,cv, NVEL*3*sizeof(int)); 

  TIMER_start(TIMER_PROP_KERNEL);
  lb_propagate_d3q19_lattice __targetLaunch__(nSites) (lb);
  targetSynchronize();
  TIMER_stop(TIMER_PROP_KERNEL);

  /* swap f and fprime */
  double* tmpptr1;
  double* tmpptr2;
  copyFromTarget(&tmpptr1,&(lb->f),sizeof(double*)); 
  copyFromTarget(&tmpptr2,&(lb->fprime),sizeof(double*)); 
  
  double* tmp=tmpptr2;
  tmpptr2=tmpptr1;
  tmpptr1=tmp;
  
  copyToTarget(&(lb->f),&tmpptr1,sizeof(double*)); 
  copyToTarget(&(lb->fprime),&tmpptr2,sizeof(double*)); 
  /* end swap f and fprime */
  
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
