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
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "coords.h"
#include "propagation.h"
#include "lb_model_s.h"
#include "targetDP.h"
#include "timer.h"

static int lb_propagate_d2q9(lb_t * lb);
static int lb_propagate_d3q15(lb_t * lb);
static int lb_propagate_d3q19(lb_t * lb);
static int lb_propagate_d2q9_r(lb_t * lb);
static int lb_propagate_d3q15_r(lb_t * lb);

/*****************************************************************************
 *
 *  lb_propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

__targetHost__ int lb_propagation(lb_t * lb) {

  assert(lb);

#ifdef LB_DATA_SOA
    if (NVEL == 9)  lb_propagate_d2q9_r(lb);
    if (NVEL == 15) lb_propagate_d3q15_r(lb);
#else
    if (NVEL == 9)  lb_propagate_d2q9(lb);
    if (NVEL == 15) lb_propagate_d3q15(lb);
#endif

  if (NVEL == 19) lb_propagate_d3q19(lb);

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagate_d2q9
 *
 *  Follows the definition of the velocities in d2q9.c
 *
 *****************************************************************************/

static int lb_propagate_d2q9(lb_t * lb) {

  int ic, jc, kc, index, n, p;
  int xstr, ystr, zstr;
  int nhalo;
  int nlocal[3];

  assert(lb);
  assert(NVEL == 9);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  zstr = lb->ndist*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);

  /* Forward moving distributions in memory */
  
  for (ic = nlocal[X]; ic >= 1; ic--) {
    for (jc = nlocal[Y]; jc >= 1; jc--) {

      kc = 1;
      index = coords_index(ic, jc, kc);

      for (n = 0; n < lb->ndist; n++) {
	p = lb->ndist*NVEL*index + n*NVEL;
	lb->f[p + 4] = lb->f[p +             (-1)*ystr + 4];
	lb->f[p + 3] = lb->f[p + (-1)*xstr + (+1)*ystr + 3];
	lb->f[p + 2] = lb->f[p + (-1)*xstr             + 2];
	lb->f[p + 1] = lb->f[p + (-1)*xstr + (-1)*ystr + 1];
      }
    }
  }

  /* Backward moving distributions in memory */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      kc = 1;
      index = coords_index(ic, jc, kc);

      for (n = 0; n < lb->ndist; n++) {
	p = lb->ndist*NVEL*index + n*NVEL;
	lb->f[p + 5] = lb->f[p             + (+1)*ystr + 5];
	lb->f[p + 6] = lb->f[p + (+1)*xstr + (-1)*ystr + 6];
	lb->f[p + 7] = lb->f[p + (+1)*xstr             + 7];
	lb->f[p + 8] = lb->f[p + (+1)*xstr + (+1)*ystr + 8];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagate_d3q15
 *
 *  Follows the definition of the velocities in d3q15.c
 *
 *****************************************************************************/

static int lb_propagate_d3q15(lb_t * lb) {

  int ic, jc, kc, index, n, p;
  int xstr, ystr, zstr;
  int nhalo;
  int nlocal[3];

  assert(lb);
  assert(NVEL == 15);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  zstr = lb->ndist*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);


  /* Forward moving distributions */
  
  for (ic = nlocal[X]; ic >= 1; ic--) {
    for (jc = nlocal[Y]; jc >= 1; jc--) {
      for (kc = nlocal[Z]; kc >= 1; kc--) {

        index = coords_index(ic, jc, kc);

	for (n = 0; n < lb->ndist; n++) {
	  p = lb->ndist*NVEL*index + n*NVEL;
	  lb->f[p + 7] = lb->f[p                         + (-1)*zstr + 7];
	  lb->f[p + 6] = lb->f[p             + (-1)*ystr             + 6];
	  lb->f[p + 5] = lb->f[p + (-1)*xstr + (+1)*ystr + (+1)*zstr + 5];
	  lb->f[p + 4] = lb->f[p + (-1)*xstr + (+1)*ystr + (-1)*zstr + 4];
	  lb->f[p + 3] = lb->f[p + (-1)*xstr                         + 3];
	  lb->f[p + 2] = lb->f[p + (-1)*xstr + (-1)*ystr + (+1)*zstr + 2];
	  lb->f[p + 1] = lb->f[p + (-1)*xstr + (-1)*ystr + (-1)*zstr + 1];
	}
      }
    }
  }

  /* Backward moving distributions */
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

	for (n = 0; n < lb->ndist; n++) {
	  p = lb->ndist*NVEL*index + n*NVEL;
	  lb->f[p +  8] = lb->f[p                         + (+1)*zstr +  8];
	  lb->f[p +  9] = lb->f[p             + (+1)*ystr             +  9];
	  lb->f[p + 10] = lb->f[p + (+1)*xstr + (-1)*ystr + (-1)*zstr + 10];
	  lb->f[p + 11] = lb->f[p + (+1)*xstr + (-1)*ystr + (+1)*zstr + 11];
	  lb->f[p + 12] = lb->f[p + (+1)*xstr                         + 12];
	  lb->f[p + 13] = lb->f[p + (+1)*xstr + (+1)*ystr + (-1)*zstr + 13];
	  lb->f[p + 14] = lb->f[p + (+1)*xstr + (+1)*ystr + (+1)*zstr + 14];
	}
      }
    }
  }

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
  
  int n,p;
  int vecIndex=0;
  int shiftIndex;
  
  char halo_site=0;
  __targetILP__(vecIndex){
    targetCoords3D(coords,tc_Nall,baseIndex+vecIndex);
    if (  coords[0] < tc_nhalo ||
	  coords[1] < tc_nhalo || 
	  coords[2] < tc_nhalo ||
	  coords[0] >= tc_Nall[X]-tc_nhalo || 
	  coords[1] >= tc_Nall[Y]-tc_nhalo ||  
	  coords[2] >= tc_Nall[Z]-tc_nhalo) halo_site=1;
  }
  
  // if not a halo site:
  if (  !halo_site){ 
    
    for (n = 0; n < tc_ndist; n++) {
      
      
      for (p=0;p<NVEL;p++){
	
	
	__targetILP__(vecIndex){
	  targetCoords3D(coords,tc_Nall,baseIndex+vecIndex);
	  shiftIndex=targetIndex3D(coords[0]-tc_cv[p][0],coords[1]-tc_cv[p][1],coords[2]-tc_cv[p][2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL,baseIndex+vecIndex , n, p)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, p)];
	}
      }
      
      
      
    }
    
    
    
  }
  
  
  else { //mopping up chunks that include halo sites
    
    
    __targetILP__(vecIndex){
      
      halo_site=0;
      
      targetCoords3D(coords,tc_Nall,baseIndex+vecIndex);
      if (  coords[0] < tc_nhalo ||
	    coords[1] < tc_nhalo || 
	    coords[2] < tc_nhalo ||
	    coords[0] >= tc_Nall[X]-tc_nhalo || 
	    coords[1] >= tc_Nall[Y]-tc_nhalo ||  
	    coords[2] >= tc_Nall[Z]-tc_nhalo) halo_site=1;
      
      
      if(!halo_site){
	
	for (n = 0; n < tc_ndist; n++) {
	  
	  
	  for (p=0;p<NVEL;p++){
	    
	    
	    targetCoords3D(coords,tc_Nall,baseIndex+vecIndex);
	    shiftIndex=targetIndex3D(coords[0]-tc_cv[p][0],coords[1]-tc_cv[p][1],coords[2]-tc_cv[p][2],tc_Nall);
	    t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex+vecIndex, n, p)] 
	      = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, p)];
	    
	  }
	  
	}
      }
      else if (  coords[0] < tc_Nall[X] &&
		 coords[1] < tc_Nall[Y] && 
		 coords[2] < tc_Nall[Z]) 
	{ //direct copy of t_f to t_fprime for halo sites 
	  
	  for (n = 0; n < tc_ndist; n++) {
	    
	    int ip;
	    for (ip=0;ip<NVEL;ip++){
	      
	      targetCoords3D(coords,tc_Nall,baseIndex+vecIndex);
	      t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex+vecIndex, n, ip)] 
		= t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex+vecIndex, n, ip)]; 
	      
	      
	    }
	    
	    
	    
	  }
	  
	  
	}
    }
  }
  
  
  return;
  
  
}

__targetEntry__  void lb_propagate_d3q19_lattice(lb_t* t_lb)
{

  int baseIndex=0;

  //partition binary collision kernel across the lattice on the target
  __targetTLP__(baseIndex,tc_nSites){
    lb_propagate_d3q19_site (t_lb->f,t_lb->fprime,baseIndex);

  }


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

  int nFields=NVEL*nDist;


  //start constant setup
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstToTarget(&tc_ndist,&nDist, sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(tc_cv,cv, NVEL*3*sizeof(int)); 
  //end constant setup

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
 *  lb_propagate_d3q19_r
 *
 *  Reverse storage implementation.
 *
 *****************************************************************************/

static int lb_propagate_d2q9_r(lb_t * lb) {

  int ic, jc, kc, index, n, p, q;
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
	q = n*lb->nsite + index;

	lb->f[4*p + q] = lb->f[4*p + q +             (-1)*ystr];
	lb->f[3*p + q] = lb->f[3*p + q + (-1)*xstr + (+1)*ystr];
	lb->f[2*p + q] = lb->f[2*p + q + (-1)*xstr            ];
	lb->f[1*p + q] = lb->f[1*p + q + (-1)*xstr + (-1)*ystr];
      }
    }
  
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

	index = coords_index(ic, jc, kc);
	q = n*lb->nsite + index;

	lb->f[5*p + q] = lb->f[5*p + q             + (+1)*ystr];
	lb->f[6*p + q] = lb->f[6*p + q + (+1)*xstr + (-1)*ystr];
	lb->f[7*p + q] = lb->f[7*p + q + (+1)*xstr            ];
	lb->f[8*p + q] = lb->f[8*p + q + (+1)*xstr + (+1)*ystr];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  lb_propagate_d3q15_r
 *
 *  Reverse memeory order implementation
 *
 *****************************************************************************/

static int lb_propagate_d3q15_r(lb_t * lb) {

  int ic, jc, kc, index, n, p, q;
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
	  q = n*lb->nsite + index;

	  lb->f[7*p + q] = lb->f[7*p + q                         + (-1)*zstr];
	  lb->f[6*p + q] = lb->f[6*p + q             + (-1)*ystr            ];
	  lb->f[5*p + q] = lb->f[5*p + q + (-1)*xstr + (+1)*ystr + (+1)*zstr];
	  lb->f[4*p + q] = lb->f[4*p + q + (-1)*xstr + (+1)*ystr + (-1)*zstr];
	  lb->f[3*p + q] = lb->f[3*p + q + (-1)*xstr                        ];
	  lb->f[2*p + q] = lb->f[2*p + q + (-1)*xstr + (-1)*ystr + (+1)*zstr];
	  lb->f[1*p + q] = lb->f[1*p + q + (-1)*xstr + (-1)*ystr + (-1)*zstr];

	}
      }
    }

    /* Distributions mvoing backward in memory. */
  
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(ic, jc, kc);
	  q = n*lb->nsite + index;

	  lb->f[ 8*p + q] = lb->f[ 8*p+q                         + (+1)*zstr];
	  lb->f[ 9*p + q] = lb->f[ 9*p+q             + (+1)*ystr            ];
	  lb->f[10*p + q] = lb->f[10*p+q + (+1)*xstr + (-1)*ystr + (-1)*zstr];
	  lb->f[11*p + q] = lb->f[11*p+q + (+1)*xstr + (-1)*ystr + (+1)*zstr];
	  lb->f[12*p + q] = lb->f[12*p+q + (+1)*xstr                        ];
	  lb->f[13*p + q] = lb->f[13*p+q + (+1)*xstr + (+1)*ystr + (-1)*zstr];
	  lb->f[14*p + q] = lb->f[14*p+q + (+1)*xstr + (+1)*ystr + (+1)*zstr];
	}
      }
    }
  }

  return 0;
}
