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

#include "pe.h"
#include "coords.h"
#include "propagation.h"
#include "lb_model_s.h"
#include "targetDP.h"
#include <string.h>

static int lb_propagate_d2q9(lb_t * lb);
static int lb_propagate_d3q15(lb_t * lb);
static int lb_propagate_d3q19(lb_t * lb);
static int lb_propagate_d2q9_r(lb_t * lb);
static int lb_propagate_d3q15_r(lb_t * lb);
static int lb_propagate_d3q19_r(lb_t * lb);

/*****************************************************************************
 *
 *  lb_propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

__targetHost__ int lb_propagation(lb_t * lb) {

  assert(lb);

  if (lb_order(lb) == MODEL) {
    if (NVEL == 9)  lb_propagate_d2q9(lb);
    if (NVEL == 15) lb_propagate_d3q15(lb);
    if (NVEL == 19) lb_propagate_d3q19(lb);
  }
  else {
    /* Reverse implementation */
    if (NVEL == 9)  lb_propagate_d2q9_r(lb);
    if (NVEL == 15) lb_propagate_d3q15_r(lb);
    //if (NVEL == 19) lb_propagate_d3q19_r(lb);
    // lb_propagate_d3q19 now uses generic addressing for both MODEL and MODEL_R
    if (NVEL == 19) lb_propagate_d3q19(lb);
  }

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



//TODO declare these somewhere sensible.
extern __targetConst__ int tc_nSites; //declared in collision.c
extern __targetConst__ int tc_Nall[3]; //declared in gradient routine

__targetConst__ int tc_ndist;
extern __targetConst__ int tc_nhalo;


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
  targetCoords3D(coords,tc_Nall,baseIndex);

  int n;

  // if not a halo site:
  if (coords[0] >= tc_nhalo && 
      coords[1] >= tc_nhalo && 
      coords[2] >= tc_nhalo &&
      coords[0] < tc_Nall[X]-tc_nhalo &&  
      coords[1] < tc_Nall[Y]-tc_nhalo &&  
      coords[2] < tc_Nall[Z]-tc_nhalo){ 

	for (n = 0; n < tc_ndist; n++) {

	  
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 0)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 0)];

	  int shiftIndex;

	  shiftIndex=targetIndex3D(coords[0]-1,coords[1]-1,coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 1)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 1)];


	  shiftIndex=targetIndex3D(coords[0]-1,coords[1],coords[2]-1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 2)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 2)];


	  shiftIndex=targetIndex3D(coords[0]-1,coords[1],coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 3)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 3)];


	  shiftIndex=targetIndex3D(coords[0]-1,coords[1],coords[2]+1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 4)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 4)];


	  shiftIndex=targetIndex3D(coords[0]-1,coords[1]+1,coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 5)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 5)];


	  shiftIndex=targetIndex3D(coords[0],coords[1]-1,coords[2]-1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 6)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 6)];


	  shiftIndex=targetIndex3D(coords[0],coords[1]-1,coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 7)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 7)];


	  shiftIndex=targetIndex3D(coords[0],coords[1]-1,coords[2]+1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 8)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 8)];


	  shiftIndex=targetIndex3D(coords[0],coords[1],coords[2]-1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 9)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 9)];


	  shiftIndex=targetIndex3D(coords[0],coords[1],coords[2]+1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 10)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 10)];


	  shiftIndex=targetIndex3D(coords[0],coords[1]+1,coords[2]-1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 11)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 11)];


	  shiftIndex=targetIndex3D(coords[0],coords[1]+1,coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 12)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 12)];


	  shiftIndex=targetIndex3D(coords[0],coords[1]+1,coords[2]+1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 13)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 13)];


	  shiftIndex=targetIndex3D(coords[0]+1,coords[1]-1,coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 14)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 14)];


	  shiftIndex=targetIndex3D(coords[0]+1,coords[1],coords[2]-1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 15)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 15)];


	  shiftIndex=targetIndex3D(coords[0]+1,coords[1],coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 16)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 16)];


	  shiftIndex=targetIndex3D(coords[0]+1,coords[1],coords[2]+1,tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 17)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 17)];


	  shiftIndex=targetIndex3D(coords[0]+1,coords[1]+1,coords[2],tc_Nall);
	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, 18)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, shiftIndex, n, 18)];



	}


 
    }

    else{ //direct copy of t_f to t_fprime for halo sites 

	for (n = 0; n < tc_ndist; n++) {

	  int ip;
	  for (ip=0;ip<NVEL;ip++){

	  t_fprime[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, ip)] 
	    = t_f[LB_ADDR(tc_nSites, tc_ndist, NVEL, baseIndex, n, ip)]; 

	  }



	}
      

    }
  



}

__targetEntry__  void lb_propagate_d3q19_lattice(const double* __restrict__ t_f, 
					      double* t_fprime){


  int baseIndex=0;

  //partition binary collision kernel across the lattice on the target
  __targetTLPNoStride__(baseIndex,tc_nSites){
    lb_propagate_d3q19_site (t_f,t_fprime,baseIndex);

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

  int nFields=NVEL*lb->ndist;


  //start constant setup
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstToTarget(&tc_ndist,&lb->ndist, sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  //end constant setup

#ifdef CUDA //temporary optimisation specific to GPU code for benchmarking
  copyToTargetBoundary3D(lb->t_f,lb->f,Nall,nFields,0,nhalo); 
#else
  copyToTarget(lb->t_f,lb->f,nSites*nFields*sizeof(double)); 
#endif

  lb_propagate_d3q19_lattice __targetLaunchNoStride__(nSites) (lb->t_f,lb->t_fprime);
  targetSynchronize();

#ifdef CUDA //temporary optimisation specific to GPU code for benchmarking
  double* tmp=lb->t_fprime;
  lb->t_fprime=lb->t_f;
  lb->t_f=tmp;
#else
    copyFromTarget(lb->f,lb->t_fprime,nSites*nFields*sizeof(double)); 
#endif

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

/*****************************************************************************
 *
 *  lb_propagate_d3q19_r
 *
 *  MODEL_R implmentation
 *
 *****************************************************************************/

static int lb_propagate_d3q19_r(lb_t * lb) {

  int ic, jc, kc, index, n, p, q;
  int xstr, ystr, zstr;
  int nlocal[3];

  assert(lb);
  assert(NVEL == 19);

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
	  lb->f[9*p + q] = lb->f[9*p + q                         + (-1)*zstr];
	  lb->f[8*p + q] = lb->f[8*p + q             + (-1)*ystr + (+1)*zstr];
	  lb->f[7*p + q] = lb->f[7*p + q             + (-1)*ystr            ];
	  lb->f[6*p + q] = lb->f[6*p + q             + (-1)*ystr + (-1)*zstr];
	  lb->f[5*p + q] = lb->f[5*p + q + (-1)*xstr + (+1)*ystr            ];
	  lb->f[4*p + q] = lb->f[4*p + q + (-1)*xstr             + (+1)*zstr];
	  lb->f[3*p + q] = lb->f[3*p + q + (-1)*xstr                        ];
	  lb->f[2*p + q] = lb->f[2*p + q + (-1)*xstr             + (-1)*zstr];
	  lb->f[1*p + q] = lb->f[1*p + q + (-1)*xstr + (-1)*ystr            ];
	}
      }
    }

    /* Distributions mvoing backward in memory. */
  
    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
	for (kc = 1; kc <= nlocal[Z]; kc++) {

	  index = coords_index(ic, jc, kc);

	  q = n*lb->nsite + index;
	  lb->f[10*p + q] = lb->f[10*p + q                         + (+1)*zstr];
	  lb->f[11*p + q] = lb->f[11*p + q             + (+1)*ystr + (-1)*zstr];
	  lb->f[12*p + q] = lb->f[12*p + q             + (+1)*ystr            ];
	  lb->f[13*p + q] = lb->f[13*p + q             + (+1)*ystr + (+1)*zstr];
	  lb->f[14*p + q] = lb->f[14*p + q + (+1)*xstr + (-1)*ystr            ];
	  lb->f[15*p + q] = lb->f[15*p + q + (+1)*xstr             + (-1)*zstr];
	  lb->f[16*p + q] = lb->f[16*p + q + (+1)*xstr                        ];
	  lb->f[17*p + q] = lb->f[17*p + q + (+1)*xstr             + (+1)*zstr];
	  lb->f[18*p + q] = lb->f[18*p + q + (+1)*xstr + (+1)*ystr            ];
	}
      }
    }
  }

  return 0;
}
