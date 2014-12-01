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

HOST int lb_propagation(lb_t * lb) {

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
    if (NVEL == 19) lb_propagate_d3q19_r(lb);
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
extern TARGET_CONST int tc_nSites; //declared in collision.c
extern TARGET_CONST int tc_Nall[3]; //declared in gradient routine

TARGET_CONST int tc_ndist;
TARGET_CONST int tc_nhalo;


/*****************************************************************************
 *
 *  lb_propagate_d3q19
 *
 *  Follows the velocities defined in d3q19.c
 *
 *****************************************************************************/

TARGET  void lb_propagate_d3q19_site(double* t_f, 
				     double* t_fprime, char* t_siteMask,
					    const int baseIndex){
  


  int zstr = tc_ndist*NVEL;
  int ystr = zstr*tc_Nall[Z];
  int xstr = ystr*tc_Nall[Y];

  int n;

  
    if(t_siteMask[baseIndex]){ //only non-halo sites
	for (n = 0; n < tc_ndist; n++) {
	  int p = tc_ndist*NVEL*baseIndex + n*NVEL;

	  t_fprime[p] = t_f[p];
	  t_fprime[p + 1] = t_f[p + (-1)*xstr + (-1)*ystr             + 1];
	  t_fprime[p + 2] = t_f[p + (-1)*xstr             + (-1)*zstr + 2];
	  t_fprime[p + 3] = t_f[p + (-1)*xstr                         + 3];
	  t_fprime[p + 4] = t_f[p + (-1)*xstr             + (+1)*zstr + 4];
	  t_fprime[p + 5] = t_f[p + (-1)*xstr + (+1)*ystr             + 5];
	  t_fprime[p + 6] = t_f[p             + (-1)*ystr + (-1)*zstr + 6];
	  t_fprime[p + 7] = t_f[p             + (-1)*ystr             + 7];
	  t_fprime[p + 8] = t_f[p             + (-1)*ystr + (+1)*zstr + 8];
	  t_fprime[p + 9] = t_f[p                         + (-1)*zstr + 9];
	  t_fprime[p + 10] = t_f[p                         + (+1)*zstr + 10];
	  t_fprime[p + 11] = t_f[p             + (+1)*ystr + (-1)*zstr + 11];
	  t_fprime[p + 12] = t_f[p             + (+1)*ystr             + 12];
	  t_fprime[p + 13] = t_f[p             + (+1)*ystr + (+1)*zstr + 13];
	  t_fprime[p + 14] = t_f[p + (+1)*xstr + (-1)*ystr             + 14];
	  t_fprime[p + 15] = t_f[p + (+1)*xstr             + (-1)*zstr + 15];
	  t_fprime[p + 16] = t_f[p + (+1)*xstr                         + 16];
	  t_fprime[p + 17] = t_f[p + (+1)*xstr             + (+1)*zstr + 17];
	  t_fprime[p + 18] = t_f[p + (+1)*xstr + (+1)*ystr             + 18];
	}
 
    }

    else{ //direct copy of t_fprime to t_f for halo sites 

	for (n = 0; n < tc_ndist; n++) {

	  int ip;
	  for (ip=0;ip<NVEL;ip++){
	  int p = tc_ndist*NVEL*baseIndex + n*NVEL + ip;

	  t_fprime[p] = t_f[p]; 

	  }



	}
      

    }
  



}

TARGET_ENTRY  void lb_propagate_d3q19_lattice(double* t_f, 
					      double* t_fprime, char* t_siteMask){


  int baseIndex=0;

  //partition binary collision kernel across the lattice on the target
  TARGET_TLP(baseIndex,tc_nSites){
    lb_propagate_d3q19_site (t_f,t_fprime,t_siteMask,baseIndex);

  }


}


static int lb_propagate_d3q19(lb_t * lb) {

  int ic, jc, kc, index;
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
  copyConstantIntToTarget(&tc_nSites,&nSites, sizeof(int)); 
  copyConstantIntToTarget(&tc_ndist,&lb->ndist, sizeof(int)); 
  copyConstantIntToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstantInt1DArrayToTarget( (int*) tc_Nall,Nall, 3*sizeof(int)); 
  //end constant setup


  memset(lb->siteMask,0,nSites*sizeof(char));
  // set all non-halo sites to 1
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

  	index=coords_index(ic, jc, kc);
	lb->siteMask[index]=1;

      }
    }
  }



  copyToTarget(lb->t_f,lb->f,nSites*nFields*sizeof(double)); 
  copyToTarget(lb->t_siteMask,lb->siteMask,nSites*sizeof(char)); 

  lb_propagate_d3q19_lattice TARGET_LAUNCH(nSites) (lb->t_f,lb->t_fprime,lb->t_siteMask);

  copyFromTarget(lb->f,lb->t_fprime,nSites*nFields*sizeof(double)); 

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
