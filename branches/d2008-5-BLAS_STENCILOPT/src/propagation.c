/*****************************************************************************
 *
 *  propagation.c
 *
 *  Propagation schemes for the different models.
 *
 *  $Id: propagation.c,v 1.4.12.1 2009-08-14 07:47:43 cevi_parker Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "model.h"

extern Site * site;
extern Site * site_pdt;

#ifdef _D3Q19_
static void d3q19_propagate_single(void);
static void d3q19_propagate_binary(void);
static void d3q19_propagate_single_halo(void);
static void d3q19_propagate_binary_halo(void);
#endif

#ifdef _D3Q15_
static void d3q15_propagate_single(void);
static void d3q15_propagate_binary(void);
#endif


/*****************************************************************************
 *
 *  propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

void propagation() {

#ifdef _D3Q19_

  TIMER_start(TIMER_PROPAGATE);

#ifdef _SINGLE_FLUID_

#ifdef _FUSED_
  d3q19_propagate_single_halo();
#else
  d3q19_propagate_single();
#endif

#else

#ifdef _FUSED_
  d3q19_propagate_binary_halo();
#else
  d3q19_propagate_binary();
#endif

#endif

  TIMER_stop(TIMER_PROPAGATE);

#endif

#ifdef _D3Q15_

  TIMER_start(TIMER_PROPAGATE);

#ifdef _SINGLE_FLUID_
  d3q15_propagate_single();
#else
  d3q15_propagate_binary();
#endif

  TIMER_stop(TIMER_PROPAGATE);

#endif

  return;
}

#ifdef _D3Q15_

/*****************************************************************************
 *
 *  d3q15_propagate_single
 *
 *****************************************************************************/

static void d3q15_propagate_single() {

  int i, j, k, ijk;
  int xfac, yfac;
  int N[3];

  get_N_local(N);

  yfac = (N[Z]+2*nhalo_);
  xfac = (N[Y]+2*nhalo_)*yfac;

  /* Forward moving distributions */
  
  for (i = N[X]; i >= 1; i--) {
    for (j = N[Y]; j >= 1; j--) {
      for (k = N[Z]; k >= 1; k--) {

        ijk = get_site_index(i, j, k);

        site[ijk].f[7] = site[ijk                         + (-1)].f[7];
        site[ijk].f[6] = site[ijk             + (-1)*yfac       ].f[6];
        site[ijk].f[5] = site[ijk + (-1)*xfac + (+1)*yfac + (+1)].f[5];
        site[ijk].f[4] = site[ijk + (-1)*xfac + (+1)*yfac + (-1)].f[4];
        site[ijk].f[3] = site[ijk + (-1)*xfac                   ].f[3];
        site[ijk].f[2] = site[ijk + (-1)*xfac + (-1)*yfac + (+1)].f[2];
        site[ijk].f[1] = site[ijk + (-1)*xfac + (-1)*yfac + (-1)].f[1];

      }
    }
  }

  /* Backward moving distributions */
  
  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

        ijk = get_site_index(i, j, k);
   
        site[ijk].f[ 8] = site[ijk                         + (+1)].f[ 8];
        site[ijk].f[ 9] = site[ijk             + (+1)*yfac       ].f[ 9];
        site[ijk].f[10] = site[ijk + (+1)*xfac + (-1)*yfac + (-1)].f[10];
        site[ijk].f[11] = site[ijk + (+1)*xfac + (-1)*yfac + (+1)].f[11];
        site[ijk].f[12] = site[ijk + (+1)*xfac                   ].f[12];
        site[ijk].f[13] = site[ijk + (+1)*xfac + (+1)*yfac + (-1)].f[13];
        site[ijk].f[14] = site[ijk + (+1)*xfac + (+1)*yfac + (+1)].f[14];

      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  d3q15_propagate_binary
 *
 *  This is the binary fluid version.
 *
 *****************************************************************************/

static void d3q15_propagate_binary() {

  int i, j, k, ijk;
  int xfac, yfac;
  int N[3];

  get_N_local(N);

  yfac = (N[Z]+2*nhalo_);
  xfac = (N[Y]+2*nhalo_)*yfac;

  /* Forward moving distributions */
  
  for (i = N[X]; i >= 1; i--) {
    for (j = N[Y]; j >= 1; j--) {
      for (k = N[Z]; k >= 1; k--) {

        ijk = get_site_index(i, j, k);

        site[ijk].f[7] = site[ijk                         + (-1)].f[7];
        site[ijk].g[7] = site[ijk                         + (-1)].g[7];

        site[ijk].f[6] = site[ijk             + (-1)*yfac       ].f[6];
        site[ijk].g[6] = site[ijk             + (-1)*yfac       ].g[6];

        site[ijk].f[5] = site[ijk + (-1)*xfac + (+1)*yfac + (+1)].f[5];
        site[ijk].g[5] = site[ijk + (-1)*xfac + (+1)*yfac + (+1)].g[5];

        site[ijk].f[4] = site[ijk + (-1)*xfac + (+1)*yfac + (-1)].f[4];
        site[ijk].g[4] = site[ijk + (-1)*xfac + (+1)*yfac + (-1)].g[4];

        site[ijk].f[3] = site[ijk + (-1)*xfac                   ].f[3];
        site[ijk].g[3] = site[ijk + (-1)*xfac                   ].g[3];

        site[ijk].f[2] = site[ijk + (-1)*xfac + (-1)*yfac + (+1)].f[2];
        site[ijk].g[2] = site[ijk + (-1)*xfac + (-1)*yfac + (+1)].g[2];

        site[ijk].f[1] = site[ijk + (-1)*xfac + (-1)*yfac + (-1)].f[1];
        site[ijk].g[1] = site[ijk + (-1)*xfac + (-1)*yfac + (-1)].g[1];

      }
    }
  }

  /* Backward moving distributions */
  
  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

        ijk = get_site_index(i, j, k);
   
        site[ijk].f[ 8] = site[ijk                         + (+1)].f[ 8];
        site[ijk].g[ 8] = site[ijk                         + (+1)].g[ 8];

        site[ijk].f[ 9] = site[ijk             + (+1)*yfac       ].f[ 9];
        site[ijk].g[ 9] = site[ijk             + (+1)*yfac       ].g[ 9];

        site[ijk].f[10] = site[ijk + (+1)*xfac + (-1)*yfac + (-1)].f[10];
        site[ijk].g[10] = site[ijk + (+1)*xfac + (-1)*yfac + (-1)].g[10];

        site[ijk].f[11] = site[ijk + (+1)*xfac + (-1)*yfac + (+1)].f[11];
        site[ijk].g[11] = site[ijk + (+1)*xfac + (-1)*yfac + (+1)].g[11];

        site[ijk].f[12] = site[ijk + (+1)*xfac                   ].f[12];
        site[ijk].g[12] = site[ijk + (+1)*xfac                   ].g[12];

        site[ijk].f[13] = site[ijk + (+1)*xfac + (+1)*yfac + (-1)].f[13];
        site[ijk].g[13] = site[ijk + (+1)*xfac + (+1)*yfac + (-1)].g[13];

        site[ijk].f[14] = site[ijk + (+1)*xfac + (+1)*yfac + (+1)].f[14];
        site[ijk].g[14] = site[ijk + (+1)*xfac + (+1)*yfac + (+1)].g[14];

      }
    }
  }

  return;
}

#endif /* _D3Q15_ */

#ifdef _D3Q19_

/*****************************************************************************
 *
 *  d3q19_propagate_single
 *
 *****************************************************************************/

static void d3q19_propagate_single() {

  int i, j, k, ijk;
  int xfac, yfac;
  int N[3];

  get_N_local(N);

  yfac = (N[Z]+2*nhalo_);
  xfac = (N[Y]+2*nhalo_)*yfac;

  /* Forward moving distributions */
  
  for (i = N[X]; i >= 1; i--) {
    for (j = N[Y]; j >= 1; j--) {
      for (k = N[Z]; k >= 1; k--) {

	ijk = get_site_index(i, j, k);

	site[ijk].f[9] = site[ijk                         + (-1)].f[9];
	site[ijk].f[8] = site[ijk             + (-1)*yfac + (+1)].f[8];
	site[ijk].f[7] = site[ijk             + (-1)*yfac       ].f[7];
	site[ijk].f[6] = site[ijk             + (-1)*yfac + (-1)].f[6];
	site[ijk].f[5] = site[ijk + (-1)*xfac + (+1)*yfac       ].f[5];
	site[ijk].f[4] = site[ijk + (-1)*xfac             + (+1)].f[4];
	site[ijk].f[3] = site[ijk + (-1)*xfac                   ].f[3];
	site[ijk].f[2] = site[ijk + (-1)*xfac             + (-1)].f[2];
	site[ijk].f[1] = site[ijk + (-1)*xfac + (-1)*yfac       ].f[1];

      }
    }
  }

  /* Backward moving distributions */
  
  for (i = 1; i <= N[X]; i++) {
    for (j = 1; j <= N[Y]; j++) {
      for (k = 1; k <= N[Z]; k++) {

	ijk = get_site_index(i, j, k);

	site[ijk].f[10] = site[ijk                         + (+1)].f[10];
	site[ijk].f[11] = site[ijk             + (+1)*yfac + (-1)].f[11];
	site[ijk].f[12] = site[ijk             + (+1)*yfac       ].f[12];
	site[ijk].f[13] = site[ijk             + (+1)*yfac + (+1)].f[13];
	site[ijk].f[14] = site[ijk + (+1)*xfac + (-1)*yfac       ].f[14];
	site[ijk].f[15] = site[ijk + (+1)*xfac             + (-1)].f[15];
	site[ijk].f[16] = site[ijk + (+1)*xfac                   ].f[16];
	site[ijk].f[17] = site[ijk + (+1)*xfac             + (+1)].f[17];
	site[ijk].f[18] = site[ijk + (+1)*xfac + (+1)*yfac       ].f[18];

      }
    }
  }

  return;
}


/*****************************************************************************
 *
 *  d3q19_propagate_binary_halo  - updates the surface of the local 3d lattice from site(t+dt) halos
 *
 *****************************************************************************/

static void d3q19_propagate_binary_halo() {

  int ll, wrap1, wrap2, ijk;
  int xfac, yfac;
  int N[3];
  int SP[3];
  get_N_local(N);

  yfac = (N[Z]+2*nhalo_);
  xfac = (N[Y]+2*nhalo_)*yfac;

  // update surface planes from halos
  for (ll = 0; ll < 3; ll++){
      wrap1 = (ll+4)%3;
      wrap2 = (ll+5)%3;
      for(SP[ll] = nhalo_ ; SP[ll] <= N[ll] ; SP[ll]+=N[ll]-nhalo_){ /* consider each dimensions plane */
	  for (SP[wrap1] = 1; SP[wrap1] <= N[wrap1]; SP[wrap1]++) {
	      for (SP[wrap2] = 1; SP[wrap2] <= N[wrap2]; SP[wrap2]++) {
		  
		  ijk = get_site_index(SP[X], SP[Y], SP[Z]);
		  
		  site_pdt[ijk].f[1] = site[ijk + (-1)*xfac + (-1)*yfac       ].f[1];
		  site_pdt[ijk].f[2] = site[ijk + (-1)*xfac             + (-1)].f[2];
		  site_pdt[ijk].f[3] = site[ijk + (-1)*xfac                   ].f[3];	
		  site_pdt[ijk].f[4] = site[ijk + (-1)*xfac             + (+1)].f[4];	
		  site_pdt[ijk].f[5] = site[ijk + (-1)*xfac + (+1)*yfac       ].f[5];
		  site_pdt[ijk].f[6] = site[ijk             + (-1)*yfac + (-1)].f[6];
		  site_pdt[ijk].f[7] = site[ijk             + (-1)*yfac       ].f[7];
		  site_pdt[ijk].f[8] = site[ijk             + (-1)*yfac + (+1)].f[8];
		  site_pdt[ijk].f[9] = site[ijk                         + (-1)].f[9];	  
		  site_pdt[ijk].f[10] = site[ijk                         + (+1)].f[10];
		  site_pdt[ijk].f[11] = site[ijk             + (+1)*yfac + (-1)].f[11];	  
		  site_pdt[ijk].f[12] = site[ijk             + (+1)*yfac       ].f[12];	  
		  site_pdt[ijk].f[13] = site[ijk             + (+1)*yfac + (+1)].f[13];
		  site_pdt[ijk].f[14] = site[ijk + (+1)*xfac + (-1)*yfac       ].f[14];	  
		  site_pdt[ijk].f[15] = site[ijk + (+1)*xfac             + (-1)].f[15];	  
		  site_pdt[ijk].f[16] = site[ijk + (+1)*xfac                   ].f[16];	  
		  site_pdt[ijk].f[17] = site[ijk + (+1)*xfac             + (+1)].f[17];	  
		  site_pdt[ijk].f[18] = site[ijk + (+1)*xfac + (+1)*yfac       ].f[18];
		  
		  site_pdt[ijk].g[1] = site[ijk + (-1)*xfac + (-1)*yfac       ].g[1];
		  site_pdt[ijk].g[2] = site[ijk + (-1)*xfac             + (-1)].g[2];
		  site_pdt[ijk].g[3] = site[ijk + (-1)*xfac                   ].g[3];	
		  site_pdt[ijk].g[4] = site[ijk + (-1)*xfac             + (+1)].g[4];	
		  site_pdt[ijk].g[5] = site[ijk + (-1)*xfac + (+1)*yfac       ].g[5];
		  site_pdt[ijk].g[6] = site[ijk             + (-1)*yfac + (-1)].g[6];
		  site_pdt[ijk].g[7] = site[ijk             + (-1)*yfac       ].g[7];
		  site_pdt[ijk].g[8] = site[ijk             + (-1)*yfac + (+1)].g[8];
		  site_pdt[ijk].g[9] = site[ijk                         + (-1)].g[9];	  
		  site_pdt[ijk].g[10] = site[ijk                         + (+1)].g[10];
		  site_pdt[ijk].g[11] = site[ijk             + (+1)*yfac + (-1)].g[11];	  
		  site_pdt[ijk].g[12] = site[ijk             + (+1)*yfac       ].g[12];	  
		  site_pdt[ijk].g[13] = site[ijk             + (+1)*yfac + (+1)].g[13];
		  site_pdt[ijk].g[14] = site[ijk + (+1)*xfac + (-1)*yfac       ].g[14];	  
		  site_pdt[ijk].g[15] = site[ijk + (+1)*xfac             + (-1)].g[15];	  
		  site_pdt[ijk].g[16] = site[ijk + (+1)*xfac                   ].g[16];	  
		  site_pdt[ijk].g[17] = site[ijk + (+1)*xfac             + (+1)].g[17];	  
		  site_pdt[ijk].g[18] = site[ijk + (+1)*xfac + (+1)*yfac       ].g[18];
		  
	      }
	  }
      }
  }
  return;
}

/*****************************************************************************
 *
 *  d3q19_propagate_single_halo  - updates the surface of the local 3d lattice from site(t+dt) halos
 *
 *****************************************************************************/

static void d3q19_propagate_single_halo() {

  int ll, wrap1, wrap2, ijk;
  int xfac, yfac;
  int N[3];
  int SP[3];
  get_N_local(N);

  yfac = (N[Z]+2*nhalo_);
  xfac = (N[Y]+2*nhalo_)*yfac;

  // update surface planes from halos
  for (ll = 0; ll < 3; ll++){
      wrap1 = (ll+4)%3;
      wrap2 = (ll+5)%3;
      for( SP[ll] = nhalo_ ; SP[ll] < N[ll] ; SP[ll]+=N[ll]-nhalo_){ /* consider each dimensions plane */
	  for (SP[wrap1] = 1; SP[wrap1] <= N[wrap1]; SP[wrap1]++) {
	      for (SP[wrap2] = 1; SP[wrap2] <= N[wrap2]; SP[wrap2]++) {
		  
		  ijk = get_site_index(SP[X], SP[Y], SP[Z]);
		  
		  site_pdt[ijk].f[1] = site[ijk + (-1)*xfac + (-1)*yfac       ].f[1];
		  site_pdt[ijk].f[2] = site[ijk + (-1)*xfac             + (-1)].f[2];
		  site_pdt[ijk].f[3] = site[ijk + (-1)*xfac                   ].f[3];	
		  site_pdt[ijk].f[4] = site[ijk + (-1)*xfac             + (+1)].f[4];	
		  site_pdt[ijk].f[5] = site[ijk + (-1)*xfac + (+1)*yfac       ].f[5];
		  site_pdt[ijk].f[6] = site[ijk             + (-1)*yfac + (-1)].f[6];
		  site_pdt[ijk].f[7] = site[ijk             + (-1)*yfac       ].f[7];
		  site_pdt[ijk].f[8] = site[ijk             + (-1)*yfac + (+1)].f[8];
		  site_pdt[ijk].f[9] = site[ijk                         + (-1)].f[9];	  
		  site_pdt[ijk].f[10] = site[ijk                         + (+1)].f[10];
		  site_pdt[ijk].f[11] = site[ijk             + (+1)*yfac + (-1)].f[11];	  
		  site_pdt[ijk].f[12] = site[ijk             + (+1)*yfac       ].f[12];	  
		  site_pdt[ijk].f[13] = site[ijk             + (+1)*yfac + (+1)].f[13];
		  site_pdt[ijk].f[14] = site[ijk + (+1)*xfac + (-1)*yfac       ].f[14];	  
		  site_pdt[ijk].f[15] = site[ijk + (+1)*xfac             + (-1)].f[15];	  
		  site_pdt[ijk].f[16] = site[ijk + (+1)*xfac                   ].f[16];	  
		  site_pdt[ijk].f[17] = site[ijk + (+1)*xfac             + (+1)].f[17];	  
		  site_pdt[ijk].f[18] = site[ijk + (+1)*xfac + (+1)*yfac       ].f[18];
		  
	      }
	  }
      }
  }
  return;
}
 
/*****************************************************************************
 *
 *  d3q19_propagate_binary
 *
 *****************************************************************************/


static void d3q19_propagate_binary() {

  int i, j, k, ijk;
  int xfac, yfac;
  int N[3];

  get_N_local(N);

  yfac = (N[Z]+2*nhalo_);
  xfac = (N[Y]+2*nhalo_)*yfac;

  /* Forward moving distributions */
  
  for (i = N[X]; i >= 1; i--) {
    for (j = N[Y]; j >= 1; j--) {
      for (k = N[Z]; k >= 1; k--) {

	ijk = get_site_index(i, j, k);

	site[ijk].f[9] = site[ijk                         + (-1)].f[9];
	site[ijk].g[9] = site[ijk                         + (-1)].g[9];

	site[ijk].f[8] = site[ijk             + (-1)*yfac + (+1)].f[8];
	site[ijk].g[8] = site[ijk             + (-1)*yfac + (+1)].g[8];

	site[ijk].f[7] = site[ijk             + (-1)*yfac       ].f[7];
	site[ijk].g[7] = site[ijk             + (-1)*yfac       ].g[7];

	site[ijk].f[6] = site[ijk             + (-1)*yfac + (-1)].f[6];
	site[ijk].g[6] = site[ijk             + (-1)*yfac + (-1)].g[6];

	site[ijk].f[5] = site[ijk + (-1)*xfac + (+1)*yfac       ].f[5];
	site[ijk].g[5] = site[ijk + (-1)*xfac + (+1)*yfac       ].g[5];

	site[ijk].f[4] = site[ijk + (-1)*xfac             + (+1)].f[4];
	site[ijk].g[4] = site[ijk + (-1)*xfac             + (+1)].g[4];

	site[ijk].f[3] = site[ijk + (-1)*xfac                   ].f[3];
	site[ijk].g[3] = site[ijk + (-1)*xfac                   ].g[3];

	site[ijk].f[2] = site[ijk + (-1)*xfac             + (-1)].f[2];
	site[ijk].g[2] = site[ijk + (-1)*xfac             + (-1)].g[2];

	site[ijk].f[1] = site[ijk + (-1)*xfac + (-1)*yfac       ].f[1];
	site[ijk].g[1] = site[ijk + (-1)*xfac + (-1)*yfac       ].g[1];

      }
    }
  }    
    
  
  /* Backward moving distributions */
  
   for (i = 1; i <= N[X]; i++) {
       for (j = 1; j <= N[Y]; j++) {
	   for (k = 1; k <= N[Z]; k++) {
	
	ijk = get_site_index(i, j, k);

	site[ijk].f[10] = site[ijk                         + (+1)].f[10];
	site[ijk].g[10] = site[ijk                         + (+1)].g[10];

	site[ijk].f[11] = site[ijk             + (+1)*yfac + (-1)].f[11];
	site[ijk].g[11] = site[ijk             + (+1)*yfac + (-1)].g[11];

	site[ijk].f[12] = site[ijk             + (+1)*yfac       ].f[12];
	site[ijk].g[12] = site[ijk             + (+1)*yfac       ].g[12];

	site[ijk].f[13] = site[ijk             + (+1)*yfac + (+1)].f[13];
	site[ijk].g[13] = site[ijk             + (+1)*yfac + (+1)].g[13];

	site[ijk].f[14] = site[ijk + (+1)*xfac + (-1)*yfac       ].f[14];
	site[ijk].g[14] = site[ijk + (+1)*xfac + (-1)*yfac       ].g[14];

	site[ijk].f[15] = site[ijk + (+1)*xfac             + (-1)].f[15];
	site[ijk].g[15] = site[ijk + (+1)*xfac             + (-1)].g[15];

	site[ijk].f[16] = site[ijk + (+1)*xfac                   ].f[16];
	site[ijk].g[16] = site[ijk + (+1)*xfac                   ].g[16];

	site[ijk].f[17] = site[ijk + (+1)*xfac             + (+1)].f[17];
	site[ijk].g[17] = site[ijk + (+1)*xfac             + (+1)].g[17];

	site[ijk].f[18] = site[ijk + (+1)*xfac + (+1)*yfac       ].f[18];
	site[ijk].g[18] = site[ijk + (+1)*xfac + (+1)*yfac       ].g[18];

      }
    }
  }

  return;
}



#endif /* _D3Q19_ */
