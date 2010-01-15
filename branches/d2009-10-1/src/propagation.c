/*****************************************************************************
 *
 *  propagation.c
 *
 *  Propagation schemes for the different models.
 *
 *  $Id: propagation.c,v 1.4.16.1 2010-01-15 16:49:58 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "model.h"

#ifndef DIST_NEW

extern Site * site;

#ifdef _D3Q19_
static void d3q19_propagate_single(void);
static void d3q19_propagate_binary(void);
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
  d3q19_propagate_single();
#else
  d3q19_propagate_binary();
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

#else


static void propagate_d3q15(void);
static void propagate_d3q19(void);

/*****************************************************************************
 *
 *  propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

void propagation() {

  TIMER_start(TIMER_PROPAGATE);

  if (NVEL == 15) propagate_d3q15();
  if (NVEL == 19) propagate_d3q19();

  TIMER_stop(TIMER_PROPAGATE);

  return;
}

/*****************************************************************************
 *
 *  propagate_d3q15
 *
 *  Follows the definition of the velocities in d3q15.c
 *
 *****************************************************************************/

static void propagate_d3q15(void) {

  int ic, jc, kc, index, n, p;
  int xstr, ystr, zstr;
  int nhalo;
  int ndist;
  int nlocal[3];

  extern double * f_;

  assert(NVEL == 15);

  nhalo = coords_nhalo();
  ndist = distribution_ndist();
  get_N_local(nlocal);

  zstr = ndist*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);

  /* Forward moving distributions */
  
  for (ic = nlocal[X]; ic >= 1; ic--) {
    for (jc = nlocal[Y]; jc >= 1; jc--) {
      for (kc = nlocal[Z]; kc >= 1; kc--) {

        index = get_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  p = ndist*NVEL*index + n*NVEL;
	  f_[p + 7] = f_[p                         + (-1)*zstr + 7];
	  f_[p + 6] = f_[p             + (-1)*ystr             + 6];
	  f_[p + 5] = f_[p + (-1)*xstr + (+1)*ystr + (+1)*zstr + 5];
	  f_[p + 4] = f_[p + (-1)*xstr + (+1)*ystr + (-1)*zstr + 4];
	  f_[p + 3] = f_[p + (-1)*xstr                         + 3];
	  f_[p + 2] = f_[p + (-1)*xstr + (-1)*ystr + (+1)*zstr + 2];
	  f_[p + 1] = f_[p + (-1)*xstr + (-1)*ystr + (-1)*zstr + 1];
	}
      }
    }
  }

  /* Backward moving distributions */
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = get_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  p = ndist*NVEL*index + n*NVEL;
	  f_[p +  8] = f_[p                         + (+1)*zstr +  8];
	  f_[p +  9] = f_[p             + (+1)*ystr             +  9];
	  f_[p + 10] = f_[p + (+1)*xstr + (-1)*ystr + (-1)*zstr + 10];
	  f_[p + 11] = f_[p + (+1)*xstr + (-1)*ystr + (+1)*zstr + 11];
	  f_[p + 12] = f_[p + (+1)*xstr                         + 12];
	  f_[p + 13] = f_[p + (+1)*xstr + (+1)*ystr + (-1)*zstr + 13];
	  f_[p + 14] = f_[p + (+1)*xstr + (+1)*ystr + (+1)*zstr + 14];
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  propagate_d3q19
 *
 *  Follows the velocities defined in d3q19.c
 *
 *****************************************************************************/

static void propagate_d3q19(void) {

  int ic, jc, kc, index, n, p;
  int xstr, ystr, zstr;
  int nhalo;
  int ndist;
  int nlocal[3];

  extern double * f_;

  assert(NVEL == 19);

  nhalo = coords_nhalo();
  ndist = distribution_ndist();
  get_N_local(nlocal);

  zstr = ndist*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);

  /* Distributions moving forward in memory. */
  
  for (ic = nlocal[X]; ic >= 1; ic--) {
    for (jc = nlocal[Y]; jc >= 1; jc--) {
      for (kc = nlocal[Z]; kc >= 1; kc--) {

	index = get_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  p = ndist*NVEL*index + n*NVEL;
	  f_[p + 9] = f_[p                         + (-1)*zstr + 9];
	  f_[p + 8] = f_[p             + (-1)*ystr + (+1)*zstr + 8];
	  f_[p + 7] = f_[p             + (-1)*ystr             + 7];
	  f_[p + 6] = f_[p             + (-1)*ystr + (-1)*zstr + 6];
	  f_[p + 5] = f_[p + (-1)*xstr + (+1)*ystr             + 5];
	  f_[p + 4] = f_[p + (-1)*xstr             + (+1)*zstr + 4];
	  f_[p + 3] = f_[p + (-1)*xstr                         + 3];
	  f_[p + 2] = f_[p + (-1)*xstr             + (-1)*zstr + 2];
	  f_[p + 1] = f_[p + (-1)*xstr + (-1)*ystr             + 1];
	}
      }
    }
  }

  /* Distributions mvoing backward in memory. */
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	for (n = 0; n < ndist; n++) {
	  p = ndist*NVEL*index + n*NVEL;
	  f_[p + 10] = f_[p                         + (+1)*zstr + 10];
	  f_[p + 11] = f_[p             + (+1)*ystr + (-1)*zstr + 11];
	  f_[p + 12] = f_[p             + (+1)*ystr             + 12];
	  f_[p + 13] = f_[p             + (+1)*ystr + (+1)*zstr + 13];
	  f_[p + 14] = f_[p + (+1)*xstr + (-1)*ystr             + 14];
	  f_[p + 15] = f_[p + (+1)*xstr             + (-1)*zstr + 15];
	  f_[p + 16] = f_[p + (+1)*xstr                         + 16];
	  f_[p + 17] = f_[p + (+1)*xstr             + (+1)*zstr + 17];
	  f_[p + 18] = f_[p + (+1)*xstr + (+1)*ystr             + 18];
	}
      }
    }
  }

  return;
}


#endif
