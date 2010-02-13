/*****************************************************************************
 *
 *  propagation.c
 *
 *  Propagation schemes for the different models.
 *
 *  $Id: propagation.c,v 1.4.16.3 2010-02-13 16:28:36 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "timer.h"
#include "coords.h"
#include "model.h"

static void propagate_d2q9(void);
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

  if (NVEL == 9) propagate_d2q9();
  if (NVEL == 15) propagate_d3q15();
  if (NVEL == 19) propagate_d3q19();

  TIMER_stop(TIMER_PROPAGATE);

  return;
}

/*****************************************************************************
 *
 *  propagate_d2q9
 *
 *  Follows the definition of the velocities in d2q9.c
 *
 *****************************************************************************/

static void propagate_d2q9(void) {

  int ic, jc, kc, index, n, p;
  int xstr, ystr, zstr;
  int nhalo;
  int ndist;
  int nlocal[3];

  extern double * f_;

  assert(NVEL == 9);

  nhalo = coords_nhalo();
  ndist = distribution_ndist();
  get_N_local(nlocal);

  zstr = ndist*NVEL;
  ystr = zstr*(nlocal[Z] + 2*nhalo);
  xstr = ystr*(nlocal[Y] + 2*nhalo);

  /* Forward moving distributions in memory */
  
  for (ic = nlocal[X]; ic >= 1; ic--) {
    for (jc = nlocal[Y]; jc >= 1; jc--) {

      kc = 1;
      index = get_site_index(ic, jc, kc);

      for (n = 0; n < ndist; n++) {
	p = ndist*NVEL*index + n*NVEL;
	f_[p + 4] = f_[p +             (-1)*ystr + 4];
	f_[p + 3] = f_[p + (-1)*xstr + (+1)*ystr + 3];
	f_[p + 2] = f_[p + (-1)*xstr             + 2];
	f_[p + 1] = f_[p + (-1)*xstr + (-1)*ystr + 1];
      }
    }
  }

  /* Backward moving distributions in memory */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      kc = 1;
      index = get_site_index(ic, jc, kc);

      for (n = 0; n < ndist; n++) {
	p = ndist*NVEL*index + n*NVEL;
	f_[p + 5] = f_[p             + (+1)*ystr + 5];
	f_[p + 6] = f_[p + (+1)*xstr + (-1)*ystr + 6];
	f_[p + 7] = f_[p + (+1)*xstr             + 7];
	f_[p + 8] = f_[p + (+1)*xstr + (+1)*ystr + 8];
      }
    }
  }

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
