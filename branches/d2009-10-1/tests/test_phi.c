/*****************************************************************************
 *
 *  test_phi.c
 *
 *  Tests for order parameter stuff.
 *
 *  $Id: test_phi.c,v 1.3.2.2 2010-04-05 06:23:46 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "phi.h"
#include "tests.h"

static void   test_phi_generic(int nop);
static void   test_phi_interface_single(void);
static void   test_phi_halo(double (*)(int, int, int, int));
static void   test_phi_2d_halo(void);
static double test_function1(int, int, int, int);
static double test_function2(int, int, int, int);
static double test_function3(int, int, int, int);

int main (int argc, char ** argv) {

  int nop;
  int nhalo = 2;
  int n[3];

  pe_init(argc, argv);

  coords_nhalo_set(nhalo);
  coords_init();

  info("\nOrder parameter tests...\n");

  /* Single order parameter */

  nop = 1;
  info("nop = 1...\n");
  phi_nop_set(nop);

  phi_init();
  test_phi_generic(nop);
  test_phi_interface_single();
  phi_finish();

  /* Tensor order parameter */

  nop = 5;
  info("nop = 5...\n");
  phi_nop_set(nop);
  phi_init();
  test_phi_generic(nop);
  phi_finish();

  coords_finish();

  /* Check the halo swap for a 2-d system. */

  n[X] = 64;
  n[Y] = 64;
  n[Z] = 1;
  nhalo = 3;

  coords_ntotal_set(n);
  coords_nhalo_set(nhalo);
  coords_init();

  nop = 1;
  phi_nop_set(nop);
  phi_init();

  test_phi_2d_halo();

  phi_finish();
  coords_finish();

  pe_finalise();

  return 0;

}

/*****************************************************************************
 *
 *  test_phi_generic
 *
 *****************************************************************************/

void test_phi_generic(int nop) {

  test_assert(nop == phi_nop());

  info("The halo region width is nhalo = %d\n", coords_nhalo());
  info("Checking X-direction halo...");
  test_phi_halo(test_function1);
  info("ok\n");

  info("Checking Y-direction halo...");
  test_phi_halo(test_function2);
  info("ok\n");

  info("Checking Z-direction halo...");
  test_phi_halo(test_function3);
  info("ok\n");

  return;
}

/*****************************************************************************
 *
 *  test_phi_interface_single
 *
 *  Specifically, routines for single order parameter. Should
 *  be phased out.
 *
 *****************************************************************************/

void test_phi_interface_single() {

  int index = 0;
  double phi_in; /* Non-zero value */
  double phi;

  info("Checking phi functions...");

  phi_in = 1.0;
  phi_set_phi_site(index, phi_in);
  phi = phi_get_phi_site(index);
  test_assert(fabs(phi - phi_in) < TEST_DOUBLE_TOLERANCE);

  info("ok\n");

  return;
}

/*****************************************************************************
 *
 *  test_phi_halo
 *
 *  Set up a test function phi(x, y, z) and ensure the halo swap
 *  works (for current nop and nhalo).
 *
 *****************************************************************************/

void test_phi_halo(double (* test_function)(int, int, int, int)) {

  int nlocal[3];
  int ic, jc, kc, index;
  int n, nop, nhalo;
  double phi, phi_original;

  coords_nlocal(nlocal);
  nhalo = coords_nhalo();
  nop = phi_nop();

  /* Set test values (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop; n++) {
	  phi = test_function(ic, jc, kc, n);
	  phi_op_set_phi_site(index, n, phi);
	}
      }
    }
  }

  /* Halo swap */
  phi_halo();

  /* Check test values (everywhere) */

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop; n++) {
	  phi_original = test_function(ic, jc, kc, n);
	  phi = phi_op_get_phi_site(index, n);
	  test_assert(fabs(phi - phi_original) < TEST_DOUBLE_TOLERANCE);
	}
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_function1
 *
 *  phi(x, y, z, n) = sin(2\pi x/L_x) + n
 *
 *****************************************************************************/

double test_function1(int ic, int jc, int kc, int n) {

  double phi, pi, x;
  int noffset[3];

  coords_nlocal_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  x = (double) (ic + noffset[X])  - Lmin(X); 
  phi = sin(2.0*pi*x/L(X)) + 1.0*n;

  return phi;
}

/*****************************************************************************
 *
 *  test_function2
 *
 *  phi(x, y, z, n) = sin(2\pi y/L_y) + n
 *
 *****************************************************************************/

double test_function2(int ic, int jc, int kc, int n) {

  double phi, pi, y;
  int noffset[3];

  coords_nlocal_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  y = (double) (jc + noffset[Y])  - Lmin(Y); 
  phi = sin(2.0*pi*y/L(Y)) + 1.0*n;

  return phi;
}

/*****************************************************************************
 *
 *  test_function3
 *
 *  phi(x, y, z, n) = sin(2\pi z/L_z) + n
 *
 *****************************************************************************/

double test_function3(int ic, int jc, int kc, int n) {

  double phi, pi, z;
  int noffset[3];

  coords_nlocal_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  z = (double) (kc + noffset[Z])  - Lmin(Z); 
  phi = sin(2.0*pi*z/L(Z)) + 1.0*n;

  return phi;
}

/*****************************************************************************
 *
 *  test_phi_2d_halo
 *
 *  In a 2-d system, check that the halos come out correctly
 *  (particlarly for nhalo > 1). The 2-d system is in XY.
 *
 *****************************************************************************/

void test_phi_2d_halo(void) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nop;
  int nextra;

  double phi;

  coords_nlocal(nlocal);
  nextra = coords_nhalo();
  nop = phi_nop();

  info("2-d halo swap test nhalo = %d...", nextra);

  /* Set everything to zero. */

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop; n++) {
	  phi_op_set_phi_site(index, n, 0.0);
	}

      }
    }
  }

  /* Set in the domain as function(x,y) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {

      kc = 1;

      index = coords_index(ic, jc, kc);

      for (n = 0; n < nop; n++) {
	phi = 0.1*n;
	phi_op_set_phi_site(index, n, phi);
      }

    }
  }

  phi_halo();

  /* Check all the halo planes, and the domain, are the same. */

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(ic, jc, kc);

	for (n = 0; n < nop; n++) {
	  phi = phi_op_get_phi_site(index, n);
	  test_assert(fabs(phi - 0.1*n) < TEST_DOUBLE_TOLERANCE);
	}

      }
    }
  }

  info("ok\n\n");

  return;
}
