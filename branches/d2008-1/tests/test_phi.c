/*****************************************************************************
 *
 *  test_phi.c
 *
 *  Tests for order parameter stuff.
 *
 *  $Id: test_phi.c,v 1.1.2.1 2008-03-21 09:58:39 kevin Exp $
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
#include "phi_gradients.h"
#include "tests.h"

static void   test_phi_interface(void);
static void   test_phi_halo(double (*)(int, int, int));
static double test_function1(int, int, int);
static double test_function2(int, int, int);
static double test_function3(int, int, int);

int main (int argc, char ** argv) {

  pe_init(argc, argv);
  coords_init();

  info("\nOrder parameter tests...\n");

  phi_init();

  test_phi_interface();

  info("The halo region width is nhalo_ = %d\n", nhalo_);
  info("Checking X-direction halo...");
  test_phi_halo(test_function1);
  info("ok\n");

  info("Checking Y-direction halo...");
  test_phi_halo(test_function2);
  info("ok\n");

  info("Checking Z-direction halo...");
  test_phi_halo(test_function3);
  info("ok\n");

  phi_finish();
  pe_finalise();

  return 0;
}

/*****************************************************************************
 *
 *  test_phi_interface
 *
 *  Check the interface is present and works at a basic level.
 *
 *****************************************************************************/

void test_phi_interface() {

  int index = 0;
  double phi_in; /* Non-zero value */
  double phi, dphi_in[3], dphi[3];

  info("Checking phi functions...");

  phi_in = 1.0;
  phi_set_phi_site(index, phi_in);
  phi = phi_get_phi_site(index);
  test_assert(fabs(phi - phi_in) < TEST_DOUBLE_TOLERANCE);

  phi_in = 2.0;
  phi_set_delsq_phi_site(index, phi_in);
  phi = phi_get_delsq_phi_site(index);
  test_assert(fabs(phi - phi_in) < TEST_DOUBLE_TOLERANCE);

  dphi_in[X] = 3.0;
  dphi_in[Y] = 4.0;
  dphi_in[Z] = 5.0;

  phi_set_grad_phi_site(index, dphi_in);
  phi_get_grad_phi_site(index, dphi);
  test_assert(fabs(dphi[X] - dphi_in[X]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(dphi[Y] - dphi_in[Y]) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(dphi[Z] - dphi_in[Z]) < TEST_DOUBLE_TOLERANCE);
  info("ok\n");

  return;
}

/*****************************************************************************
 *
 *  test_phi_halo
 *
 *  Set up a test function phi(x, y, z) and ensure the halo swap
 *  works (for current nhalo).
 *
 *****************************************************************************/

void test_phi_halo(double (* test_function)(int, int, int)) {

  int nlocal[3];
  int ic, jc, kc, index;
  double phi, phi_original;

  get_N_local(nlocal);

  /* Set test values (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	phi = test_function(ic, jc, kc);
	phi_set_phi_site(index, phi);
      }
    }
  }

  /* Halo swap */
  phi_halo();

  /* Check test values (everywhere) */

  for (ic = 1 - nhalo_; ic <= nlocal[X] + nhalo_; ic++) {
    for (jc = 1 - nhalo_; jc <= nlocal[Y] + nhalo_; jc++) {
      for (kc = 1 - nhalo_; kc <= nlocal[Z] + nhalo_; kc++) {

	index = get_site_index(ic, jc, kc);

	phi_original = test_function(ic, jc, kc);
	phi = phi_get_phi_site(index);
	test_assert(fabs(phi - phi_original) < TEST_DOUBLE_TOLERANCE);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_function1
 *
 *  phi(x, y, z) = sin(2\pi x/L_x)
 *
 *****************************************************************************/

double test_function1(int ic, int jc, int kc) {

  double phi, pi, x;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  x = (double) (ic + noffset[X])  - Lmin(X); 
  phi = sin(2.0*pi*x/L(X));

  return phi;
}

/*****************************************************************************
 *
 *  test_function2
 *
 *  phi(x, y, z) = sin(2\pi y/L_y)
 *
 *****************************************************************************/

double test_function2(int ic, int jc, int kc) {

  double phi, pi, y;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  y = (double) (jc + noffset[Y])  - Lmin(Y); 
  phi = sin(2.0*pi*y/L(Y));

  return phi;
}

/*****************************************************************************
 *
 *  test_function3
 *
 *  phi(x, y, z) = sin(2\pi z/L_z)
 *
 *****************************************************************************/

double test_function3(int ic, int jc, int kc) {

  double phi, pi, z;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  z = (double) (kc + noffset[Z])  - Lmin(Z); 
  phi = sin(2.0*pi*z/L(Z));

  return phi;
}
