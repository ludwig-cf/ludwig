/*****************************************************************************
 *
 *  test_phi_gradients
 *
 *  Test the finite-difference approximations to the various
 *  gradients required.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "phi.h"
#include "phi_gradients.h"
#include "tests.h"

static void   test_grad_phi(double (*)(int, int, int),
			    void (*)(int, int, int, double [3]));
static void   test_grad_delsq_phi(double (*)(int, int, int),
				  void (*)(int, int, int, double [3]));
static void   test_delsq_phi(double (*)(int, int, int),
			     double (*)(int, int, int));
static void   test_delsq_delsq_phi(double (*)(int, int, int),
				   double (*)(int, int, int));
static double test_function1(int, int, int);
static double test_function2(int, int, int);
static double test_function3(int, int, int);
static void   test_function1_grad(int, int, int, double [3]);
static void   test_function2_grad(int, int, int, double [3]);
static void   test_function3_grad(int, int, int, double [3]);
static double test_function1_delsq(int, int, int);
static double test_function2_delsq(int, int, int);
static double test_function3_delsq(int, int, int);
static void   test_function1_grad_delsq(int, int, int, double[3]);
static void   test_function2_grad_delsq(int, int, int, double[3]);
static void   test_function3_grad_delsq(int, int, int, double[3]);

int main(int argc, char ** argv) {

  MPI_Init(&argc, &argv);
  pe_init();
  coords_init();
  le_init();
  phi_init();

  info("The number of halo points for phi is: %d\n", nhalo_);

  info("Testing grad phi...\n");

  /* We require halo at least width 1 ... */
  test_assert(nhalo_ >= 1);

  test_grad_phi(test_function1, test_function1_grad);
  test_grad_phi(test_function2, test_function2_grad);
  test_grad_phi(test_function3, test_function3_grad);

  info("Testing \\nabla^2 phi...\n");

  test_delsq_phi(test_function1, test_function1_delsq);
  test_delsq_phi(test_function2, test_function2_delsq);
  test_delsq_phi(test_function3, test_function3_delsq);

  if (nhalo_ >= 2) {
    
    info("Testing grad \\nabla^2 phi...\n");
    test_grad_delsq_phi(test_function1, test_function1_grad_delsq);
    test_grad_delsq_phi(test_function2, test_function2_grad_delsq);
    test_grad_delsq_phi(test_function3, test_function3_grad_delsq);

    info("Testing (\\nabla^2)^2 phi...\n");

    test_delsq_delsq_phi(test_function1, test_function1);
    test_delsq_delsq_phi(test_function2, test_function2);
    test_delsq_delsq_phi(test_function3, test_function3);
  }

  info("All gradients ok\n");

  pe_finalise();
  MPI_Finalize();

  return 0;
}


/*****************************************************************************
 *
 *  test_grad_phi
 *
 *  Test the finite-difference approximation to \grad phi.
 *
 *  The first argument is the function returning phi(i,j,k)
 *  and the second is the function to compute exact grad phi(i,j,k).
 *
 *****************************************************************************/

void test_grad_phi(double (* f)(int, int, int),
		   void (* grad)(int, int, int, double [3])) {

  int nlocal[3];
  int ic, jc, kc, index;
  int nextra = nhalo_ - 1;
  double phi, dphi[3];
  double dphi_exact[3];

  double tolerance;

  /* We assume the tolerance for the finite difference gradient
   * is roughly the maximum value / L */

  tolerance = 2.0*4.0*atan(1.0)/(L(X)*L(X));

  /* Set gradient calculation to fluid only */

  phi_gradients_set_fluid();

  get_N_local(nlocal);

  /* Set test values (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	phi = f(ic, jc, kc);
	phi_set_phi_site(index, phi);
      }
    }
  }

  phi_halo();
  phi_gradients_compute();

  /* Test gradients, including nextra points of the halo */

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = get_site_index(ic, jc, kc);

	phi_get_grad_phi_site(index, dphi);
	grad(ic, jc, kc, dphi_exact);

	test_assert(fabs(dphi[X] - dphi_exact[X]) < tolerance);
	test_assert(fabs(dphi[Y] - dphi_exact[Y]) < tolerance);
	test_assert(fabs(dphi[Z] - dphi_exact[Z]) < tolerance);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_delsq_phi
 *
 *  Test the second derivative.
 *
 *****************************************************************************/

void test_delsq_phi(double (* f)(int, int, int),
		    double (* delsq)(int, int, int)) {

  int nlocal[3];
  int ic, jc, kc, index;
  int nextra = nhalo_ - 1;
  double phi, delsq_phi, delsq_exact;

  double tolerance;

  /* We assume the tolerance for the finite difference gradient
   * is roughly the maximum value / L */

  tolerance = pow(2.0*4.0*atan(1.0), 2)/(L(X)*L(X)*L(X));

  /* Set gradient calculation to fluid only */

  phi_gradients_set_fluid();

  get_N_local(nlocal);

  /* Set test values (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	phi = f(ic, jc, kc);
	phi_set_phi_site(index, phi);
      }
    }
  }

  phi_halo();
  phi_gradients_compute();

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = get_site_index(ic, jc, kc);
	delsq_phi = phi_get_delsq_phi_site(index);
	delsq_exact = delsq(ic, jc, kc);

	test_assert(fabs(delsq_phi - delsq_exact) < tolerance);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_grad_delsq_phi
 *
 *  Test the gradient of \nabla^2 f.
 *
 *  The first argument is the function returning f, while the second
 *  is the function returning the exact result.
 *
 *****************************************************************************/

void test_grad_delsq_phi(double (* f)(int, int, int),
			 void (* grad_delsq)(int, int, int, double [3])) {

  int nlocal[3];
  int ic, jc, kc, index;
  int nextra = nhalo_ - 2;
  double phi, grad_delsq_phi[3];
  double grad_delsq_exact[3];

  double tolerance;

  /* We assume the tolerance for the finite difference gradient
   * is roughly the maximum value / L */

  tolerance = pow(2.0*4.0*atan(1.0), 3)/(pow(L(X), 3)*L(X));

  /* Set gradient calculation to fluid only */

  phi_gradients_set_fluid();

  get_N_local(nlocal);

  /* Set test values (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	phi = f(ic, jc, kc);
	phi_set_phi_site(index, phi);
      }
    }
  }

  phi_halo();
  phi_gradients_compute();
  phi_gradients_double_fluid();

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = get_site_index(ic, jc, kc);
	phi_get_grad_delsq_phi_site(index, grad_delsq_phi);
	grad_delsq(ic, jc, kc, grad_delsq_exact);

	test_assert(fabs(grad_delsq_phi[X] - grad_delsq_exact[X]) < tolerance);
	test_assert(fabs(grad_delsq_phi[Y] - grad_delsq_exact[Y]) < tolerance);
	test_assert(fabs(grad_delsq_phi[Z] - grad_delsq_exact[Z]) < tolerance);
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  test_delsq_delsq_phi
 *
 *  Test the fourth derivative.
 *
 *  The first argument is the function to be tested, while the second
 *  gives the exact result.
 *
 *****************************************************************************/

void test_delsq_delsq_phi(double (* f)(int, int, int),
			  double (* delsqsq)(int, int, int)) {

  int nlocal[3];
  int ic, jc, kc, index;
  int nextra = nhalo_ - 2;
  double phi, delsqsq_phi, delsqsq_exact;

  double tolerance;

  /* We assume the tolerance for the finite difference gradient
   * is roughly the maximum value / L */

  tolerance = pow(2.0*4.0*atan(1.0), 4)/(pow(L(X), 4)*L(X));

  /* Set gradient calculation to fluid only */

  phi_gradients_set_fluid();

  get_N_local(nlocal);

  /* Set test values (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = get_site_index(ic, jc, kc);

	phi = f(ic, jc, kc);
	phi_set_phi_site(index, phi);
      }
    }
  }

  phi_halo();
  phi_gradients_compute();
  phi_gradients_double_fluid();


  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = get_site_index(ic, jc, kc);

	phi = phi_get_phi_site(index);
	delsqsq_phi = phi_get_delsq_delsq_phi_site(index);
	delsqsq_exact = L(X)*tolerance*f(ic, jc, kc);

	test_assert(fabs(delsqsq_phi - delsqsq_exact) < tolerance);
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
 *  test_function1_grad
 *
 *  The gradient of the above, exactly.
 *
 *****************************************************************************/

void test_function1_grad(int ic, int jc, int kc, double grad[3]) {

  double dphi, pi, x;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  x = (double) (ic + noffset[X])  - Lmin(X); 
  dphi = 2.0*pi*cos(2.0*pi*x/L(X))/L(X);

  grad[X] = dphi;
  grad[Y] = 0.0;
  grad[Z] = 0.0;

  return;
}

/*****************************************************************************
 *
 *  test_function1_delsq
 *
 *  Return \nabla^2 sin(2pi x / L_x)
 *
 *****************************************************************************/

double test_function1_delsq(int ic, int jc, int kc) {

  double delsq, pi, x;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  x = (double) (ic + noffset[X])  - Lmin(X); 
  delsq = -4.0*pi*pi*sin(2.0*pi*x/L(X))/(L(X)*L(X));

  return delsq;
}

/*****************************************************************************
 *
 *  test_function1_grad_delsq
 *
 *  phi(x, y, z) = sin(2\pi x/L_x)
 *
 *****************************************************************************/

void test_function1_grad_delsq(int ic, int jc, int kc, double grad[3]) {

  double graddelsq, pi, x;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global x coordinate */

  x = (double) (ic + noffset[X])  - Lmin(X); 
  graddelsq = -8.0*pi*pi*pi*cos(2.0*pi*x/L(X))/(pow(L(X), 3));

  grad[X] = graddelsq;
  grad[Y] = 0.0;
  grad[Z] = 0.0;

  return;
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
 *  test_function2_grad
 *
 *  The gradient of the above, exactly.
 *
 *****************************************************************************/

void test_function2_grad(int ic, int jc, int kc, double grad[3]) {

  double dphi, pi, y;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global y coordinate */

  y = (double) (jc + noffset[Y])  - Lmin(Y); 
  dphi = 2.0*pi*cos(2.0*pi*y/L(Y))/L(Y);

  grad[X] = 0.0;
  grad[Y] = dphi;
  grad[Z] = 0.0;

  return;
}

/*****************************************************************************
 *
 *  test_function2_delsq
 *
 *  Return \nabla^2 sin(2pi y / L_y)
 *
 *****************************************************************************/

double test_function2_delsq(int ic, int jc, int kc) {

  double delsq, pi, y;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global y coordinate */

  y = (double) (jc + noffset[Y])  - Lmin(Y); 
  delsq = -4.0*pi*pi*sin(2.0*pi*y/L(Y))/(L(Y)*L(Y));

  return delsq;
}

/*****************************************************************************
 *
 *  test_function2_grad_delsq
 *
 *  phi(x, y, z) = sin(2\pi y/L_y)
 *
 *****************************************************************************/

void test_function2_grad_delsq(int ic, int jc, int kc, double grad[3]) {

  double graddelsq, pi, y;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global y coordinate */

  y = (double) (jc + noffset[Y])  - Lmin(Y); 
  graddelsq = -8.0*pi*pi*pi*cos(2.0*pi*y/L(Y))/(pow(L(Y), 3));

  grad[X] = 0.0;
  grad[Y] = graddelsq;
  grad[Z] = 0.0;

  return;
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

/*****************************************************************************
 *
 *  test_function3_grad
 *
 *  The gradient of the above, exactly.
 *
 *****************************************************************************/

void test_function3_grad(int ic, int jc, int kc, double grad[3]) {

  double dphi, pi, z;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global z coordinate */

  z = (double) (kc + noffset[Z])  - Lmin(Z); 
  dphi = 2.0*pi*cos(2.0*pi*z/L(Z))/L(Z);

  grad[X] = 0.0;
  grad[Y] = 0.0;
  grad[Z] = dphi;

  return;
}

/*****************************************************************************
 *
 *  test_function3_delsq
 *
 *  Return \nabla^2 sin(2pi z / L_z)
 *
 *****************************************************************************/

double test_function3_delsq(int ic, int jc, int kc) {

  double delsq, pi, z;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global z coordinate */

  z = (double) (kc + noffset[Z])  - Lmin(Z); 
  delsq = -4.0*pi*pi*sin(2.0*pi*z/L(Z))/(L(Z)*L(Z));

  return delsq;
}

/*****************************************************************************
 *
 *  test_function3_grad_delsq
 *
 *  phi(x, y, z) = sin(2\pi z/L_z)
 *
 *****************************************************************************/

void test_function3_grad_delsq(int ic, int jc, int kc, double grad[3]) {

  double graddelsq, pi, z;
  int noffset[3];

  get_N_offset(noffset);
  pi = 4.0*atan(1.0);

  /* Work out the global z coordinate */

  z = (double) (kc + noffset[Z])  - Lmin(Z); 
  graddelsq = -8.0*pi*pi*pi*cos(2.0*pi*z/L(Z))/(pow(L(Z), 3));

  grad[X] = 0.0;
  grad[Y] = 0.0;
  grad[Z] = graddelsq;

  return;
}
