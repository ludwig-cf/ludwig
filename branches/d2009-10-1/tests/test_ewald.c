/*****************************************************************************
 *
 *  test_ewald.c
 *
 *  Some simple tests of the dipole-dipole interactions and
 *  the Ewald sum.
 *
 *  This only works in serial.
 *
 *  $Id: test_ewald.c,v 1.1.2.4 2010-09-30 18:09:42 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "ewald.h"
#include "tests.h"

#define TOLERANCE 1.0e-07

int main (int argc, char ** argv) {

  double mu = 0.285;  /* dipole strength */
  double rc = 32.0;   /* real space cut off (default L / 2) */
  double r1[3];
  double r2[3];
  double r12[3];

  double e;
  double ereal;
  double efourier;
  double eself;

  colloid_t * p_c1;
  colloid_t * p_c2;

  pe_init(argc, argv);

  if (pe_size() > 1) return 0;

  coords_init();

  test_assert(fabs(L(X) - 64.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(L(Y) - 64.0) < TEST_DOUBLE_TOLERANCE);
  test_assert(fabs(L(Z) - 64.0) < TEST_DOUBLE_TOLERANCE);

  colloids_init();
  ewald_init(mu, rc);

  /* First test */

  test_assert(fabs(rc - 32.0) < TOLERANCE);
  test_assert(fabs(mu - 0.285) < TOLERANCE);
  test_assert(fabs(ewald_kappa() - 0.078125) < TOLERANCE);

  r1[X] = 3.0;
  r1[Y] = 3.0;
  r1[Z] = 3.0;

  r2[X] = 3.0;
  r2[Y] = 3.0;
  r2[Z] = 13.0;

  p_c1 = colloid_add_local(1, r1);
  p_c2 = colloid_add_local(2, r2);
  test_assert(p_c1 != NULL);
  test_assert(p_c2 != NULL);
  colloids_ntotal_set();

  /* First colloid .... */

  p_c1->s.a0 = 2.3;
  p_c1->s.ah = 2.3;

  p_c1->s.s[X] = 0.0;
  p_c1->s.s[Y] = 0.0;
  p_c1->s.s[Z] = 1.0;

  /* Second colloid ... */

  p_c2->s.a0 = 2.3;
  p_c2->s.ah = 2.3;

  p_c2->s.s[X] = 0.0;
  p_c2->s.s[Y] = 0.0;
  p_c2->s.s[Z] = -1.0;

  info("Particle 1: %f %f %f %f %f %f\n",
       p_c1->s.r[X], p_c1->s.r[Y], p_c1->s.r[Z],
       p_c1->s.s[X], p_c1->s.s[Y], p_c1->s.s[Z]);

  info("Particle 2: %f %f %f %f %f %f\n",
       p_c2->s.r[X], p_c2->s.r[Y], p_c2->s.r[Z],
       p_c2->s.s[X], p_c2->s.s[Y], p_c2->s.s[Z]);

  coords_minimum_distance(r1, r2, r12);


  e = ewald_real_space_energy(p_c1->s.s, p_c2->s.s, r12);

  info("Real space energy: %g\n", e);
  test_assert(fabs(e - 0.000168995) < TOLERANCE);

  e = ewald_fourier_space_energy();

  info("Fourier space energy: %g\n", e);
  test_assert(fabs(e - 2.25831e-05) < TOLERANCE);

  e = ewald_self_energy();

  info("Self energy term: %g\n", e);
  test_assert(fabs(e - -2.91356e-05) < TOLERANCE);

  /* Now forces */

  info("Computing force and torque...\n");

  ewald_real_space_sum();
  ewald_total_energy(&ereal, &efourier, &eself);

  info("Real space energy: %g\n", ereal);
  test_assert(fabs(ereal - 0.000168995) < TOLERANCE);


  info("Real space force on particle 1: %g %g %g\n",
       p_c1->force[X], p_c1->force[Y], p_c1->force[Z]);

  test_assert(fabs(p_c1->force[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->force[Z] - -5.17464e-05) < TOLERANCE);


  info("Real space force on particle 2: %g %g %g\n",
       p_c2->force[X], p_c2->force[Y], p_c2->force[Z]);

  test_assert(fabs(p_c2->force[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->force[Z] - 5.17464e-05) < TOLERANCE);

  test_assert(fabs(p_c1->force[Z] + p_c2->force[Z]) < TOLERANCE);
  info("Momentum conserved.\n");

  info("Real space torque on particle 1: %g %g %g\n",
       p_c1->torque[X], p_c1->torque[Y], p_c1->torque[Z]);

  test_assert(fabs(p_c1->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Z] - 0.0) < TOLERANCE);

  info("Real space torque on particle 2: %g %g %g\n",
       p_c2->torque[X], p_c2->torque[Y], p_c2->torque[Z]);

  test_assert(fabs(p_c2->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Z] - 0.0) < TOLERANCE);


  /* Fourier space */
  p_c1->force[X] = 0.0; p_c1->force[Y] = 0.0; p_c1->force[Z] = 0.0;
  p_c2->force[X] = 0.0; p_c2->force[Y] = 0.0; p_c2->force[Z] = 0.0;

  ewald_fourier_space_sum();
  ewald_total_energy(&ereal, &efourier, &eself);

  info("Fourier space energy: %g\n", efourier);
  test_assert(fabs(efourier - 2.25831e-05) < TOLERANCE);

  info("Fourier space force on particle 1: %g %g %g\n",
       p_c1->force[X], p_c1->force[Y], p_c1->force[Z]);

  test_assert(fabs(p_c1->force[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->force[Z] - 3.08611e-06) < TOLERANCE);

  info("Fourier space force on particle 2: %g %g %g\n",
       p_c2->force[X], p_c2->force[Y], p_c2->force[Z]);

  test_assert(fabs(p_c2->force[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->force[Z] - -3.08611e-06) < TOLERANCE);

  test_assert(fabs(p_c1->force[Z] + p_c2->force[Z]) < TOLERANCE);
  info("Momentum conserved.\n");

  info("Fourier space torque on particle 1: %g %g %g\n",
       p_c1->torque[X], p_c1->torque[Y], p_c1->torque[Z]);

  test_assert(fabs(p_c1->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Z] - 0.0) < TOLERANCE);

  info("Fourier space torque on particle 2: %g %g %g\n",
       p_c2->torque[X], p_c2->torque[Y], p_c2->torque[Z]);

  test_assert(fabs(p_c2->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Z] - 0.0) < TOLERANCE);


  /* New orientation (non-zero torques). */

  info("\nNew orientation\n");

  p_c2->s.s[X] = 1.0;
  p_c2->s.s[Y] = 0.0;
  p_c2->s.s[Z] = 0.0;

  p_c1->force[X] = 0.0; p_c1->force[Y] = 0.0; p_c1->force[Z] = 0.0;
  p_c2->force[X] = 0.0; p_c2->force[Y] = 0.0; p_c2->force[Z] = 0.0;

  info("Particle 1: %f %f %f %f %f %f\n",
       p_c1->s.r[X], p_c1->s.r[Y], p_c1->s.r[Z],
       p_c1->s.s[X], p_c1->s.s[Y], p_c1->s.s[Z]);

  info("Particle 2: %f %f %f %f %f %f\n",
       p_c2->s.r[X], p_c2->s.r[Y], p_c2->s.r[Z],
       p_c2->s.s[X], p_c2->s.s[Y], p_c2->s.s[Z]);

  /* Energy */

  e = ewald_real_space_energy(p_c1->s.s, p_c2->s.s, r12);

  info("Real space energy: %g\n", e);
  test_assert(fabs(e - 0.0) < TOLERANCE);

  e = ewald_fourier_space_energy();

  info("Fourier space energy: %g\n", e);
  test_assert(fabs(e - 2.76633e-05) < TOLERANCE);

  e = ewald_self_energy();

  info("Self energy term: %g\n", e);
  test_assert(fabs(e - -2.91356e-05) < TOLERANCE);

  /* Forces */

  ewald_real_space_sum();
  ewald_total_energy(&ereal, &efourier, &eself);

  info("Real space energy: %g\n", ereal);
  test_assert(fabs(ereal - 0.0) < TOLERANCE);

  info("Real space force on particle 1: %g %g %g\n",
       p_c1->force[X], p_c1->force[Y], p_c1->force[Z]);

  test_assert(fabs(p_c1->force[X] - -2.29755e-05) < TOLERANCE);
  test_assert(fabs(p_c1->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->force[Z] - 0.0) < TOLERANCE);


  info("Real space force on particle 2: %g %g %g\n",
       p_c2->force[X], p_c2->force[Y], p_c2->force[Z]);

  test_assert(fabs(p_c2->force[X] - 2.29755e-05) < TOLERANCE);
  test_assert(fabs(p_c2->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->force[Z] - 0.0) < TOLERANCE);

  test_assert(fabs(p_c1->force[X] + p_c2->force[X]) < TOLERANCE);
  info("Momentum conserved.\n");


  info("Real space torque on particle 1: %g %g %g\n",
       p_c1->torque[X], p_c1->torque[Y], p_c1->torque[Z]);

  test_assert(fabs(p_c1->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Y] - -6.07598e-05) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Z] - 0.0) < TOLERANCE);

  info("Real space torque on particle 2: %g %g %g\n",
       p_c2->torque[X], p_c2->torque[Y], p_c2->torque[Z]);

  test_assert(fabs(p_c2->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Y] - -0.000168995) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Z] - 0.0) < TOLERANCE);


  /* Fourier space */
  p_c1->force[X] = 0.0; p_c1->force[Y] = 0.0; p_c1->force[Z] = 0.0;
  p_c2->force[X] = 0.0; p_c2->force[Y] = 0.0; p_c2->force[Z] = 0.0;
  p_c1->torque[X] = 0.0; p_c1->torque[Y] = 0.0; p_c1->torque[Z] = 0.0;
  p_c2->torque[X] = 0.0; p_c2->torque[Y] = 0.0; p_c2->torque[Z] = 0.0;


  ewald_fourier_space_sum();
  ewald_total_energy(&ereal, &efourier, &eself);

  info("Fourier space energy: %g\n", efourier);
  test_assert(fabs(efourier - 2.76633e-05) < TOLERANCE);


  info("Fourier space force on particle 1: %g %g %g\n",
       p_c1->force[X], p_c1->force[Y], p_c1->force[Z]);

  test_assert(fabs(p_c1->force[X] - -1.35013e-06) < TOLERANCE);
  test_assert(fabs(p_c1->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->force[Z] - 0.0) < TOLERANCE);

  info("Fourier space force on particle 2: %g %g %g\n",
       p_c2->force[X], p_c2->force[Y], p_c2->force[Z]);

  test_assert(fabs(p_c2->force[X] - 1.35013e-06) < TOLERANCE);
  test_assert(fabs(p_c2->force[Y] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->force[Z] - 0.0) < TOLERANCE);

  test_assert(fabs(p_c1->force[X] + p_c2->force[X]) < TOLERANCE);
  info("Momentum conserved.\n");

  info("Fourier space torque on particle 1: %g %g %g\n",
       p_c1->torque[X], p_c1->torque[Y], p_c1->torque[Z]);

  test_assert(fabs(p_c1->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Y] - -1.92945e-05) < TOLERANCE);
  test_assert(fabs(p_c1->torque[Z] - 0.0) < TOLERANCE);

  info("Fourier space torque on particle 2: %g %g %g\n",
       p_c2->torque[X], p_c2->torque[Y], p_c2->torque[Z]);

  test_assert(fabs(p_c2->torque[X] - 0.0) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Y] - 5.08024e-06) < TOLERANCE);
  test_assert(fabs(p_c2->torque[Z] - 0.0) < TOLERANCE);

  /* New orientation (non-zero torques). */

  info("\nNew orientation\n");

  p_c1->s.r[X] = 3.0;
  p_c1->s.r[Y] = 3.0;
  p_c1->s.r[Z] = 3.0;

  p_c1->s.s[X] = 0.0;
  p_c1->s.s[Y] = 0.0;
  p_c1->s.s[Z] = 1.0;


  p_c2->s.r[X] = 3.0;
  p_c2->s.r[Y] = 13.0;
  p_c2->s.r[Z] = 3.0;

  p_c2->s.s[X] =  0.0;
  p_c2->s.s[Y] = 1.0;
  p_c2->s.s[Z] = 0.0;

  p_c1->force[X] = 0.0; p_c1->force[Y] = 0.0; p_c1->force[Z] = 0.0;
  p_c2->force[X] = 0.0; p_c2->force[Y] = 0.0; p_c2->force[Z] = 0.0;
  p_c1->torque[X] = 0.0; p_c1->torque[Y] = 0.0; p_c1->torque[Z] = 0.0;
  p_c2->torque[X] = 0.0; p_c2->torque[Y] = 0.0; p_c2->torque[Z] = 0.0;

  info("Particle 1: %f %f %f %f %f %f\n",
       p_c1->s.r[X], p_c1->s.r[Y], p_c1->s.r[Z],
       p_c1->s.s[X], p_c1->s.s[Y], p_c1->s.s[Z]);

  info("Particle 2: %f %f %f %f %f %f\n",
       p_c2->s.r[X], p_c2->s.r[Y], p_c2->s.r[Z],
       p_c2->s.s[X], p_c2->s.s[Y], p_c2->s.s[Z]);

  ewald_real_space_sum();
  ewald_total_energy(&ereal, &efourier, &eself);

  info("Real space energy: %g\n", ereal);

  info("Real space force on particle 1: %g %g %g\n",
       p_c1->force[X], p_c1->force[Y], p_c1->force[Z]);

  info("Real space force on particle 2: %g %g %g\n",
       p_c2->force[X], p_c2->force[Y], p_c2->force[Z]);

  info("Real space torque on particle 1: %g %g %g\n",
       p_c1->torque[X], p_c1->torque[Y], p_c1->torque[Z]);

  info("Real space torque on particle 2: %g %g %g\n",
       p_c2->torque[X], p_c2->torque[Y], p_c2->torque[Z]);


  p_c1->force[X] = 0.0; p_c1->force[Y] = 0.0; p_c1->force[Z] = 0.0;
  p_c2->force[X] = 0.0; p_c2->force[Y] = 0.0; p_c2->force[Z] = 0.0;

  ewald_fourier_space_sum();

  info("Fourier space force on particle 1: %g %g %g\n",
       p_c1->force[X], p_c1->force[Y], p_c1->force[Z]);

  info("Fourier space force on particle 2: %g %g %g\n",
       p_c2->force[X], p_c2->force[Y], p_c2->force[Z]);

  info("Fourier space torque on particle 1: %g %g %g\n",
       p_c1->torque[X], p_c1->torque[Y], p_c1->torque[Z]);

  info("Fourier space torque on particle 2: %g %g %g\n",
       p_c2->torque[X], p_c2->torque[Y], p_c2->torque[Z]);

  ewald_finish();



  /* Now set cut-off = 8.0. */

  ewald_init(0.285, 8.0);

  e = ewald_real_space_energy(p_c1->s.s, p_c2->s.s, r12);

  info("Real space energy: %g\n", e);
  test_assert(fabs(e - 0.0) < TOLERANCE);

  e = ewald_fourier_space_energy();

  info("Fourier space energy: %g\n", e);
  /* test_assert(fabs(e - 0.000265242) < TOLERANCE);*/

  e = ewald_self_energy();

  info("Self energy term: %g\n", e);
  /* test_assert(fabs(e - -0.00186468) < TOLERANCE);*/

  ewald_finish();

  info("Ewald tests ok\n");

  coords_finish();
  pe_finalise();

  return 0;
}
