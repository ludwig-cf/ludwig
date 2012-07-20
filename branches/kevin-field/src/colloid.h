/*****************************************************************************
 *
 *  colloid.h
 *
 *  The implementation is exposed for the time being.
 *
 *  $Id: colloid.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOID_H
#define COLLOID_H

/* These describe the padding etc, and are really for internal
 * unit test consumption. The total number of variables is
 * useful to know to check the ASCII read/write. */

#define NTOT_VAR (32+48)
#define NPAD_INT  24
#define NPAD_DBL  20

#include <stdio.h>

typedef struct colloid_state_type colloid_state_t;

struct colloid_state_type {

  int index;            /* Unique global index for colloid */
  int rebuild;          /* Rebuild flag */
  int nbonds;           /* Number of bonds e.g. fene (always 2 at moment) */
  int nangles;          /* Number of angles, e.g., fene (1 at the moment) */

  int isfixedr;         /* Set to 1 for no position update */
  int isfixedv;         /* Set to 1 for no velocity update */
  int isfixedw;         /* Set to 1 for no angular velocity update */
  int isfixeds;         /* Set to zero for no s, m update */

  int intpad[NPAD_INT]; /* I'm going to pad to 32 ints to allow for future
			 * expansion. Additions should be appended here,
			 * and the padding reduced appropriately. */

  double a0;            /* Input radius (lattice units) */
  double ah;            /* Hydrodynamic radius (from calibration) */
  double r[3];          /* Position */
  double v[3];          /* Velocity */
  double w[3];          /* Angular velocity omega */
  double s[3];          /* Magnetic dipole, or spin */
  double m[3];          /* Current direction of motion vector (squirmer) */
  double b1;	        /* squirmer active parameter b1 */
  double b2;            /* squirmer active parameter b2 */
  double c;             /* Wetting free energy parameter C */
  double h;             /* Wetting free energy parameter H */
  double dr[3];         /* r update (pending refactor of move/build process) */
  double deltaphi;      /* order parameter bbl net; required to restart */

  /* Charges. We allow two charge valencies (cf a general number
   * number in the electrokinetics section). q0 will be associated
   * with psi->rho[0] and q1 to psi->rho[1] in the electrokinetics.
   * The charge will
   * be converted to a density by dividing by the current discrete
   * volume to ensure conservation. */

  double q0;            /* magnitude charge 0 */
  double q1;            /* magnitude charge 1 */
  double epsilon;       /* permeativity */

  double dpad[NPAD_DBL];/* Again, this pads to 512 bytes to allow
			 * for future expansion. */
};

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp);
int colloid_state_read_binary(colloid_state_t * ps, FILE * fp);
int colloid_state_write_ascii(colloid_state_t ps, FILE * fp);
int colloid_state_write_binary(colloid_state_t ps, FILE * fp);

#endif
