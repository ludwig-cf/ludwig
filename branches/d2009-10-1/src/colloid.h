/*****************************************************************************
 *
 *  colloid.h
 *
 *  The implementation is exposed for the time being.
 *
 *  $Id: colloid.h,v 1.1.2.1 2010-06-06 11:39:28 kevin Exp $
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

#include <stdio.h>

typedef struct colloid_state_type colloid_state_t;

struct colloid_state_type {

  int    index;         /* Unique global index for colloid */
  int    rebuild;       /* Rebuild flag */
  double a0;            /* Input radius (lattice units) */
  double ah;            /* Hydrodynamic radius (from calibration) */
  double r[3];          /* Position */
  double v[3];          /* Velocity */
  double w[3];          /* Angular velocity omega */
  double s[3];          /* Magnetic dipole, or spin */
  double m[3];          /* Currect direction of motion vector (squirmer) */
  double b1;	        /* squirmer active parameter b1 */
  double b2;            /* squirmer active parameter b2 */
  double c;             /* Wetting free energy parameter C */
  double h;             /* Wetting free energy parameter H */
  double dr[3];         /* r update (pending refactor of move/build process) */
  double deltaphi;      /* order parameter bbl net; required to restart */
  double spare1;        /* spare scalar */
  double spare2[3];     /* spare vector */

};

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp);
int colloid_state_read_binary(colloid_state_t * ps, FILE * fp);
int colloid_state_write_ascii(colloid_state_t ps, FILE * fp);
int colloid_state_write_binary(colloid_state_t ps, FILE * fp);

#endif
