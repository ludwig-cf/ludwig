/*****************************************************************************
 *
 *  magnetic_field.c
 *
 *  Uniform magnetic field. This could be extended to cover
 *  other types of (non-uniform) field.
 *
 *  In a uniform field b0 the force and torque on a dipole mu
 *  are:
 *
 *     force = 0
 *     torque = mu x b0
 *
 *  regardless of position.
 *
 *  Currently includes external electric field.
 *
 *  $Id: magnetic_field.c,v 1.1 2010-03-24 11:43:09 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "pe.h"
#include "coords.h"
#include "magnetic_field.h"

static double bzero_[3] = {0.0, 0.0, 0.0};
static double ezero_[3] = {0.0, 0.0, 0.0};

/*****************************************************************************
 *
 *  magnetic_field_b0
 *
 *****************************************************************************/

void magnetic_field_b0(double b0[3]) {

  b0[X] = bzero_[X];
  b0[Y] = bzero_[Y];
  b0[Z] = bzero_[Z];

  return;
}

/*****************************************************************************
 *
 *  magnetic_field_b0_set
 *
 *****************************************************************************/

void magnetic_field_b0_set(const double b0[3]) {

  bzero_[X] = b0[X];
  bzero_[Y] = b0[Y];
  bzero_[Z] = b0[Z];

  return;
}

/*****************************************************************************
 *
 *  magnetic_field_force
 *
 *****************************************************************************/

void magnetic_field_force(const double mu[3], double force[3]) {

  force[X] = 0.0;
  force[Y] = 0.0;
  force[Z] = 0.0;

  return;
}

/*****************************************************************************
 *
 *  magnetic_field_torque
 *
 *****************************************************************************/

void magnetic_field_torque(const double mu[3], double torque[3]) {

  torque[X] = mu[Y]*bzero_[Z] - mu[Z]*bzero_[Y];
  torque[Y] = mu[Z]*bzero_[X] - mu[X]*bzero_[Z];
  torque[Z] = mu[X]*bzero_[Y] - mu[Y]*bzero_[X];

  return;
}

/*****************************************************************************
 *
 *  electric_field_e0
 *
 *****************************************************************************/

int electric_field_e0(double e0[3]) {

  e0[0] = ezero_[0];
  e0[1] = ezero_[1];
  e0[2] = ezero_[2];

  return 0;
}

/*****************************************************************************
 *
 *  electric_field_e0_set
 *
 *****************************************************************************/

int electric_field_e0_set(const double e0[3]) {

  ezero_[0] = e0[0];
  ezero_[1] = e0[1];
  ezero_[2] = e0[2];

  return 0;
}
