/*****************************************************************************
 *
 *  magnetic_field.h
 *
 *  $Id: magnetic_field.h,v 1.1 2010-03-24 11:43:09 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef MAGNETIC_FIELD_H
#define MAGNETIC_FIELD_H

void magnetic_field_b0(double b0[3]);
void magnetic_field_b0_set(const double b0[3]);

void magnetic_field_force(const double mu[3], double force[3]);
void magnetic_field_torque(const double mu[3], double torque[3]);

int electric_field_e0(double e0[3]);
int electric_field_e0_set(const double e0[3]);

#endif
