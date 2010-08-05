/*****************************************************************************
 *
 *  physics.h
 *
 *  $Id: physics.h,v 1.3.16.1 2010-08-05 17:22:27 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHYSICS_H
#define PHYSICS_H

void   init_physics(void);

double get_eta_shear(void);
double get_eta_bulk(void);
double get_kT(void);
double get_rho0(void);
double get_phi0(void);

void set_eta(const double);

double fluid_kt(void);
void   fluid_body_force(double f[3]);
void   fluid_body_force_set(const double f[3]);

#endif
