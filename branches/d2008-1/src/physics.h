/*****************************************************************************
 *
 *  physics.h
 *
 *  $Id: physics.h,v 1.2.4.1 2008-03-21 09:20:18 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _PHYSICS_H
#define _PHYSICS_H

void   init_physics(void);

double get_eta_shear(void);
double get_eta_bulk(void);
double get_kT(void);
void   set_kT(const double);
double get_rho0(void);
double get_phi0(void);
void   get_gravity(double []);

void set_eta(const double);

#endif
