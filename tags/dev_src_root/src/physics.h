/*****************************************************************************
 *
 *  physics.h
 *
 *  $Id: physics.h,v 1.2 2007-03-09 12:56:09 kevin Exp $
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

#endif
