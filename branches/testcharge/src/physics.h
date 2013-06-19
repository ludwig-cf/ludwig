/*****************************************************************************
 *
 *  physics.h
 *
 *  $Id: physics.h,v 1.4 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2013 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHYSICS_H
#define PHYSICS_H

typedef struct physics_s physics_t;

int physics_ref(physics_t ** ref);
int physics_free(void);

int physics_rho0(double * rho);
int physics_rho0_set(double rho0);
int physics_phi0(double * phi0);
int physics_phi0_set(double phi0);

int physics_eta_shear(double * eta);
int physics_eta_shear_set(double eta);
int physics_eta_bulk(double * zeta);
int physics_eta_bulk_set(double zeta);
int physics_kt(double * kt);
int physics_kt_set(double kt);

int physics_b0(double b0[3]);
int physics_b0_set(double b0[3]);
int physics_e0(double e0[3]);
int physics_e0_set(double e0[3]);
int physics_fbody(double f[3]);
int physics_fbody_set(double f[3]);

int physics_mobility(double * mobility);
int physics_mobility_set(double mobility);
int physics_lc_gamma_rot(double * gamma);
int physics_lc_gamma_rot_set(double gamma);

#endif
