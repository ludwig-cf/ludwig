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
 *  (c) 2013-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHYSICS_H
#define PHYSICS_H

#include "targetDP.h"

typedef struct physics_s physics_t;

__targetHost__ int physics_ref(physics_t ** ref);
__targetHost__ int physics_free(void);

__targetHost__ int physics_rho0(double * rho);
__targetHost__ int physics_rho0_set(double rho0);
__targetHost__ int physics_phi0(double * phi0);
__targetHost__ int physics_phi0_set(double phi0);

__targetHost__ int physics_eta_shear(double * eta);
__targetHost__ int physics_eta_shear_set(double eta);
__targetHost__ int physics_eta_bulk(double * zeta);
__targetHost__ int physics_eta_bulk_set(double zeta);
__targetHost__ int physics_kt(double * kt);
__targetHost__ int physics_kt_set(double kt);

__targetHost__ int physics_b0(double b0[3]);
__targetHost__ int physics_b0_set(double b0[3]);
__targetHost__ int physics_e0(double e0[3]);
__targetHost__ int physics_e0_set(double e0[3]);
__targetHost__ int physics_e0_frequency(double * e0_frequency);
__targetHost__ int physics_e0_frequency_set(double e0_frequency);
__targetHost__ int is_physics_e0();
__targetHost__ int physics_fbody(double f[3]);
__targetHost__ int physics_fbody_set(double f[3]);
__targetHost__ int physics_fgrav(double g[3]);
__targetHost__ int physics_fgrav_set(double g[3]);

__targetHost__ int physics_mobility(double * mobility);
__targetHost__ int physics_mobility_set(double mobility);

#endif
