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

#include "targetDP.h"

typedef struct physics_s physics_t;

HOST int physics_ref(physics_t ** ref);
HOST int physics_free(void);

HOST int physics_rho0(double * rho);
HOST int physics_rho0_set(double rho0);
HOST int physics_phi0(double * phi0);
HOST int physics_phi0_set(double phi0);

HOST int physics_eta_shear(double * eta);
HOST int physics_eta_shear_set(double eta);
HOST int physics_eta_bulk(double * zeta);
HOST int physics_eta_bulk_set(double zeta);
HOST int physics_kt(double * kt);
HOST int physics_kt_set(double kt);

HOST int physics_b0(double b0[3]);
HOST int physics_b0_set(double b0[3]);
HOST int physics_e0(double e0[3]);
HOST int physics_e0_set(double e0[3]);
HOST int physics_e0_frequency(double * e0_frequency);
HOST int physics_e0_frequency_set(double e0_frequency);
HOST int physics_fbody(double f[3]);
HOST int physics_fbody_set(double f[3]);
HOST int physics_fgrav(double g[3]);
HOST int physics_fgrav_set(double g[3]);

HOST int physics_mobility(double * mobility);
HOST int physics_mobility_set(double mobility);
HOST int physics_lc_gamma_rot(double * gamma);
HOST int physics_lc_gamma_rot_set(double gamma);

#endif
