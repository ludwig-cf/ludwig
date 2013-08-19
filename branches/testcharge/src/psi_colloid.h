/*****************************************************************************
 *
 *  psi_colloid.h
 *
 *****************************************************************************/

#ifndef PSI_COLLOID_H
#define PSI_COLLOID_H

#include "psi.h"
#include "colloids.h"

int psi_colloid_rho_set(psi_t * obj);
int psi_colloid_electroneutral(psi_t * obj);
int psi_colloid_zetapotential(psi_t * obj, double * psi_zeta);
int psi_colloid_remove_charge(psi_t * psi, colloid_t * colloid, int index);
int psi_colloid_replace_charge(psi_t * psi, colloid_t * colloid, int index);

#endif
