/*****************************************************************************
 *
 *  psi_colloid.h
 *
 *****************************************************************************/

#ifndef PSI_COLLOID_H
#define PSI_COLLOID_H

#include "psi.h"

int psi_colloid_rho_set(psi_t * obj);
int psi_colloid_electroneutral(psi_t * obj);
int psi_colloid_zetapotential(psi_t * obj, double * psi_zeta);

#endif
