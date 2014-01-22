/*****************************************************************************
 *
 *  psi_colloid.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2013 The University of Edinburgh
 *  Contributing Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PSI_COLLOID_H
#define PSI_COLLOID_H

#include "psi.h"
#include "colloids.h"

int psi_colloid_rho_set(psi_t * obj, colloids_info_t * cinfo);
int psi_colloid_electroneutral(psi_t * obj, colloids_info_t * cinfo);
int psi_colloid_remove_charge(psi_t * psi, colloids_info_t * cinfo,
			      colloid_t * colloid, int index);
int psi_colloid_replace_charge(psi_t * psi, colloids_info_t * cinfo,
			       colloid_t * colloid, int index);
int psi_colloid_zetapotential(psi_t * obj, colloids_info_t * cinfo,
			      double * psi_zeta);
#endif
