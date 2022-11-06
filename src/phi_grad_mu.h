/*****************************************************************************
 *
 *  phi_grad_mu.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PHI_GRAD_MU_H
#define LUDWIG_PHI_GRAD_MU_H

#include "coords.h"
#include "field.h"
#include "free_energy.h"
#include "hydro.h"
#include "map.h"

__host__ int phi_grad_mu_fluid(cs_t * cs, field_t * phi, fe_t * fe,
			       hydro_t * hydro);
__host__ int phi_grad_mu_solid(cs_t * cs, field_t * phi, fe_t * fe,
			       hydro_t * hydro, map_t * map);
__host__ int phi_grad_mu_external(cs_t * cs, field_t * phi, hydro_t * hydro);

__host__ int phi_grad_mu_correction(cs_t * cs, field_t * phi, fe_t * fe,
				    hydro_t * hydro, map_t * map, int opt);
#endif
