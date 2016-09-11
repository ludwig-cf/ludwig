/*****************************************************************************
 *
 *  phi_cahn_hilliard.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  $Id: phi_cahn_hilliard.h,v 1.5 2010-10-15 12:40:03 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_CAHN_HILLIARD_H
#define PHI_CAHN_HILLIARD_H

#include "pe.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "field.h"
#include "hydro.h"
#include "map.h"
#include "noise.h"

typedef struct phi_ch_s phi_ch_t;
typedef struct phi_ch_info_s phi_ch_info_t;

__host__ int phi_ch_create(pe_t * pe, lees_edw_t * le, phi_ch_info_t * info,
			   phi_ch_t ** pch);
__host__ int phi_ch_free(phi_ch_t * pch);

__host__ int phi_cahn_hilliard(phi_ch_t * pch, fe_t * fe, field_t * phi,
			       hydro_t * hydro, map_t * map,
			       noise_t * noise);

#endif
