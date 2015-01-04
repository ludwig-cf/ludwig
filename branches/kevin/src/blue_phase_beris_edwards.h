/*****************************************************************************
 *
 *  blue_phase_beris_edwards.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2015 The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_BERIS_EDWARDS_H
#define BLUE_PHASE_BERIS_EDWARDS_H

#include "leesedwards.h"
#include "hydro.h"
#include "field.h"
#include "map.h"
#include "noise.h"

int blue_phase_beris_edwards(le_t * le, field_t * fq, hydro_t * hydro,
			     map_t * map, noise_t * noise);
int blue_phase_be_tmatrix_set(double t[3][3][NQAB]);

#endif
