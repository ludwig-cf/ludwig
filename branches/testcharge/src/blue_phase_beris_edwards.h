/*****************************************************************************
 *
 *  blue_phase_beris_edwards.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2009)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_BERIS_EDWARDS_H
#define BLUE_PHASE_BERIS_EDWARDS_H

#include "hydro.h"
#include "field.h"
#include "map.h"

int blue_phase_beris_edwards(field_t * fq, hydro_t * hydro, map_t * map);
void   blue_phase_be_set_rotational_diffusion(double);
double blue_phase_be_get_rotational_diffusion(void);
int blue_phase_be_tmatrix_set(double t[3][3][NQAB]);

#endif
