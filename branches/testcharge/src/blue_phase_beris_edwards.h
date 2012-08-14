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

int blue_phase_beris_edwards(field_t * fq, hydro_t * hydro);
void   blue_phase_be_set_rotational_diffusion(double);
double blue_phase_be_get_rotational_diffusion(void);

#endif
