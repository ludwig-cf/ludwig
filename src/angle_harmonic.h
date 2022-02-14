//CHANGE1
/*****************************************************************************
 *
 *  angle_harmonic.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Kai Qi (kai.qi@epfl.ch)
 *
 *****************************************************************************/

#ifndef LUDWIG_ANGLE_HARMONIC_H
#define LUDWIG_ANGLE_HARMONIC_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct angle_harmonic_s angle_harmonic_t;

int angle_harmonic_create(pe_t * pe, cs_t * cs, angle_harmonic_t ** pobj);
int angle_harmonic_free(angle_harmonic_t * obj);
int angle_harmonic_info(angle_harmonic_t * obj);
int angle_harmonic_param_set(angle_harmonic_t * obj, double kappa, double theta0);
int angle_harmonic_register(angle_harmonic_t * obj, interact_t * parent);
int angle_harmonic_compute(colloids_info_t * cinfo, void * self);
int angle_harmonic_stats(void * self, double * stats);

#endif
