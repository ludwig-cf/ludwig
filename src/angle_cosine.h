/*****************************************************************************
 *
 *  angle_cosine.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_ANGLE_COSINE_H
#define LUDWIG_ANGLE_COSINE_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct angle_cosine_s angle_cosine_t;

int angle_cosine_create(pe_t * pe, cs_t * cs, angle_cosine_t ** pobj);
int angle_cosine_free(angle_cosine_t * obj);
int angle_cosine_info(angle_cosine_t * obj);
int angle_cosine_param_set(angle_cosine_t * obj, double kappa);
int angle_cosine_register(angle_cosine_t * obj, interact_t * parent);
int angle_cosine_compute(colloids_info_t * cinfo, void * self);
int angle_cosine_stats(void * self, double * stats);

#endif
