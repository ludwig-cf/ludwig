/*****************************************************************************
 *
 *  angle_cosine.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef ANGLE_COSINE_H
#define ANGLE_COSINE_H

#include "colloids.h"
#include "interaction.h"

typedef struct angle_cosine_s angle_cosine_t;

int angle_cosine_create(angle_cosine_t ** pobj);
void angle_cosine_free(angle_cosine_t * obj);
int angle_cosine_info(angle_cosine_t * obj);
int angle_cosine_param_set(angle_cosine_t * obj, double kappa);
int angle_cosine_register(angle_cosine_t * obj, interact_t * parent);
int angle_cosine_compute(colloids_info_t * cinfo, void * self);
int angle_cosine_stats(void * self, double * stats);

#endif
