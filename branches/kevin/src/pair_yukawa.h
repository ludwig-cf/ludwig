/*****************************************************************************
 *
 *  pair_yukawa.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2014)
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PAIR_YUKAWA_H
#define PAIR_YUKAWA_H

typedef struct pair_yukawa_s pair_yukawa_t;

#include "coords.h"
#include "colloids.h"
#include "interaction.h"

int pair_yukawa_create(coords_t * cs, pair_yukawa_t ** pobj);
int pair_yukawa_free(pair_yukawa_t * obj);
int pair_yukawa_info(pair_yukawa_t * obj);
int pair_yukawa_param_set(pair_yukawa_t * obj, double epsilon, double kappa,
                          double rc);
int pair_yukawa_register(pair_yukawa_t * obj, interact_t * parent);
int pair_yukawa_compute(colloids_info_t * cinfo, void * self);
int pair_yukawa_stats(void * self, double * stat);
int pair_yukawa_single(pair_yukawa_t * obj, double r, double * v, double * f);

#endif
