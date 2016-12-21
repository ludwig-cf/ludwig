/*****************************************************************************
 *
 *  pair_lj_cut.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2014)
 *  Contibuting Authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PAIR_LJ_CUT_H
#define PAIR_LJ_CUT_H

#include "colloids.h"
#include "interaction.h"

typedef struct pair_lj_cut_s pair_lj_cut_t;

int pair_lj_cut_create(pair_lj_cut_t ** pobj);
void pair_lj_cut_free(pair_lj_cut_t * obj);
int pair_lj_cut_param_set(pair_lj_cut_t * obj, double epsilon, double sigma,
			  double rc);
int pair_lj_cut_info(pair_lj_cut_t * obj);
int pair_lj_cut_register(pair_lj_cut_t * obj, interact_t * parent);
int pair_lj_cut_compute(colloids_info_t * cinfo, void * self);
int pair_lj_cut_stats(void * self, double * stats);
int pair_lj_cut_single(pair_lj_cut_t * obj, double r, double * f, double * v);

#endif
