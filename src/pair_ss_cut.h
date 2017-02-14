/*****************************************************************************
 *
 *  pair_ss_cut.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2014)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PAIR_SS_CUT_H
#define PAIR_SS_CUT_H

#include "colloids.h"
#include "interaction.h"

typedef struct pair_ss_cut_s pair_ss_cut_t;

int pair_ss_cut_create(pair_ss_cut_t ** pobj);
void pair_ss_cut_free(pair_ss_cut_t * obj);
int pair_ss_cut_info(pair_ss_cut_t * obj);
int pair_ss_cut_register(pair_ss_cut_t * obj, interact_t * parent);
int pair_ss_cut_compute(colloids_info_t * cinfo, void * self);
int pair_ss_cut_stats(void * self, double * stats);
int pair_ss_cut_single(pair_ss_cut_t * obj, double h, double * f, double * v);
int pair_ss_cut_param_set(pair_ss_cut_t * obj, double epsilon, double sigma,
			  int nu, double hc);

#endif
