/*****************************************************************************
 *
 *  pair_lj_cut.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2017 The University of Edinburgh
 *
 *  Contibuting Authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PAIR_LJ_CUT_H
#define LUDWIG_PAIR_LJ_CUT_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct pair_lj_cut_s pair_lj_cut_t;

int pair_lj_cut_create(pe_t * pe, cs_t * cs, pair_lj_cut_t ** pobj);
int pair_lj_cut_free(pair_lj_cut_t * obj);
int pair_lj_cut_param_set(pair_lj_cut_t * obj, double epsilon, double sigma,
			  double rc);
int pair_lj_cut_info(pair_lj_cut_t * obj);
int pair_lj_cut_register(pair_lj_cut_t * obj, interact_t * parent);
int pair_lj_cut_compute(colloids_info_t * cinfo, void * self);
int pair_lj_cut_stats(void * self, double * stats);
int pair_lj_cut_single(pair_lj_cut_t * obj, double r, double * f, double * v);

#endif
