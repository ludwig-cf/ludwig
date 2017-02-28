/*****************************************************************************
 *
 *  pair_ss_cut.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PAIR_SS_CUT_H
#define LUDWIG_PAIR_SS_CUT_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

typedef struct pair_ss_cut_s pair_ss_cut_t;

int pair_ss_cut_create(pe_t * pe, cs_t * cs, pair_ss_cut_t ** pobj);
int pair_ss_cut_free(pair_ss_cut_t * obj);
int pair_ss_cut_info(pair_ss_cut_t * obj);
int pair_ss_cut_register(pair_ss_cut_t * obj, interact_t * parent);
int pair_ss_cut_compute(colloids_info_t * cinfo, void * self);
int pair_ss_cut_stats(void * self, double * stats);
int pair_ss_cut_single(pair_ss_cut_t * obj, double h, double * f, double * v);
int pair_ss_cut_param_set(pair_ss_cut_t * obj, double epsilon, double sigma,
			  int nu, double hc);

#endif
