/*****************************************************************************
 *
 *  wall_ss_cut.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2022 The University of Edinburgh
 *  Juho Lintuvuori (juho.lintuvuori@u-psud.fr)
 *
 *****************************************************************************/

#ifndef LUDWIG_WALL_SS_CUT_H
#define LUDWIG_WALL_SS_CUT_H

#include "pe.h"
#include "coords.h"
#include "wall.h"
#include "colloids.h"
#include "interaction.h"

typedef struct wall_ss_cut_options_s wall_ss_cut_options_t;
typedef struct wall_ss_cut_s wall_ss_cut_t;

struct wall_ss_cut_options_s {
  double epsilon;       /* Energy scale */
  double sigma;         /* Sigma (length) */
  double nu;            /* exponent */
  double hc;            /* Surface-surface cut off */
};

int wall_ss_cut_create(pe_t * pe, cs_t * cs, wall_t * wall,
		       const wall_ss_cut_options_t * opts,
		       wall_ss_cut_t ** pobj);
int wall_ss_cut_free(wall_ss_cut_t * obj);
int wall_ss_cut_info(wall_ss_cut_t * obj);
int wall_ss_cut_register(wall_ss_cut_t * obj, interact_t * parent);
int wall_ss_cut_compute(colloids_info_t * cinfo, void * self);
int wall_ss_cut_stats(void * self, double * stats);
int wall_ss_cut_single(wall_ss_cut_t * obj, double h, double * f, double * v);

#endif
