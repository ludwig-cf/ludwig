/*****************************************************************************
 *
 *  pair_ss_cut_ij.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kai Qi (kai.qi@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_PAIR_SS_CUT_IJ_H
#define LUDWIG_PAIR_SS_CUT_IJ_H

#include "pe.h"
#include "coords.h"
#include "colloids.h"
#include "interaction.h"

struct pair_ss_cut_ij_s {
  pe_t * pe;             /* Parallel environemnt */
  cs_t * cs;             /* Coordinate system */
  int ntypes;            /* Number of pair types */
  double ** epsilon;     /* epsilon (energy) types [i][j] */
  double ** sigma;       /* sigma (length) for types [i][j] */
  double ** nu;          /* exponent for types [i][j] */
  double ** hc;          /* cut-off for types [i][j] */
  double vlocal;         /* local contribution to energy */
  double hminlocal;      /* local nearest separation */
  double rminlocal;      /* local min centre-centre separation */
};

typedef struct pair_ss_cut_ij_s pair_ss_cut_ij_t;

int pair_ss_cut_ij_create(pe_t * pe, cs_t * cs, int nt, double * epsilon,
			  double * sigma, double * nu, double * hc,
			  pair_ss_cut_ij_t ** pobj);
int pair_ss_cut_ij_free(pair_ss_cut_ij_t * obj);
int pair_ss_cut_ij_info(pair_ss_cut_ij_t * obj);
int pair_ss_cut_ij_register(pair_ss_cut_ij_t * obj, interact_t * parent);
int pair_ss_cut_ij_compute(colloids_info_t * cinfo, void * self);
int pair_ss_cut_ij_stats(void * self, double * stats);
int pair_ss_cut_ij_single(pair_ss_cut_ij_t * obj, int i, int j, double h,
			  double * f, double * v);
int pair_ss_cut_ij_param_set(pair_ss_cut_ij_t * obj, double * epsilon,
			     double * sigma, double * nu, double * hc);

#endif
