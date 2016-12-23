/*****************************************************************************
 *
 *  blue_phase_init.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (o.henrich@ucl.ac.uk) wrote most of these.
 *
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_INIT_H
#define BLUE_PHASE_INIT_H

#include "blue_phase.h"

int blue_phase_twist_init(fe_lc_param_t * param, field_t * fq, int helical_axis);
int blue_phase_O8M_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_O2_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_O5_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_H2D_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_H3DA_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_H3DB_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_DTC_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_BPIII_init(fe_lc_param_t * param, field_t * fq, const double specs[3]);
int blue_phase_nematic_init(fe_lc_param_t * param, field_t * fq, const double n[3]);
int blue_phase_active_nematic_init(fe_lc_param_t * param, field_t * fq, const double n[3]);
int lc_active_nematic_init_q2d(fe_lc_param_t * param, field_t * q, int istrip);
int blue_phase_chi_edge(fe_lc_param_t * param, field_t * fq, int n, double z0, double x0);
int blue_phase_random_q_init(fe_lc_param_t * param, field_t * fq);
int blue_phase_random_q_rectangle(fe_lc_param_t * param, field_t * q, int rmin[3], int rmax[3]);
int blue_phase_cf1_init(fe_lc_param_t * param, field_t * fq, int axis);
int blue_phase_random_cf1_init(fe_lc_param_t * param, field_t * fq, int axis);

#endif
