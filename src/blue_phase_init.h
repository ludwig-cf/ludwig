/*****************************************************************************
 *
 *  blue_phase_init.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Oliver Henrich (o.henrich@ucl.ac.uk) wrote most of these.
 *
 *****************************************************************************/

#ifndef LUDWIG_BLUE_PHASE_INIT_H
#define LUDWIG_BLUE_PHASE_INIT_H

#include "coords.h"
#include "blue_phase.h"

int blue_phase_twist_init(cs_t * cs, fe_lc_param_t * param, field_t * fq, int helical_axis);
int blue_phase_O8M_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_O2_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_O5_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_H2D_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_H3DA_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_H3DB_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_DTC_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_BPIII_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
			  const double specs[3]);
int blue_phase_nematic_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
			    const double n[3]);
int blue_phase_active_nematic_init(cs_t * cs, fe_lc_param_t * param,
				   field_t * fq, const double n[3]);
int lc_active_nematic_init_q2d(cs_t * cs, fe_lc_param_t * param, field_t * q,
			       int istrip);
int blue_phase_chi_edge(cs_t * cs, fe_lc_param_t * param, field_t * fq, int n,
			double z0, double x0);
int blue_phase_random_q_init(cs_t * cs, fe_lc_param_t * param, field_t * fq);
int blue_phase_random_q_rectangle(cs_t * cs, fe_lc_param_t * param,
				  field_t * q, int rmin[3], int rmax[3]);
int blue_phase_cf1_init(cs_t * cs, fe_lc_param_t * param, field_t * fq, int axis);
int blue_phase_random_cf1_init(cs_t * cs, fe_lc_param_t * param, field_t * fq, int axis);

#endif
