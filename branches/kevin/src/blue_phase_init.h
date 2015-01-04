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
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef BLUE_PHASE_INIT_H
#define BLUE_PHASE_INIT_H

#include "coords.h"
#include "field.h"

int blue_phase_twist_init(coords_t * cs, field_t * fq, int helical_axis);
int blue_phase_O8M_init(coords_t * cs, field_t * fq);
int blue_phase_O2_init(coords_t * cs, field_t * fq);
int blue_phase_O5_init(coords_t * cs, field_t * fq);
int blue_phase_H2D_init(coords_t * cs, field_t * fq);
int blue_phase_H3DA_init(coords_t * cs, field_t * fq);
int blue_phase_H3DB_init(coords_t * cs, field_t * fq);
int blue_phase_DTC_init(coords_t * cs, field_t * fq);
int blue_phase_BPIII_init(coords_t * cs, field_t * fq, const double specs[3]);
int blue_phase_nematic_init(coords_t * cs, field_t * fq, const double n[3]);
int blue_phase_active_nematic_init(coords_t * cs, field_t * fq,
				   const double n[3]);
int blue_phase_chi_edge(coords_t * cs, field_t * fq, int N, double z0,
			double x0);
int blue_phase_random_q_init(coords_t * cs, field_t * fq);
int blue_phase_random_q_rectangle(coords_t * cs, field_t * q, int rmin[3],
				  int rmax[3]);
int blue_phase_cf1_init(coords_t * cs, field_t * fq, const int axis);
int blue_phase_random_cf1_init(coords_t * cs, field_t * fq, const int axis);

void blue_phase_M_rot(double M[3][3], int dim, double alpha);
void blue_phase_init_amplitude_set(const double amplitude);
double blue_phase_init_amplitude(void);

#endif
