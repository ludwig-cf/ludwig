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

void blue_phase_O8M_init(void);
void blue_phase_O2_init(void);
void blue_phase_O5_init(void);
void blue_phase_H2D_init(void);
void blue_phase_H3DA_init(void);
void blue_phase_H3DB_init(void);
void blue_phase_DTC_init(void);
void blue_phase_BPIII_init(const double specs[3]);

void blue_phase_nematic_init(const double n[3]);
void blue_phase_twist_init(const int helical_axis);
void blue_phase_chi_edge(int N, double z0, double x0);
void blue_set_random_q_init(void);
void blue_set_random_q_rectangle_init(const double xmin, const double xmax,
				      const double ymin,const double ymax,
				      const double zmin,const double zmax);
void blue_phase_M_rot(double M[3][3], int dim, double alpha);
void blue_phase_init_amplitude_set(const double amplitude);
double blue_phase_init_amplitude(void);

#endif
