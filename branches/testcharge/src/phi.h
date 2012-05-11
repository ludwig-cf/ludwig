/*****************************************************************************
 *
 *  phi.h
 *
 *  $Id: phi.h,v 1.7 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *  (c) 2008 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef PHI_H
#define PHI_H

#include "io_harness.h"

void phi_init(void);
void phi_finish(void);
void phi_halo(void);
 
double phi_get_phi_site(const int);
void   phi_set_phi_site(const int, const double);
double phi_get_delsq_phi_site(const int);
void   phi_leesedwards_transformation(void);
int    phi_is_finite_difference(void);
void   phi_set_finite_difference(void);
int    phi_nop(void);
void   phi_nop_set(const int n);

double phi_op_get_phi_site(const int, const int);
void phi_op_set_phi_site(const int index, const int nop, const double value);

void   phi_set_q_tensor(const int, double q[3][3]);
void   phi_get_q_tensor(const int, double q[3][3]);

void   phi_vector_set(const int index, const double q[3]);
void   phi_vector(const int index, double q[3]);

int phi_init_io_info(int grid[3], int form_in, int form_out);
int phi_io_info(io_info_t ** info);

#endif
