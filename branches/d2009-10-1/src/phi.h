/*****************************************************************************
 *
 *  phi.h
 *
 *  $Id: phi.h,v 1.6.4.6 2010-04-02 07:56:03 kevin Exp $
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

extern struct io_info_t * io_info_phi;

#endif
