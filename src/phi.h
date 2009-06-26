/*****************************************************************************
 *
 *  phi.h
 *
 *  $Id: phi.h,v 1.4 2009-06-26 08:42:25 kevin Exp $
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
void phi_compute_phi_site(void);
 
double phi_get_phi_site(const int);
void   phi_set_phi_site(const int, const double);
double phi_get_delsq_phi_site(const int);
void   phi_set_delsq_phi_site(const int, const double);
void   phi_get_grad_phi_site(const int, double *);
void   phi_set_grad_phi_site(const int, const double *);
void   phi_get_grad_delsq_phi_site(const int, double *);
double phi_get_delsq_delsq_phi_site(const int);
void   phi_leesedwards_transformation(void);
int    phi_is_finite_difference(void);
void   phi_set_finite_difference(void);

double phi_op_get_phi_site(const int, const int);
double phi_op_get_delsq_phi_site(const int, const int);
void   phi_op_get_grad_phi_site(const int, const int, double *);
void   phi_op_set_phi_site(const int, const int, const double);

void   phi_set_q_tensor(const int, double q[3][3]);
void   phi_get_q_tensor(const int, double q[3][3]);
void   phi_get_q_gradient_tensor(const int, double dq[3][3][3]);
void   phi_get_q_delsq_tensor(const int, double dsq[3][3]);

extern struct io_info_t * io_info_phi;
extern const int nop_;
#endif
