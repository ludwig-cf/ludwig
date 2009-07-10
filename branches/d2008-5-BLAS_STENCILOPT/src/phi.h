/*****************************************************************************
 *
 *  phi.h
 *
 *  $Id: phi.h,v 1.3.10.2 2009-07-10 09:01:58 cevi_parker Exp $
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

extern struct io_info_t * io_info_phi;
extern const int nop_;

/* tile depths */

extern int ti;
extern int tj; 

#endif
