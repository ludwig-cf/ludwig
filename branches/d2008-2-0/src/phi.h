/*****************************************************************************
 *
 *  phi.h
 *
 *  $Id: phi.h,v 1.2 2008-08-24 16:58:10 kevin Exp $
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

extern struct io_info_t * io_info_phi;
#endif
