/*****************************************************************************
 *
 *  phi_lb_coupler.h
 *
 *  $Id: phi_lb_coupler.h,v 1.3 2009-10-23 16:26:27 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2009)
 *
 *****************************************************************************/

#ifndef PHI_LB_COUPLER_H
#define PHI_LB_COUPLER_H

void phi_compute_phi_site(void);
void phi_set_mean_phi(double phi0);
void phi_init_drop(double r, double xi0);
void phi_lb_init_drop(double r, double xi0);

#endif
