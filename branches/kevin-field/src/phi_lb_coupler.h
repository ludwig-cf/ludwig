/*****************************************************************************
 *
 *  phi_lb_coupler.h
 *
 *  $Id: phi_lb_coupler.h,v 1.4 2010-10-15 12:40:03 kevin Exp $
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
void phi_lb_coupler_phi_set(const int index, const double phi);

#ifdef OLD_PHI
#else
#include "field.h"
int phi_lb_to_field(field_t * phi);
int phi_lb_from_field(field_t * phi);
#endif

#endif
