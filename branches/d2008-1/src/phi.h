/*****************************************************************************
 *
 *  phi.h
 *
 *  $Id: phi.h,v 1.1.2.1 2008-02-12 17:15:47 kevin Exp $
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
void phi_set_mean_phi(double);

#endif
