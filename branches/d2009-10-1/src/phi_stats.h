/*****************************************************************************
 *
 *  phi_stats.h
 *
 *  $Id: phi_stats.h,v 1.4.14.1 2010-05-12 18:19:40 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) The University of Edinburgh (2008)
 *
 *****************************************************************************/

#ifndef PHI_STATS_
#define PHI_STATS_

void phi_set_mean_phi(double);
void phi_stats_print_stats(void);
void phi_init_block(const double xi0);
void phi_init_bath(void);
void phi_init_surfactant(double);
#endif
