/*****************************************************************************
 *
 *  phi_stats.h
 *
 *  $Id: phi_stats.h,v 1.3 2008-10-21 17:19:18 kevin Exp $
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
void phi_init_block(void);
void phi_init_bath(void);
#endif
