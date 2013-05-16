/*****************************************************************************
 *
 *  phi_fluctuations.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_FLUCTUATIONS_H
#define PHI_FLUCTUATIONS_H

int  phi_fluctuations_on(void);
void phi_fluctuations_on_set(int flag);
void phi_fluctuations_init(unsigned int master_seed);
void phi_fluctuations_finalise(void);
void phi_fluctuations_site(int n, double var, double * jsite);
int  phi_fluctuations_qab(int index, double var, double chi[5]);

#endif
