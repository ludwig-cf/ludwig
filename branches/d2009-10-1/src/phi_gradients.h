/*****************************************************************************
 *
 *  phi_gradients.h
 *
 *  $Id: phi_gradients.h,v 1.3.6.1 2010-03-30 14:25:47 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Phyiscs Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2008 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef PHI_GRADIENTS_H
#define PHI_GRADIENTS_H

void phi_gradients_init(void);
void phi_gradients_finish(void);
void phi_gradients_compute(void);
void phi_gradients_grad_dyadic(const int index, double dqq[3][3][3]);
void phi_gradients_delsq_dyadic(const int index, double delsq[3][3]);

#endif
