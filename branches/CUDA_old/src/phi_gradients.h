/*****************************************************************************
 *
 *  phi_gradients.h
 *
 *  $Id: phi_gradients.h,v 1.4 2010-10-15 12:40:03 kevin Exp $
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

void   phi_gradients_init(void);
void   phi_gradients_finish(void);
void   phi_gradients_compute(void);

void   phi_gradients_level_set(const int level);
void   phi_gradients_dyadic_set(const int level);

void   phi_gradients_grad(const int index, double grad[3]);
void   phi_gradients_grad_n(const int index, const int n, double grad[3]);
double phi_gradients_delsq(const int index);
double phi_gradients_delsq_n(const int index, const int n);
void   phi_gradients_grad_delsq(const int index, double grad[3]);
double phi_gradients_delsq_delsq(const int index);

void   phi_gradients_vector_delsq(const int index, double dq[3]);
void   phi_gradients_vector_gradient(const int index, double dq[3][3]);
void   phi_gradients_grad_dyadic(const int index, double dqq[3][3][3]);
void   phi_gradients_delsq_dyadic(const int index, double delsq[3][3]);

void   phi_gradients_tensor_delsq(const int, double dsq[3][3]);
void   phi_gradients_tensor_gradient(const int, double dq[3][3][3]);

#endif
