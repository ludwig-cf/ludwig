/*****************************************************************************
 *
 *  gradient.h
 *
 *  $Id: gradient.h,v 1.1.2.1 2010-03-30 14:09:27 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef GRADIENT_H
#define GRADIENT_H

void gradient_d2_set(void (* f)(const int nop,
				const double * field,
				double * grad,
				double * delsq));
void gradient_d4_set(void (* f)(const int nop,
				const double * field,
				double * grad,
				double * delsq));

void gradient_d2_dyadic_set(void (* f)(const int nop,
				       const double * field,
				       double * grad,
				       double * delsq));

void gradient_d2(const int nop, const double * field,
		 double * grad, double * delsq);
void gradient_d4(const int nop, const double * field,
		 double * grad, double * delsq);
void gradient_d2_dyadic(const int nop, const double * field,
			double * grad, double * delsq);

#endif
