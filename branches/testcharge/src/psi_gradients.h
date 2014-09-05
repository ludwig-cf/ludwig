/****************************************************************************
 *
 *  psi_gradients.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *  (c) 2014 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef  PSI_GRADIENTS_H
#define  PSI_GRADIENTS_H

#ifdef NP_D3Q6
#define PSI_NGRAD 7
#endif

#ifdef NP_D3Q18
#define PSI_NGRAD 19
#endif

#ifdef NP_D3Q26
#define PSI_NGRAD 27
#endif

extern const int    psi_gr_cv[PSI_NGRAD][3];
extern const double psi_gr_wv[PSI_NGRAD];
extern const double psi_gr_rnorm[PSI_NGRAD];
extern const double psi_gr_rcs2;

#endif                               
