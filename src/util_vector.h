/*****************************************************************************
 *
 *  util_vector.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_VECTOR_H
#define LUDWIG_UTIL_VECTOR_H

double util_vector_l2_norm(int n, const double * a);
void   util_vector_normalise(int n, double * a);
void   util_vector_copy(int n, const double * a, double * b);

#endif
