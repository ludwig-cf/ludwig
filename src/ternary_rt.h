/****************************************************************************
 *
 *  fe_ternary_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef TERNARY_RT_H1
#ifndef TERNARY_RT_H2
#ifndef TERNARY_RT_H3
#define TERNARY_RT_H1
#define TERNARY_RT_H2
#define TERNARY_RT_H3

#include "pe.h"
#include "runtime.h"
#include "ternary.h"

__host__ int fe_ternary_param_rt(pe_t * pe, rt_t * rt, fe_ternary_param_t * p);

#endif
