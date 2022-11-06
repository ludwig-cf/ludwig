/*****************************************************************************
 *
 *  fe_force_method_rt.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing author:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef FE_LUDWIG_FE_FORCE_METHOD_RT
#define FE_LUDWIG_FE_FORCE_METHOD_RT

#include "runtime.h"
#include "fe_force_method.h"

int fe_force_method_rt(rt_t * rt, rt_enum_t lv, fe_force_method_enum_t * meth);
int fe_force_method_rt_messages(rt_t * rt, rt_enum_t lv);

#endif
