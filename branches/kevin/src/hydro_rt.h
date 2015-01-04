/*****************************************************************************
 *
 *  hydro_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015 The University of Edinbrugh
 *
 *****************************************************************************/

#ifndef HYDRO_RT_H
#define HYDRO_RT_H

#include "coords.h"
#include "leesedwards.h"
#include "runtime.h"
#include "hydro.h"

int hydro_rt(rt_t * rt, coords_t * cs, le_t * le, hydro_t ** phydro);

#endif
