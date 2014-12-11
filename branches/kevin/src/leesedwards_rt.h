/*****************************************************************************
 *
 *  leesedwards_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2015 The University of Edinburgh
 *  Contributing authors:
 *    Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LEESEDWARDS_RT_H
#define LEESEDWARDS_RT_H

#include "runtime.h"
#include "coords.h"
#include "leesedwards.h"

int le_rt(rt_t * rt, coords_t * cs, le_t ** le);

#endif
