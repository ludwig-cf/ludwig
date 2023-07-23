/*****************************************************************************
 *
 *  stencil_d3q19.h
 *
 *  A 19 point finite deifferrence stencil following lb_d3q19.h.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STENCIL_D3Q19_H
#define LUDWIG_STENCIL_D3Q19_H

#include "lb_d3q19.h"
#include "stencil.h"

int stencil_d3q19_create(stencil_t ** s);

#endif
