/*****************************************************************************
 *
 *  stencil_d3q27.h
 *
 *  A finite difference stencil following lb_d3q27.h.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_STENCIL_D3Q27_H
#define LUDWIG_STENCIL_D3Q27_H

#include "lb_d3q27.h"
#include "stencil.h"

int stencil_d3q27_create(stencil_t ** s);

#endif
