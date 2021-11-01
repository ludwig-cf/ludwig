/*****************************************************************************
 *
 *  lb_d3q19.h
 *
 *  D3Q19 model definition. See lb_d3q19.c for details.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_D3Q19_H
#define LUDWIG_LB_D3Q19_H

#include "lb_model.h"

/* Velocity set, weights, and normalisers. */

enum {NVEL_D3Q19 = 19};

#define LB_CV_D3Q19(cv) const int8_t cv[NVEL_D3Q19][3] = {       \
    { 0,  0,  0},                              \
    { 1,  1,  0}, { 1,  0,  1}, { 1,  0,  0},  \
    { 1,  0, -1}, { 1, -1,  0}, { 0,  1,  1},  \
    { 0,  1,  0}, { 0,  1, -1}, { 0,  0,  1},  \
    { 0,  0, -1}, { 0, -1,  1}, { 0, -1,  0},  \
    { 0, -1, -1}, {-1,  1,  0}, {-1,  0,  1},  \
    {-1,  0,  0}, {-1,  0, -1}, {-1, -1,  0}};

#define LB_WEIGHTS_D3Q19(wv) const double wv[NVEL_D3Q19] = {   12.0/36.0, \
    1.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 2.0/36.0, \
    1.0/36.0, 2.0/36.0, 2.0/36.0, 1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0, \
    1.0/36.0, 2.0/36.0, 1.0/36.0, 1.0/36.0}; 

#define LB_NORMALISERS_D3Q19(na) const double na[NVEL_D3Q19] = { \
    1.0, \
    3.0, 3.0, 3.0, \
    9.0/2.0, 9.0, 9.0, 9.0/2.0, 9.0, 9.0/2.0, \
    3.0/4.0, 3.0/2.0, 3.0/2.0, 3.0/2.0, \
    9.0/4.0, 9.0/2.0, 9.0/2.0, 9.0/2.0, \
    1.0/2.0};

int lb_d3q19_create(lb_model_t * model);

#endif
