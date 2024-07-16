/*****************************************************************************
 *
 *  kernel.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016-2024 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_KERNEL_H
#define LUDWIG_KERNEL_H

#include "kernel_3d.h"
#include "kernel_3d_v.h"

int kernel_launch_param(int iterations, dim3 * nblk, dim3 * ntpb);

#endif
