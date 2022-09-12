/*****************************************************************************
 *
 *  lb_data_options.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_DATA_OPTIONS_H
#define LUDWIG_LB_DATA_OPTIONS_H

#include "io_info_args.h"

typedef enum lb_relaxation_enum {LB_RELAXATION_M10,
                                 LB_RELAXATION_BGK,
				 LB_RELAXATION_TRT}
  lb_relaxation_enum_t;

typedef enum lb_halo_enum {LB_HALO_TARGET,
                           LB_HALO_OPENMP_FULL,
                           LB_HALO_OPENMP_REDUCED} lb_halo_enum_t;

typedef struct lb_data_options_s lb_data_options_t;

struct lb_data_options_s {
  int ndim;
  int nvel;
  int ndist;
  lb_relaxation_enum_t nrelax;
  lb_halo_enum_t halo;
  int reportimbalance;
  int usefirsttouch;

  io_info_args_t data;
  io_info_args_t rho;
};

lb_data_options_t lb_data_options_default(void);
int lb_data_options_valid(const lb_data_options_t * opts);

#endif
