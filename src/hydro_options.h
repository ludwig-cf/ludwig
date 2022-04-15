/*****************************************************************************
 *
 *  hydro_options.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#ifndef LUDWIG_HYDRO_OPTIONS_H
#define LUDWIG_HYDRO_OPTIONS_H

/* Possible halo schemes */

typedef enum hydro_halo_enum {
  HYDRO_U_HALO_HOST,                 /* older host version */
  HYDRO_U_HALO_TARGET,               /* host or target */
  HYDRO_U_HALO_OPENMP                /* Host-only OpenMP implementation */ 
} hydro_halo_enum_t;

typedef struct hydro_options_s hydro_options_t;

struct hydro_options_s {
  int nhcomm;                        /* Actual halo width */
  hydro_halo_enum_t haloscheme;      /* Halo exchange method */
};

hydro_options_t hydro_options_default(void);
hydro_options_t hydro_options_nhalo(int nhalo);

#endif
