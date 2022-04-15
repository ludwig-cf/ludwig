/*****************************************************************************
 *
 *  field_options.h
 *
 *  Container.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contrbuting authors:
 *  Kevin Stratford
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_OPTIONS_H
#define LUDWIG_FIELD_OPTIONS_H

#include "io_info_args.h"

typedef enum field_halo_enum {FIELD_HALO_HOST,
                              FIELD_HALO_TARGET,
			      FIELD_HALO_OPENMP} field_halo_enum_t;

typedef struct field_options_s field_options_t;

struct field_options_s {
  int ndata;                            /* Number of field components */
  int nhcomm;                           /* Actual halo width required */

  field_halo_enum_t haloscheme;         /* Halo swap method */
  int haloverbose;                      /* Halo information level */

  io_info_args_t iodata;                /* I/O information */
};

field_options_t field_options_default(void);
field_options_t field_options_ndata_nhalo(int ndata, int nhalo);
int field_options_valid(const field_options_t * opts);

#endif
