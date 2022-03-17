/*****************************************************************************
 *
 *  field_temperature_init.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_TEMPERATURE_INIT_H
#define LUDWIG_FIELD_TEMPERATURE_INIT_H

#include "field.h"

typedef struct field_temperature_info_s field_temperature_info_t;

struct field_temperature_info_s {
  double T0;            /* should be set to zero in input file */
  double Tc;            
  double Tj1;
  double Tj2;
};

int field_temperature_init_uniform(field_t * temperature, double T0);
int field_temperature_init_drop(field_t * temperature, double xi, double radius, double phistar);
#endif
