/*****************************************************************************
 *
 *  field_psi_init_rt.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_PSI_INIT_RT_H
#define LUDWIG_FIELD_PSI_INIT_RT_H

#include "pe.h"
#include "runtime.h"
/* #include "field_psi_init.h" */
#include "field.h"

typedef struct field_psi_info_s field_psi_info_t;

struct field_psi_info_s {
  double xi0;      /* An equilibrium interfacial width */
  double psi0;     /* A mean value */
};

int field_psi_init_rt(pe_t * pe, rt_t * rt, field_psi_info_t param,
		      field_t * phi);

#endif
