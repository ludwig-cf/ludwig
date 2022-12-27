/*****************************************************************************
 *
 *  leslie_ericksen.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LESLIE_ERICKSEN_H
#define LUDWIG_LESLIE_ERICKSEN_H

#include "pe.h"
#include "coords.h"
#include "field.h"
#include "hydro.h"
#include "polar_active.h"

typedef struct leslie_param_s leslie_param_t;
typedef struct leslie_ericksen_s leslie_ericksen_t;

struct leslie_param_s {
  double Gamma;                  /* Rotational diffusion constant */
  double swim;                   /* Self-advection parameter */
  double lambda;                 /* Flow aligning/flow tumbling parameter */
};

struct leslie_ericksen_s {
  pe_t * pe;                     /* Parallel environment */
  cs_t * cs;                     /* Coordinate system */
  fe_polar_t * fe;               /* Free energy */
  field_t * p;                   /* Vector order parameter field */
  leslie_param_t param;          /* Parameters */
};

int leslie_ericksen_create(pe_t * pe, cs_t * cs, fe_polar_t * fe, field_t * p,
			   const leslie_param_t * param,
			   leslie_ericksen_t ** obj);
int leslie_ericksen_free(leslie_ericksen_t ** obj);

int leslie_ericksen_update(leslie_ericksen_t * obj, hydro_t * hydro);
int leslie_ericksen_self_advection(leslie_ericksen_t * obj, hydro_t * hydro);

#endif
