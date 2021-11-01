/*****************************************************************************
 *
 *  lb_openbc_options.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Comuting Centre
 *
 *  (c) 2021 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_LB_OPENBC_OPTIONS_H
#define LUDWIG_LB_OPENBC_OPTIONS_H

#include "pe.h"

/* TYPES */

enum lb_inflow_enum {LB_INFLOW_INVALID = 0, LB_INFLOW_NONE, LB_INFLOW_MAX};
enum lb_outflow_enum {LB_OUTFLOW_INVALID = 0, LB_OUTFLOW_NONE, LB_OUTFLOW_MAX};

typedef enum lb_inflow_enum        lb_inflow_enum_t;
typedef enum lb_outflow_enum       lb_outflow_enum_t;
typedef struct lb_openbc_options_s lb_openbc_options_t;

struct lb_openbc_options_s {
  int bctype;                  /* Combination of inflow/outflow? */
  int nvel;
  lb_inflow_enum_t  inflow;    /* Inflow boundary condition type */
  lb_outflow_enum_t outflow;   /* Outflow boundary condition type */
  int flow[3];                 /* Flow coordinate direction (exactly 1) */ 
};

__host__ lb_inflow_enum_t lb_openbc_options_inflow_default(void);
__host__ lb_outflow_enum_t lb_openbc_options_outflow_default(void);
__host__ lb_openbc_options_t lb_openbc_options_default(void);

__host__ int lb_openbc_options_valid(const lb_openbc_options_t * options);
__host__ int lb_openbc_options_inflow_valid(lb_inflow_enum_t inflow);
__host__ int lb_openbc_options_outflow_valid(lb_outflow_enum_t outflow);
__host__ int lb_openbc_options_flow_valid(const int flow[3]);

__host__ int lb_openbc_options_info(pe_t * pe, const lb_openbc_options_t *);

#endif
