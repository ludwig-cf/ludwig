/*****************************************************************************
 *
 *  fe_s.h
 *
 *  Implemenation.
 *
 *****************************************************************************/

#ifndef FREE_ENERGY_S_H
#define FREE_ENERGY_S_H

#include "fe.h"

/* These type signatures are mainly for convenience. */

typedef int (* fe_free_ft)(fe_t * fe);
typedef int (* fe_fed_ft)(fe_t * fe, int index, double * fed);
typedef int (* fe_mu_ft)(fe_t * fe, int index, double * mu);
typedef int (* fe_str_ft)(fe_t * fe, int index, double s[3][3]);
typedef int (* fe_mu_solv_ft)(fe_t * fe, int index, int n, double * mu);
typedef int (* fe_hvector_ft)(fe_t * fe, int index, double h[3]);
typedef int (* fe_htensor_ft)(fe_t * fe, int index, double h[3][3]);

/* Virtual function table. */

typedef struct fe_vtable_s fe_vtable_t;

struct fe_vtable_s {
  /* Order is important: actual vtables must be the same */
  fe_free_ft free;          /* Destructor */
  fe_fed_ft fed;            /* Free energy density function */
  fe_mu_ft  mu;             /* Chemical potential function */
  fe_str_ft stress;         /* Stress function */
  fe_mu_solv_ft mu_solv;    /* Solvation chemical potential function */
  fe_hvector_ft hvector;    /* Vector molecular field */
  fe_htensor_ft htensor;    /* Tensor molecular field */
};

struct fe_s {
  fe_vtable_t * vtable;
};

#endif
