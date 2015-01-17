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

struct fe_s {
  void * child;             /* Child actual free energy pointer (void *) */
  fe_fed_ft fed;            /* Free energy density function */
  fe_mu_ft  mu;             /* Chemical potential function */
  fe_str_ft stress;         /* Stress function */
  fe_mu_solv_ft mu_solv;    /* Solvation chemical potential function */
  fe_hvector_ft hvector;    /* Vector molecular field */
  fe_htensor_ft htensor;    /* Tensor molecular field */
};

#endif
