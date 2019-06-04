/****************************************************************************
 *
 *  fe_ternary.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_FE_TERNARY_H
#define LUDWIG_FE_TERNARY_H

#include "memory.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"

typedef struct fe_ternary_s fe_ternary_t;
typedef struct fe_ternary_param_s fe_ternary_param_t;

struct fe_ternary_param_s {
    double alpha;              /* interface width  alpha  */
    double kappa1;         /* Ternary kappa */
    double kappa2;
    double kappa3;
};


__host__ int fe_ternary_create(pe_t * pe, cs_t * cs, field_t * phi,
                             field_grad_t * dphi, fe_ternary_param_t param,
                             fe_ternary_t ** fe);
__host__ int fe_ternary_free(fe_ternary_t * fe);
__host__ int fe_ternary_info(fe_ternary_t * fe);
__host__ int fe_ternary_param_set(fe_ternary_t * fe, fe_ternary_param_t vals);
__host__ int fe_ternary_sigma(fe_ternary_t * fe, double * sigma1);
__host__ int fe_ternary_xi1(fe_ternary_t * fe,  double * xi1);
__host__ int fe_ternary_target(fe_ternary_t * fe, fe_t ** target);

__host__ int fe_ternary_param(fe_ternary_t * fe, fe_ternary_param_t * param);
__host__ int fe_ternary_fed(fe_ternary_t * fe, int index, double * fed);
__host__ int fe_ternary_mu(fe_ternary_t * fe, int index, double * mu);
__host__ int fe_ternary_str(fe_ternary_t * fe, int index, double s[3][3]);
__host__ int fe_ternary_str_v(fe_ternary_t * fe, int index, double s[3][3][NSIMDVL]);


#endif
