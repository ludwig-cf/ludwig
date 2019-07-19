/****************************************************************************
 *
 *  ternary_rt.c
 *
 *  Run time initialisation for the surfactant free energy.
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

#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "ternary.h"
#include "ternary_rt.h"

/****************************************************************************
 *
 *  fe_surfactant1_param_rt
 *
 ****************************************************************************/

__host__ int fe_ternary_param_rt(pe_t * pe, rt_t * rt, fe_ternary_param_t * p) {
    
    double sigma1;
    double xi1;
    
    /* Parameters */
    
    rt_double_parameter(rt, "surf_kappa1", &p->kappa1);
    rt_double_parameter(rt, "surf_kappa2", &p->kappa2);
    rt_double_parameter(rt, "surf_kappa3", &p->kappa3);
    rt_double_parameter(rt, "surf_alpha", &p->alpha);
    
    /* For the surfactant should have... */
    
    assert(p->kappa1 >= 0.0);
    assert(p->kappa2 >= 0.0);
    assert(p->kappa3 >= 0.0);
    assert(p->alpha > 0.0);
    
    return 0;
}
