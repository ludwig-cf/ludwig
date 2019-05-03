/****************************************************************************
 *
 *  surfactant_rt.c
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
#include "surfactant.h"
#include "surfactant_rt.h"

/****************************************************************************
 *
 *  fe_surfactant1_param_rt
 *
 ****************************************************************************/

__host__ int fe_surf1_param_rt(pe_t * pe, rt_t * rt, fe_surf1_param_t * p) {

  double sigma;
  double xi0;
  double psi_c;

  /* Parameters */

  rt_double_parameter(rt, "surf_A", &p->a);
  rt_double_parameter(rt, "surf_B", &p->b);
  rt_double_parameter(rt, "surf_K", &p->kappa);

  rt_double_parameter(rt, "surf_kt", &p->kt);
  rt_double_parameter(rt, "surf_epsilon", &p->epsilon);
  rt_double_parameter(rt, "surf_beta", &p->beta);
  rt_double_parameter(rt, "surf_w", &p->w);

  /* For the surfactant should have... */

  assert(p->kappa > 0.0);
  assert(p->kt > 0.0);
  assert(p->epsilon > 0.0);
  assert(p->beta >= 0.0);
  assert(p->w >= 0.0);

  return 0;
}
