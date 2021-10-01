/****************************************************************************
 *
 *  fe_ternary_rt.c
 *
 *  Run time initialisation for the surfactant free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Shan Chen (shan.chen@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "fe_ternary.h"
#include "fe_ternary_rt.h"
#include "field_s.h"
#include "field_ternary_init.h"

/****************************************************************************
 *
 *  fe_ternary_param_rt
 *
 ****************************************************************************/

__host__ int fe_ternary_param_rt(pe_t * pe, rt_t * rt,
				 fe_ternary_param_t * p) {
  assert(pe);
  assert(rt);
  assert(p);

  /* Parameters */

  rt_key_required(rt, "ternary_kappa1", RT_FATAL);
  rt_key_required(rt, "ternary_kappa2", RT_FATAL);
  rt_key_required(rt, "ternary_kappa3", RT_FATAL);
  rt_key_required(rt, "ternary_alpha",  RT_FATAL);

  rt_double_parameter(rt, "ternary_kappa1", &p->kappa1);
  rt_double_parameter(rt, "ternary_kappa2", &p->kappa2);
  rt_double_parameter(rt, "ternary_kappa3", &p->kappa3);
  rt_double_parameter(rt, "ternary_alpha",  &p->alpha);
    
  /* For the surfactant should have... */

  if (p->kappa1 < 0.0) pe_fatal(pe, "Please use ternary_kappa1 >= 0\n");
  if (p->kappa2 < 0.0) pe_fatal(pe, "Please use ternary_kappa2 >= 0\n");
  if (p->kappa3 < 0.0) pe_fatal(pe, "Please use ternary_kappa3 >= 0\n");
  if (p->alpha <= 0.0) pe_fatal(pe, "Please use ternary_alpha > 0\n");

  /* Optional wetting parameters */

  {
    int have_wetting = 0;

    have_wetting += rt_double_parameter(rt, "ternary_h1", &p->h1);
    have_wetting += rt_double_parameter(rt, "ternary_h2", &p->h2);

    /* Only h1 and h2 may be specified independently. h3 is then
       determined by the constraint h1/k1 + h2/k2 + h3/k3 = 0 */

    if (have_wetting) {
      /* Must have both... */
      rt_key_required(rt, "ternary_h1", RT_FATAL);
      rt_key_required(rt, "ternary_h2", RT_FATAL);

      /* h_3 is from the constraint h1/k1 + h2/k2 + h3/k3 = 0 */
      /* Not specified independently in the input. */

      p->h3 = -p->kappa3*(p->h1/p->kappa1 + p->h2/p->kappa2);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_ternary_init_rt
 *
 *  Initialise fields: phi, psi, rho. These are related.
 *
 *****************************************************************************/

__host__ int fe_ternary_init_rt(pe_t * pe, rt_t * rt, fe_ternary_t * fe,
				field_t * phi) {
  int p;
  char value[BUFSIZ];
    
  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  p = rt_string_parameter(rt, "ternary_initialisation", value, BUFSIZ);

  pe_info(pe, "\n");
  pe_info(pe, "Initialising fields for ternary fluid\n");

  if (p != 0 && strcmp(value, "ternary_X") == 0) {
    field_ternary_init_X(phi);
  }

  if (p != 0 && strcmp(value, "2d_double_emulsion") == 0) {

    fti_block_t param = {};

    /* Default parameters (historically to give roughly equal areas) */
    param.xf1 = 0.2;
    param.xf2 = 0.5;
    param.xf3 = 0.8;
    param.yf1 = 0.3;
    param.yf2 = 0.7;

    /* Optional user input */
    rt_double_parameter(rt, "2d_double_emulsion_xf1", &param.xf1);
    rt_double_parameter(rt, "2d_double_emulsion_xf2", &param.xf2);
    rt_double_parameter(rt, "2d_double_emulsion_xf3", &param.xf3);
    rt_double_parameter(rt, "2d_double_emulsion_yf1", &param.yf1);
    rt_double_parameter(rt, "2d_double_emulsion_yf2", &param.yf2);

    field_ternary_init_2d_double_emulsion(phi, &param);

    pe_info(pe, "Composition is 2d block double emulsion initialisation\n");
    pe_info(pe, "Interface at xf1 Lx (left)     %12.5e\n", param.xf1);
    pe_info(pe, "Interface at xf2 Lx (centre)   %12.5e\n", param.xf2);
    pe_info(pe, "Interface at xf3 Lx (right)    %12.5e\n", param.xf3);
    pe_info(pe, "Interface at yf1 Ly (bottom)   %12.5e\n", param.yf1);
    pe_info(pe, "Interface at yf2 Ly (top)      %12.5e\n", param.yf2);
    pe_info(pe, "\n");
  }

  if (p != 0 && strcmp(value, "2d_tee") == 0) {

    fti_block_t param = {};

    /* Default parameters (roughly equal area) */

    param.xf1 = 0.50;
    param.yf1 = 0.33; /* (sic) */

    /* User optional parameter settings */
    rt_double_parameter(rt, "ternary_2d_tee_xf1", &param.xf1);
    rt_double_parameter(rt, "tarnary_2d_tee_yf1", &param.yf1);

    field_ternary_init_2d_tee(phi, &param);

    pe_info(pe, "Composition is 2d T-shape initialisation\n");
    pe_info(pe, "Interface at xf1 Lx (vertical)   %12.5e\n", param.xf1);
    pe_info(pe, "Interface at yf1 Ly (horizontal) %12.5e\n", param.yf1);
    pe_info(pe, "\n");
  }

  if (p != 0 && strcmp(value, "2d_lens") == 0) {

    fti_drop_t drop = {};

    /* No defaults */ 
    rt_key_required(rt, "ternary_2d_lens_centre", RT_FATAL);
    rt_key_required(rt, "ternary_2d_lens_radius", RT_FATAL);

    rt_double_nvector(rt,   "ternary_2d_lens_centre", 2, drop.r0, RT_FATAL);
    rt_double_parameter(rt, "ternary_2d_lens_radius", &drop.r);

    field_ternary_init_2d_lens(phi, &drop);
  }

  if (p != 0 && strcmp(value, "2d_double_drop") == 0) {

    fti_drop_t drop1 = {};
    fti_drop_t drop2 = {};

    /* No defaults */
    rt_key_required(rt, "ternary_2d_drop1_centre", RT_FATAL);
    rt_key_required(rt, "ternary_2d_drop1_radius", RT_FATAL);
    rt_key_required(rt, "ternary_2d_drop2_centre", RT_FATAL);
    rt_key_required(rt, "ternary_2d_drop2_radius", RT_FATAL);

    rt_double_parameter(rt, "ternary_2d_drop1_radius", &drop1.r);
    rt_double_parameter(rt, "ternary_2d_drop2_radius", &drop2.r);
    rt_double_nvector(rt,   "ternary_2d_drop1_centre", 2, drop1.r0, RT_FATAL);
    rt_double_nvector(rt,   "ternary_2d_drop2_centre", 2, drop2.r0, RT_FATAL);

    field_ternary_init_2d_double_drop(phi, &drop1, &drop2);
  }

  /* File initialisations */

  if (p != 0 && strcmp(value, "from_file") == 0) {

    io_info_t * iohandler = NULL;
    char filestub[FILENAME_MAX] = "ternary.init";

    rt_string_parameter(rt, "ternary_file_stub", filestub, FILENAME_MAX);
    pe_info(pe, "Initial order parameter requested with file stub %s\n",
	    filestub);
    field_io_info(phi, &iohandler);
    io_read_data(iohandler, filestub, phi);
  }

  return 0;
}
