/****************************************************************************
 *
 *  symmetric_rt.c
 *
 *  Run time initialisation for the symmetric phi^4 free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>


#include "field_phi_init_rt.h"
#include "symmetric_rt.h"

#include "gradient_3d_27pt_solid.h"
#include "physics.h"
#include "util.h"

/****************************************************************************
 *
 *  fe_symmetric_init_rt
 *
 ****************************************************************************/

int fe_symmetric_init_rt(pe_t * pe, rt_t * rt, fe_symm_t * fe) {

  int old_keys = 0;
  double sigma;
  double xi;
  fe_symm_param_t param = {0};

  assert(pe);
  assert(rt);
  assert(fe);

  pe_info(pe, "Symmetric phi^4 free energy selected.\n");
  pe_info(pe, "\n");

  /* Parameters */

  old_keys += rt_double_parameter(rt, "A", &param.a);
  old_keys += rt_double_parameter(rt, "B", &param.b);
  old_keys += rt_double_parameter(rt, "K", &param.kappa);

  if (old_keys) {

  pe_info(pe, "Parameters:\n");
  pe_info(pe, "Bulk parameter A      = %12.5e\n", param.a);
  pe_info(pe, "Bulk parameter B      = %12.5e\n", param.b);
  pe_info(pe, "Surface penalty kappa = %12.5e\n", param.kappa);

  fe_symm_param_set(fe, param);

  fe_symm_interfacial_tension(fe, &sigma);
  fe_symm_interfacial_width(fe, &xi);

  pe_info(pe, "Surface tension       = %12.5e\n", sigma);
  pe_info(pe, "Interfacial width     = %12.5e\n", xi);

  }
  else {

    /* Updated parameters (now the preferred names) */
    int have_symm = 0;
    int have_wetting = 0;
    int have_theta = 0;
    double theta = 0.0; /* If present, to be angle in degrees */

    have_symm += rt_double_parameter(rt, "symmetric_a",     &param.a);
    have_symm += rt_double_parameter(rt, "symmetric_b",     &param.b);
    have_symm += rt_double_parameter(rt, "symmetric_kappa", &param.kappa);

    if (have_symm) {
      /* must have all three */
      rt_key_required(rt, "symmetric_a",     RT_FATAL);
      rt_key_required(rt, "symmetric_b",     RT_FATAL);
      rt_key_required(rt, "symmetric_kappa", RT_FATAL);
    }

    /* Uniform wetting */
    have_wetting += rt_double_parameter(rt, "symmetric_c", &param.c);
    have_wetting += rt_double_parameter(rt, "symmetric_h", &param.h);
    have_theta   += rt_double_parameter(rt, "symmetric_theta", &theta);

    if (have_theta) {
      /* Set appropriate H = h sqrt(kappa B), C = 0 */
      double h = 0.0; /* dimensionless "small" h */
      fe_symm_theta_to_h(theta, &h);
      param.c = 0.0;
      param.h = h*sqrt(param.kappa*param.b);
      {
	/* Sign of h will reflect sign of cos(theta) as input */
	PI_DOUBLE(pi);
	if (cos(pi*theta/180.0) < 0.0) param.h = -param.h;
      }
    }

    if (have_wetting && have_theta) {
      /* Please have one or the other; not both. */
      pe_info(pe, "Both symmetric_theta and symmetric_[ch] are present.\n");
      pe_info(pe, "Please use symmetric_theta (only) or, one or both of\n"
	          "symmetric_c / symmetric_h for uniform wetting\n");
      pe_fatal(pe, "Please check and try again.\n");
    }

    /* Report */

    pe_info(pe, "Parameters:\n");
    pe_info(pe, "Bulk parameter A      = %12.5e\n", param.a);
    pe_info(pe, "Bulk parameter B      = %12.5e\n", param.b);
    pe_info(pe, "Surface penalty kappa = %12.5e\n", param.kappa);

    fe_symm_param_set(fe, param);

    fe_symm_interfacial_tension(fe, &sigma);
    fe_symm_interfacial_width(fe, &xi);

    pe_info(pe, "Surface tension       = %12.5e\n", sigma);
    pe_info(pe, "Interfacial width     = %12.5e\n", xi);

    if (have_wetting || have_theta) {
      double costheta = 0.0;
      double mytheta  = 0.0;
      PI_DOUBLE(pi);
      double h = param.h/sqrt(param.kappa*param.b); /* Small h */
      fe_symm_h_to_costheta(h, &costheta);
      mytheta = 180.0*acos(costheta)/pi;
      pe_info(pe, "Surface parameter C      = %12.5e\n", param.c);
      pe_info(pe, "Surface parameter H      = %12.5e\n", param.h);
      pe_info(pe, "Dimensionless h          = %12.5e\n", h);
      pe_info(pe, "Uniform wetting angle    = %12.5e degrees\n", mytheta);
    }

  }

  /* Initialise */
  grad_3d_27pt_solid_fe_set(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_phi_init_rt
 *
 *****************************************************************************/

__host__
int fe_symmetric_phi_init_rt(pe_t * pe, rt_t * rt, fe_symm_t * fe,
			     field_t * phi) {

  physics_t * phys = NULL;
  field_phi_info_t param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  physics_ref(&phys);
  physics_phi0(phys, &param.phi0);

  fe_symm_interfacial_width(fe, &param.xi0);

  field_phi_init_rt(pe, rt, param, phi);

  return 0;
}
