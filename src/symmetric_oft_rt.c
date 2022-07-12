/****************************************************************************
 *
 *  symmetric_oft_rt.c
 *
 *  Run time initialisation for the symmetric_oft phi^4 free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2021 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "field_phi_init_rt.h"
#include "field_temperature_init_rt.h"

#include "symmetric_oft_rt.h"

#include "gradient_3d_27pt_solid.h"
#include "physics.h"
#include "util.h"

/****************************************************************************
 *
 *  fe_symmetric_oft_init_rt
 *
 ****************************************************************************/

int fe_symmetric_oft_init_rt(pe_t * pe, rt_t * rt, fe_symm_oft_t * fe) {

  int old_keys = 0;
  double sigma, lambda;
  double xi;
  fe_symm_oft_param_t param = {};

  assert(pe);
  assert(rt);
  assert(fe);

  pe_info(pe, "Temperature-dependent symmetric phi^4 free energy selected.\n");
  pe_info(pe, "\n");

  /* Parameters */

  /* Updated parameters (now the preferred names) */
  int have_symm = 0;
  int have_wetting = 0;
  int have_theta = 0;
  double theta = 0.0; /* If present, to be angle in degrees */


/* <-------------- CHECK SYMM PARAMETERS -------------------> */
  have_symm += rt_double_parameter(rt, "symmetric_oft_a0",     &param.a0);
  have_symm += rt_double_parameter(rt, "symmetric_oft_a",     &param.a);
  have_symm += rt_double_parameter(rt, "symmetric_oft_b0",     &param.b0);
  have_symm += rt_double_parameter(rt, "symmetric_oft_b",     &param.b);
  have_symm += rt_double_parameter(rt, "symmetric_oft_kappa0", &param.kappa0);
  have_symm += rt_double_parameter(rt, "symmetric_oft_kappa", &param.kappa);
  have_symm += rt_double_parameter(rt, "symmetric_oft_lambda", &param.lambda);
  have_symm += rt_double_parameter(rt, "symmetric_oft_entropy", &param.entropy);
  if (have_symm) {
      /* must have all parameters */
    rt_key_required(rt, "symmetric_oft_a0",     RT_FATAL);
    rt_key_required(rt, "symmetric_oft_a",     RT_FATAL);
    rt_key_required(rt, "symmetric_oft_b0",     RT_FATAL);
    rt_key_required(rt, "symmetric_oft_b",     RT_FATAL);
    rt_key_required(rt, "symmetric_oft_kappa0", RT_FATAL);
    rt_key_required(rt, "symmetric_oft_kappa", RT_FATAL);
    rt_key_required(rt, "symmetric_oft_lambda", RT_FATAL);
    rt_key_required(rt, "symmetric_oft_entropy", RT_FATAL);
    param.lambda = lambda;
  }


/* <-------------- CHECK WETTING PARAMETERS -------------------> */
  /* Uniform wetting */
  have_wetting += rt_double_parameter(rt, "symmetric_oft_c", &param.c);
  have_wetting += rt_double_parameter(rt, "symmetric_oft_h", &param.h);
  have_theta   += rt_double_parameter(rt, "symmetric_oft_theta", &theta);

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
    pe_info(pe, "Both symmetric_theta and symmetric[ch] are present.\n");
    pe_info(pe, "Please use symmetric_theta (only) or, one or both of\n"
         "symmetric_c / symmetric_h for uniform wetting\n");
    pe_fatal(pe, "Please check and try again.\n");
  }


/* <-------------- REPORT -------------------> */
  pe_info(pe, "Parameters:\n");
  pe_info(pe, "Bulk parameter A0     = %12.5e\n", param.a0);
  pe_info(pe, "Bulk parameter A      = %12.5e\n", param.a);
  pe_info(pe, "Bulk parameter B0      = %12.5e\n", param.b0);
  pe_info(pe, "Bulk parameter B      = %12.5e\n", param.b);
  pe_info(pe, "Surface penalty kappa0 = %12.5e\n", param.kappa0);
  pe_info(pe, "Surface penalty kappa = %12.5e\n", param.kappa);
  pe_info(pe, "Thermal diffusivity  = %12.5e\n", param.lambda);
  pe_info(pe, "Entropy = %12.5e\n", param.entropy);
  fe_symm_oft_param_set(fe, param);
  fe_symm_oft_interfacial_tension(fe, &sigma);
  fe_symm_oft_interfacial_width(fe, &xi);
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

  /* Initialise */
  grad_3d_27pt_solid_symm_oft_set(fe);
  
  return 0;
}

/*****************************************************************************
 *
 *  fe_symmetric_oft_phi_init_rt
 *
 *****************************************************************************/

__host__
int fe_symmetric_oft_phi_init_rt(pe_t * pe, rt_t * rt, fe_symm_oft_t * fe,
			     field_t * phi) {

  physics_t * phys = NULL;
  field_phi_info_t param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(phi);

  physics_ref(&phys);
  physics_phi0(phys, &param.phi0);

  fe_symm_oft_interfacial_width(fe, &param.xi0);

  field_phi_init_rt(pe, rt, param, phi);

  return 0;
}


/*****************************************************************************
 *
 *  fe_symmetric_oft_temperature_init_rt
 *
 *****************************************************************************/

__host__
int fe_symmetric_oft_temperature_init_rt(pe_t * pe, rt_t * rt, fe_symm_oft_t * fe, field_t * temperature) {

  assert(pe);
  assert(rt);
  assert(fe);
  assert(temperature);
 
  field_temperature_init_rt(pe, rt, temperature);

  return 0;
}
