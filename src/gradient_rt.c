/*****************************************************************************
 *
 *  gradient_rt.c
 *
 *  Set the gradient routine. 
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2017 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "pe.h"
#include "runtime.h"
#include "gradient_rt.h"

/*****************************************************************************
 *
 *  gradient_rt_init
 *
 *  For the given field gradient object, set the call back to
 *  function do the computation.
 *
 *  TODO: one could try to filter at this stage valid methods,
 *  but a bit of a pain (what works in what situation?).
 *
 *****************************************************************************/

int gradient_rt_init(pe_t * pe, rt_t * rt, const char * fieldname,
		     field_grad_t * grad, map_t * map,
		     colloids_info_t * cinfo) {

  int n;
  char keyvalue[BUFSIZ];

  int (* f2) (field_grad_t * fg) = NULL;
  int (* f4) (field_grad_t * fg) = NULL;

  assert(grad);

  /* fd_gradient_calculation is     defualt
   * fd_gradient_calculation_phi    overrides if and only if field is phi
   * fd_gradient_calculation_q      overrides if and only if field is Q_ab
   */

  n = rt_string_parameter(rt, "fd_gradient_calculation", keyvalue, BUFSIZ);

  if (strcmp(fieldname, "phi") == 0) {
    n += rt_string_parameter(rt, "fd_gradient_calculation_phi", keyvalue,
			     BUFSIZ);
  }
  if (strcmp(fieldname, "q") == 0) {
    n += rt_string_parameter(rt, "fd_gradient_calculation_q", keyvalue,
			     BUFSIZ);
  }

  if (n == 0) {
    pe_info(pe, "You must specify the keyvalue fd_gradient_calculation\n");
    pe_fatal(pe, "Please check and try again\n");
  }
  else {
    pe_info(pe, "Gradient calculation: ");
    if (strcmp(keyvalue, "2d_5pt_fluid") == 0) {
      pe_info(pe, "2d_5pt_fluid\n");
      f2 = grad_2d_5pt_fluid_d2;
      f4 = grad_2d_5pt_fluid_d4;
    }
    else if (strcmp(keyvalue, "2d_tomita_fluid") == 0) {
      pe_info(pe, "2d_tomita_fluid\n");
      f2 = grad_2d_tomita_fluid_d2;
      f4 = grad_2d_tomita_fluid_d4;
    }
    else if (strcmp(keyvalue, "3d_7pt_fluid") == 0) {
      pe_info(pe, "3d_7pt_fluid\n");
      f2 = grad_3d_7pt_fluid_d2;
      f4 = grad_3d_7pt_fluid_d4;
      field_grad_dab_set(grad, grad_3d_7pt_fluid_dab);
    }
    else if (strcmp(keyvalue, "3d_7pt_solid") == 0) {
      pe_info(pe, "3d_7pt_solid\n");
      f2 = grad_3d_7pt_solid_d2;
      f4 = NULL;
      assert(map);
      grad_3d_7pt_solid_set(map, cinfo);
    }
    else if (strcmp(keyvalue, "3d_27pt_fluid") == 0) {
      pe_info(pe, "3d_27pt_fluid\n");
      f2 = grad_3d_27pt_fluid_d2;
      f4 = grad_3d_27pt_fluid_d4;
      field_grad_dab_set(grad, grad_3d_27pt_fluid_dab);
    }
    else if (strcmp(keyvalue, "3d_27pt_solid") == 0) {
      pe_info(pe, "3d_27pt_solid\n");
      f2 = grad_3d_27pt_solid_d2;
      f4 = NULL;
      field_grad_dab_set(grad, grad_3d_27pt_solid_dab);
      assert(map);
      grad_3d_27pt_solid_map_set(map);
    }
    else {
      /* Not recognised */
      pe_info(pe, "\nfd_gradient_calculation %s not recognised\n", keyvalue);
      pe_fatal(pe, "Please check and try again\n");
    }
  }

  field_grad_set(grad, f2, f4);

  return 0;
}
