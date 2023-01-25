/*****************************************************************************
 *
 *  fe_force_method_rt
 *
 *  Map the input onto the method to implement the thermodynamic force
 *  on the fluid (and potentially colloids).
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "fe_force_method_rt.h"

/*****************************************************************************
 *
 *  fe_force_method_rt
 *
 *  Return one of phi_force_stress enum type depending on key
 *  "force_method"
 *
 *  If no valid value is present, the input method is unchanged.
 *
 *****************************************************************************/

int fe_force_method_rt(rt_t * rt, rt_enum_t lv,
		       fe_force_method_enum_t * method) {

  int key_present = 0;

  assert(rt);
  assert(method);

  key_present = rt_key_present(rt, "fe_force_method");

  if (key_present) {
    char value[BUFSIZ] = {0};
    fe_force_method_enum_t imethod = FE_FORCE_METHOD_INVALID;

    rt_string_parameter(rt, "fe_force_method", value, BUFSIZ);
    imethod = fe_force_method_to_enum(value);

    if (imethod != FE_FORCE_METHOD_INVALID) {
      *method = imethod;
    }
    else {
      rt_vinfo(rt, lv,  "Input file: fe_force_method %s\n", value);
      rt_vinfo(rt, lv,  "Input file: not recognised\n");
      /* PENDING UPDATE TO ft_fatal() */
      rt_vinfo(rt, lv, "Please check and try again\n");
    }
  }

  return key_present;
}

/*****************************************************************************
 *
 *  fe_force_methid_rt_messages
 *
 *  Messages for various conditions which are always in force.
 *  Returns 0 if no messages were issued.
 *
 *****************************************************************************/

int fe_force_method_rt_messages(rt_t * rt, rt_enum_t lv) {

  int ifail = 0;
  assert(rt);

  /* V 0.19.0 */
  /* fd_force_divergence is now replaced by fe_force_method */

  if (rt_key_present(rt, "fd_force_divergence")) {

    rt_vinfo(rt, lv, "Input file contains key: fd_force_divergence\n");
    rt_vinfo(rt, lv, "This should be replaced by \"fe_force_method\"\n");
    rt_vinfo(rt, lv, "See https://ludwig.epcc.ed.ac.uk/inputs/force.html\n");
    /* PENDING REPLACEMENT BY rt_fatal() */
    rt_vinfo(rt, lv, "Please check and try again\n");
    ifail = -1;
  }

  /* V 0.19.0 */
  /* fe_use_stress_relaxation also replaced by fe_force_method */

  if (rt_key_present(rt, "fe_use_stress_relaxation")) {
    rt_vinfo(rt, lv, "Input file contains key: fe_use_stress_relaxation\n");
    rt_vinfo(rt, lv, "This should be replaced by \"fe_force_method\"\n");
    rt_vinfo(rt, lv, "See https://ludwig.epcc.ed.ac.uk/inputs/force.html\n");
    /* PENDING REPLACEMENT BY rt_fatal() */
    rt_vinfo(rt, lv, "Please check and try again\n");
    ifail = -2;
  }

  return ifail;
}
