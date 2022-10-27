/*****************************************************************************
 *
 *  fe_force_method.c
 *
 *  Utility to describe the implementation of the force from the free
 *  energy sector.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing author:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "fe_force_method.h"

/*****************************************************************************
 *
 *  fe_force_method_default
 *
 *****************************************************************************/

fe_force_method_enum_t fe_force_method_default(void) {

  return FE_FORCE_METHOD_STRESS_DIVERGENCE;
}

/*****************************************************************************
 *
 * fe_force_method_to_enum
 *
 *****************************************************************************/

fe_force_method_enum_t fe_force_method_to_enum(const char * method) {

  fe_force_method_enum_t imethod = FE_FORCE_METHOD_INVALID;

  assert(method);

  if (strcmp(method, "no_force") == 0) {
    imethod = FE_FORCE_METHOD_NO_FORCE;
  }
  else if (strcmp(method, "stress_divergence") == 0) {
    imethod = FE_FORCE_METHOD_STRESS_DIVERGENCE;
  }
  else if (strcmp(method, "phi_gradmu") == 0) {
    imethod = FE_FORCE_METHOD_PHI_GRADMU;
  }
  else if (strcmp(method, "phi_gradmu_correction") == 0) {
    imethod = FE_FORCE_METHOD_PHI_GRADMU_CORRECTION;
  }
  else if (strcmp(method, "relaxation_symmetric") == 0) {
    imethod = FE_FORCE_METHOD_RELAXATION_SYMM;
  }
  else if (strcmp(method, "relaxation_antisymmetric") == 0) {
    imethod = FE_FORCE_METHOD_RELAXATION_ANTI;
  }

  return imethod;
}

/*****************************************************************************
 *
 *  fe_force_method_to_string
 *
 *****************************************************************************/

const char * fe_force_method_to_string(fe_force_method_enum_t method) {

  const char * mstring = "Invalid";

  switch (method) {
  case (FE_FORCE_METHOD_NO_FORCE):
    mstring = "no_force";
    break;
  case (FE_FORCE_METHOD_STRESS_DIVERGENCE):
    mstring = "stress_divergence";
    break;
  case (FE_FORCE_METHOD_PHI_GRADMU):
    mstring = "phi_gradmu";
    break;
  case (FE_FORCE_METHOD_PHI_GRADMU_CORRECTION):
    mstring = "phi_gradmu_correction";
    break;
  case (FE_FORCE_METHOD_RELAXATION_SYMM):
    mstring = "relaxation_symmetric";
    break;
  case (FE_FORCE_METHOD_RELAXATION_ANTI):
    mstring = "relaxation_antisymmetric";
    break;
  default:
    mstring = "Not found";
  }

  return mstring;
}
