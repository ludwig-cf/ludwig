/*****************************************************************************
 *
 *  test_fe_force_method.c
 *
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing author:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "pe.h"
#include "fe_force_method.h"

int test_fe_force_method_default(void);
int test_fe_force_method_to_enum(void);
int test_fe_force_method_to_string(void);

/*****************************************************************************
 *
 *  test_fe_force_method_suite
 *
 *****************************************************************************/

int test_fe_force_method_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* If the implementation has changed, the tests need to change. */
  assert(FE_FORCE_METHOD_MAX == 7);

  test_fe_force_method_default();
  test_fe_force_method_to_enum();
  test_fe_force_method_to_string();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_force_method_default
 *
 *****************************************************************************/

int test_fe_force_method_default(void) {

  fe_force_method_enum_t method = fe_force_method_default();

  assert(method == FE_FORCE_METHOD_STRESS_DIVERGENCE);

  return (method == FE_FORCE_METHOD_STRESS_DIVERGENCE);
}

/*****************************************************************************
 *
 *  test_fe_force_method_to_enum
 *
 *****************************************************************************/

int test_fe_force_method_to_enum(void) {

  int ifail = 0;
  fe_force_method_enum_t method = FE_FORCE_METHOD_INVALID;

  method = fe_force_method_to_enum("no_force");
  assert(method == FE_FORCE_METHOD_NO_FORCE);

  method = fe_force_method_to_enum("stress_divergence");
  assert(method == FE_FORCE_METHOD_STRESS_DIVERGENCE);

  method = fe_force_method_to_enum("phi_gradmu");
  assert(method == FE_FORCE_METHOD_PHI_GRADMU);

  method = fe_force_method_to_enum("phi_gradmu_correction");
  assert(method == FE_FORCE_METHOD_PHI_GRADMU_CORRECTION);

  method = fe_force_method_to_enum("relaxation_symmetric");
  assert(method == FE_FORCE_METHOD_RELAXATION_SYMM);

  method = fe_force_method_to_enum("relaxation_antisymmetric");
  assert(method == FE_FORCE_METHOD_RELAXATION_ANTI);
  if (method != FE_FORCE_METHOD_RELAXATION_ANTI) ifail = -1;

  return ifail;
}

/*****************************************************************************
 *
 *  test_fe_force_method_to_string
 *
 *****************************************************************************/

int test_fe_force_method_to_string(void) {

  int ifail = 0;


  {
    fe_force_method_enum_t method = FE_FORCE_METHOD_NO_FORCE;
    const char * s = fe_force_method_to_string(method);
    ifail = strcmp(s, "no_force");
    assert(ifail == 0);
  }

  {
    fe_force_method_enum_t method = FE_FORCE_METHOD_STRESS_DIVERGENCE;
    const char * s = fe_force_method_to_string(method);
    ifail = strcmp(s, "stress_divergence");
    assert(ifail == 0);
  }

  {
    fe_force_method_enum_t method = FE_FORCE_METHOD_PHI_GRADMU;
    const char * s = fe_force_method_to_string(method);
    ifail = strcmp(s, "phi_gradmu");
    assert(ifail == 0);
  }

  {
    fe_force_method_enum_t method = FE_FORCE_METHOD_PHI_GRADMU_CORRECTION;
    const char * s = fe_force_method_to_string(method);
    ifail = strcmp(s, "phi_gradmu_correction");
    assert(ifail == 0);
  }

  {
    fe_force_method_enum_t method = FE_FORCE_METHOD_RELAXATION_SYMM;
    const char * s = fe_force_method_to_string(method);
    ifail = strcmp(s, "relaxation_symmetric");
    assert(ifail == 0);
  }

  {
    fe_force_method_enum_t method = FE_FORCE_METHOD_RELAXATION_ANTI;
    const char * s = fe_force_method_to_string(method);
    ifail = strcmp(s, "relaxation_antisymmetric");
    assert(ifail == 0);
  }

  return ifail;
}
