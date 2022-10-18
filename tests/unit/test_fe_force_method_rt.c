/*****************************************************************************
 *
 *  test_fe_force_method_rt.c
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

#include "fe_force_method_rt.h"

int test_fe_force_method_rt(pe_t * pe, fe_force_method_enum_t method);
int test_fe_force_method_rt_messages(pe_t * pe);

/*****************************************************************************
 *
 *  test_fe_force_method_rt_suite
 *
 *****************************************************************************/

int test_fe_force_method_rt_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_fe_force_method_rt(pe, FE_FORCE_METHOD_STRESS_DIVERGENCE);
  test_fe_force_method_rt(pe, FE_FORCE_METHOD_PHI_GRADMU);
  test_fe_force_method_rt(pe, FE_FORCE_METHOD_PHI_GRADMU_CORRECTION);
  test_fe_force_method_rt(pe, FE_FORCE_METHOD_RELAXATION_SYMM);
  test_fe_force_method_rt(pe, FE_FORCE_METHOD_RELAXATION_ANTI);

  test_fe_force_method_rt_messages(pe);

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_fe_force_method_rt
 *
 *****************************************************************************/

int test_fe_force_method_rt(pe_t * pe, fe_force_method_enum_t method) {

  int ifail = 0;
  rt_t * rt = NULL;

  assert(pe);

  rt_create(pe, &rt);

  /* No keys present */
  {
    fe_force_method_enum_t imethod = fe_force_method_default();
    int iret = fe_force_method_rt(rt, RT_NONE, &imethod);
    assert(iret == 0);
    assert(imethod == fe_force_method_default());
    if (iret != 0) ifail += 1;
  }

  switch (method) {
  case (FE_FORCE_METHOD_STRESS_DIVERGENCE):
    rt_add_key_value(rt, "fe_force_method", "stress_divergence");
    {
      fe_force_method_enum_t imethod = FE_FORCE_METHOD_INVALID;
      int iret = fe_force_method_rt(rt, RT_NONE, &imethod);
      assert(iret == 1);
      assert(imethod == FE_FORCE_METHOD_STRESS_DIVERGENCE);
      if (iret != 1) ifail += 1;
    }
    break;
  case (FE_FORCE_METHOD_PHI_GRADMU):
    rt_add_key_value(rt, "fe_force_method", "phi_gradmu");
    {
      fe_force_method_enum_t imethod = FE_FORCE_METHOD_INVALID;
      int iret = fe_force_method_rt(rt, RT_NONE, &imethod);
      assert(iret == 1);
      assert(imethod == FE_FORCE_METHOD_PHI_GRADMU);
      if (iret != 1) ifail += 1;
    }
    break;
  case (FE_FORCE_METHOD_PHI_GRADMU_CORRECTION):
    rt_add_key_value(rt, "fe_force_method", "phi_gradmu_correction");
    {
      fe_force_method_enum_t imethod = FE_FORCE_METHOD_INVALID;
      int iret = fe_force_method_rt(rt, RT_NONE, &imethod);
      assert(iret == 1);
      assert(imethod == FE_FORCE_METHOD_PHI_GRADMU_CORRECTION);
      if (iret != 1) ifail += 1;
    }
    break;
  case (FE_FORCE_METHOD_RELAXATION_SYMM):
    rt_add_key_value(rt, "fe_force_method", "relaxation_symmetric");
    {
      fe_force_method_enum_t imethod = FE_FORCE_METHOD_INVALID;
      int iret = fe_force_method_rt(rt, RT_NONE, &imethod);
      assert(iret == 1);
      assert(imethod == FE_FORCE_METHOD_RELAXATION_SYMM);
      if (iret != 1) ifail += 1;
    }
    break;
  case (FE_FORCE_METHOD_RELAXATION_ANTI):
    rt_add_key_value(rt, "fe_force_method", "relaxation_antisymmetric");
    {
      fe_force_method_enum_t imethod = FE_FORCE_METHOD_INVALID;
      int iret = fe_force_method_rt(rt, RT_NONE, &imethod);
      assert(iret == 1);
      assert(imethod == FE_FORCE_METHOD_RELAXATION_ANTI);
      if (iret != 1) ifail += 1;
    }
    break;
  default:
    /* Nothing. */
    ;
  }

  rt_free(rt);

  return ifail;
}

/*****************************************************************************
 *
 *  test_fe_force_method_rt_messages
 *
 *****************************************************************************/

int test_fe_force_method_rt_messages(pe_t * pe) {

  int ifail = -1;
  rt_t * rt = NULL;

  assert(pe);

  rt_create(pe, &rt);

  ifail = fe_force_method_rt_messages(rt, RT_NONE);
  assert(ifail == 0);

  {
    int ierr = 0;
    rt_add_key_value(rt, "fd_force_divergence", "0");
    ierr = fe_force_method_rt_messages(rt, RT_NONE);
    assert(ierr == -1);
    if (ierr != -1) ifail += 1;
  }

  {
    int ierr = 0;
    rt_add_key_value(rt, "fe_use_stress_relaxation", "yes");
    ierr = fe_force_method_rt_messages(rt, RT_NONE);
    assert(ierr == -2);
    if (ierr != -2) ifail += 1;
  }

  rt_free(rt);

  return ifail;
}
