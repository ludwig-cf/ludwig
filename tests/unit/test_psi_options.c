/*****************************************************************************
 *
 *  test_psi_options.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#include "pe.h"
#include "psi_options.h"

int test_psi_options_default(void);
int test_psi_options_to_json(void);
int test_psi_options_from_json(void);
int test_psi_bjerrum_length(void);
int test_psi_debye_length(void);

/*****************************************************************************
 *
 *  test_psi_options_suite
 *
 *****************************************************************************/

int test_psi_options_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* A change in components requires a test update... */
  assert(sizeof(psi_options_t) == 392);
  assert(PSI_NKMAX >= 2);

  test_psi_options_default();
  test_psi_options_to_json();
  test_psi_options_from_json();
  test_psi_bjerrum_length();
  test_psi_debye_length();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_psi_options_default
 *
 *****************************************************************************/

int test_psi_options_default(void) {

  int ifail = 0;

  psi_options_t opts = psi_options_default(0);

  assert(opts.nk == 2);
  if (opts.nk != 2) ifail = -1;

  /* Physics */
  assert(fabs(opts.e              - 1.0)     < DBL_EPSILON);
  assert(fabs(opts.beta           - 1.0)     < DBL_EPSILON);
  assert(fabs(opts.epsilon1       - 10000.0) < DBL_EPSILON);
  assert(fabs(opts.epsilon2       - 10000.0) < DBL_EPSILON);
  assert(fabs(opts.e0[0]          - 0.0)     < DBL_EPSILON);
  assert(fabs(opts.e0[1]          - 0.0)     < DBL_EPSILON);
  assert(fabs(opts.e0[2]          - 0.0)     < DBL_EPSILON);
  assert(fabs(opts.diffusivity[0] - 0.01)    < DBL_EPSILON);
  assert(fabs(opts.diffusivity[1] - 0.01)    < DBL_EPSILON);

  assert(opts.valency[0] == +1);
  assert(opts.valency[1] == -1);

  /* Solver */
  assert(opts.solver.psolver == PSI_POISSON_SOLVER_SOR);

  /* Nernst Planck */
  assert(opts.nsolver    == -1);
  assert(opts.nsmallstep ==  1);
  assert(fabs(opts.diffacc - 0.0) < DBL_EPSILON);

  /* Other */

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_options_to_json
 *
 *****************************************************************************/

int test_psi_options_to_json(void) {

  int ifail = 0;
  psi_options_t opts = psi_options_default(0);
  cJSON * json = NULL;

  ifail = psi_options_to_json(&opts, &json);
  assert(ifail == 0);

  {
    psi_options_t check = {0};
    ifail = psi_options_from_json(json, &check);
    assert(ifail == 0);
    if (check.nk != 2) ifail = -1;
    assert(check.nk == 2);
  }

  cJSON_Delete(json);

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_options_from_json
 *
 *  Using the minimum requirement for solver options.
 *
 *****************************************************************************/

int test_psi_options_from_json(void) {

  int ifail = 0;
  const char * jstr = "{ \"Number of species\":     2,"
                      "  \"Unit charge\":           2.5,"
                      "  \"Boltzmann factor\":      3.0,"
                      "  \"First permittivity\":    2000.0,"
                      "  \"Second permittivity\":   3000.0,"
                      "  \"Valencies\":             [1, -1],"
                      "  \"Diffusivities\":         [0.01, 0.02],"
                      "  \"External field\":        [1.0, 2.0, 3.0],"
                      "  \"Solver options\":        {"
                      "         \"Solver type\":    \"sor\" "
                      "}}";

  cJSON * json = cJSON_Parse(jstr);
  assert(json);

  {
    /* Check result */
    psi_options_t opts = {0};
    ifail = psi_options_from_json(json, &opts);
    assert(ifail == 0);

    assert(opts.nk == 2);
    assert(fabs(opts.e    - 2.5) < DBL_EPSILON);
    assert(fabs(opts.beta - 3.0) < DBL_EPSILON);
    assert(fabs(opts.epsilon1 - 2000.0) < DBL_EPSILON);
    assert(fabs(opts.epsilon2 - 3000.0) < DBL_EPSILON);
    assert(opts.valency[0] == +1);
    assert(opts.valency[1] == -1);
    assert(fabs(opts.diffusivity[0] - 0.01) < DBL_EPSILON);
    assert(fabs(opts.diffusivity[1] - 0.02) < DBL_EPSILON);
    assert(fabs(opts.e0[0] - 1.0) < DBL_EPSILON);
    assert(fabs(opts.e0[1] - 2.0) < DBL_EPSILON);
    assert(fabs(opts.e0[2] - 3.0) < DBL_EPSILON);

    assert(opts.solver.psolver == PSI_POISSON_SOLVER_SOR);
  }

  cJSON_Delete(json);

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_bjerrum_length
 *
 *  Check both Bjerrum lengths.
 *
 *****************************************************************************/

int test_psi_bjerrum_length(void) {

  int ifail = 0;

  {
    psi_options_t opts = psi_options_default(0);
    double b1 = 0.0;
    double b2 = 0.0;

    ifail = psi_bjerrum_length1(&opts, &b1);
    assert(ifail == 0);
    ifail = psi_bjerrum_length2(&opts, &b2);
    assert(ifail == 0);
    {
      double e        = opts.e;
      double kt       = 1.0/opts.beta;
      double epsilon  = opts.epsilon1;
      double lbjerrum = e*e/(4.0*4.0*atan(1.0)*epsilon*kt);

      ifail = (fabs(b1 - lbjerrum) > DBL_EPSILON);
      assert(ifail == 0);
      ifail = (fabs(b2 - lbjerrum) > DBL_EPSILON);
      assert(ifail == 0);
    }
  }

  {
    double e = 1.0;
    double ktref = 0.00001;
    double epsilon1 = 41.4*1000.0;
    double epsilon2 = 57.1*1000.0;
    psi_options_t opts = {.e = e, .beta = 1.0/ktref, .epsilon1 = epsilon1,
                          .epsilon2 = epsilon2};
    double b1 = 0.0;
    double b2 = 0.0;

    ifail = psi_bjerrum_length1(&opts, &b1);
    assert(ifail == 0);
    assert(fabs(b1 - 0.19221611) < FLT_EPSILON);

    ifail = psi_bjerrum_length2(&opts, &b2);
    assert(ifail == 0);
    assert(fabs(b2 - 0.13936510) < FLT_EPSILON);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_debye_length
 *
 *  Check both Debye lengths.
 *
 *****************************************************************************/

int test_psi_debye_length(void) {

  int ifail = 0;

  {
    psi_options_t opts = psi_options_default(0);
    double rho_b   = 6.0;
    double ldebye1 = 0.0;
    double ldebye2 = 0.0;
    double ldebye0 = 0.0;

    ifail = psi_debye_length1(&opts, rho_b, &ldebye1);
    assert(ifail == 0);
    ifail = psi_debye_length2(&opts, rho_b, &ldebye2);
    assert(ifail == 0);

    {
      double lbjerrum = 0.0;

      psi_bjerrum_length1(&opts, &lbjerrum);
      ldebye0 = 1.0/sqrt(8.0*4.0*atan(1.0)*lbjerrum*rho_b);
      ifail = (fabs(ldebye1 - ldebye0) > DBL_EPSILON);
      assert(ifail == 0);

      psi_bjerrum_length2(&opts, &lbjerrum);
      ldebye0 = 1.0/sqrt(8.0*4.0*atan(1.0)*lbjerrum*rho_b);
      ifail = (fabs(ldebye2 - ldebye0) > DBL_EPSILON);
      assert(ifail == 0);
    }
  }

  {
    /* Some numbers from an historical example. */
    double e = 1.0;
    double beta = 1.0/0.00033333;
    double epsilon1 = 300.0;
    double epsilon2 = 400.0;
    double rho_el   = 0.00047;
    double ldebye1  = 0.0;
    double ldebye2  = 0.0;
    psi_options_t opts = {.e = e, .beta = beta, .epsilon1 = epsilon1,
                          .epsilon2 = epsilon2};

    ifail = psi_debye_length1(&opts, rho_el, &ldebye1);
    assert(ifail == 0);
    ifail = psi_debye_length2(&opts, rho_el, &ldebye2);
    assert(ifail == 0);

    assert(fabs(ldebye1 - 1.03141609e+01) < FLT_EPSILON);
    assert(fabs(ldebye2 - 1.19097671e+01) < FLT_EPSILON);
  }

  return ifail;
}
