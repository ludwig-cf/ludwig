/*****************************************************************************
 *
 *  test_psi_solver_options.c
 *
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
#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include <float.h>
#include <math.h>

#include "pe.h"
#include "psi_solver_options.h"

int test_psi_poisson_solver_to_string(void);
int test_psi_poisson_solver_from_string(void);
int test_psi_poisson_solver_default(int argc, char ** argv);
int test_psi_solver_options_default(void);
int test_psi_solver_options_type(void);
int test_psi_solver_options_to_json(void);
int test_psi_solver_options_from_json(void);

/*****************************************************************************
 *
 *  test_psi_solver_options_suite
 *
 *****************************************************************************/

int test_psi_solver_options_suite(int argc, char ** argv) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* Change in size means change in tests required ... */
  assert(sizeof(psi_solver_options_t) == 40);

  test_psi_poisson_solver_to_string();
  test_psi_poisson_solver_from_string();
  test_psi_poisson_solver_default(argc, argv);
  test_psi_solver_options_default();
  test_psi_solver_options_type();
  test_psi_solver_options_to_json();
  test_psi_solver_options_from_json();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_psi_poisson_solver_to_string
 *
 *****************************************************************************/

int test_psi_poisson_solver_to_string(void) {

  int ifail = 0;

  {
    const char * str = psi_poisson_solver_to_string(PSI_POISSON_SOLVER_SOR);
    ifail += strcmp(str, "sor");
    assert(ifail == 0);
  }

  {
    const char * str = psi_poisson_solver_to_string(PSI_POISSON_SOLVER_PETSC);
    ifail += strcmp(str, "petsc");
    assert(ifail == 0);
  }

  {
    const char * str = psi_poisson_solver_to_string(PSI_POISSON_SOLVER_NONE);
    ifail += strcmp(str, "none");
    assert(ifail == 0);
  }

  {
    const char * str = psi_poisson_solver_to_string(PSI_POISSON_SOLVER_INVALID);
    ifail += strcmp(str, "invalid");
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_poisson_solver_from_string
 *
 *****************************************************************************/

int test_psi_poisson_solver_from_string(void) {

  int ifail = 0;

  {
    psi_poisson_solver_enum_t ps = psi_poisson_solver_from_string("SOR");
    if (ps != PSI_POISSON_SOLVER_SOR) ifail += 1;
    assert(ifail == 0);
  }

  {
    psi_poisson_solver_enum_t ps = psi_poisson_solver_from_string("PETSC");
    if (ps != PSI_POISSON_SOLVER_PETSC) ifail += 1;
    assert(ifail == 0);
  }

  {
    psi_poisson_solver_enum_t ps = psi_poisson_solver_from_string("NONE");
    if (ps != PSI_POISSON_SOLVER_NONE) ifail += 1;
  }

  {
    /* Sample rubbish */
    psi_poisson_solver_enum_t ps = psi_poisson_solver_from_string("RUBBISH");
    if (ps != PSI_POISSON_SOLVER_INVALID) ifail += 1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_poisson_solver_default
 *
 *****************************************************************************/

int test_psi_poisson_solver_default(int argc, char ** argv) {

  int ifail = 0;

  {
    /* No Petsc */
    psi_poisson_solver_enum_t psolver = psi_poisson_solver_default();
    if (psolver != PSI_POISSON_SOLVER_SOR) ifail = -1;
    assert(ifail == 0);
  }

  {
    /* Petsc. This requires check with PetscInitialised(). */
    int havePetsc = 0;

    PetscInitialize(&argc, &argv, (char *) 0, NULL);
    PetscInitialised(&havePetsc);
    if (havePetsc) {
      psi_poisson_solver_enum_t psolver = psi_poisson_solver_default();
      if (psolver != PSI_POISSON_SOLVER_PETSC) ifail = -1;
      assert(ifail == 0);
    }
    PetscFinalize();
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_solver_options_default
 *
 *****************************************************************************/

int test_psi_solver_options_default(void) {

  int ifail = 0;
  psi_solver_options_t pso = psi_solver_options_default();

  assert(pso.psolver  == psi_poisson_solver_default());
  assert(pso.maxits   == 10000);
  assert(pso.verbose  == 0);
  assert(pso.nfreq    == INT_MAX);
  assert(pso.nstencil == 7);

  assert(pso.reltol == 1.0e-08);
  assert(pso.abstol == 1.0e-15);

  ifail = pso.verbose;
  assert(ifail == 0);

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_solver_options_type
 *
 *****************************************************************************/

int test_psi_solver_options_type(void) {

  int ifail = 0;

  /* Assume default has passed, and then only the type is relevant... */
  {
    psi_solver_options_t p = psi_solver_options_type(PSI_POISSON_SOLVER_SOR);
    if (p.psolver != PSI_POISSON_SOLVER_SOR) ifail += 1;
    assert(ifail == 0);
  }

  {
    psi_solver_options_t p = psi_solver_options_type(PSI_POISSON_SOLVER_PETSC);
    if (p.psolver != PSI_POISSON_SOLVER_PETSC) ifail += 1;
    assert(ifail == 0);
  }

  {
    psi_solver_options_t p = psi_solver_options_type(PSI_POISSON_SOLVER_NONE);
    if (p.psolver != PSI_POISSON_SOLVER_NONE) ifail += 1;
    assert(ifail == 0);
  }

  return ifail;
}


/*****************************************************************************
 *
 *  test_psi_solver_options_to_json
 *
 *****************************************************************************/

int test_psi_solver_options_to_json(void) {

  int ifail = 0;
  psi_solver_options_t pso = psi_solver_options_default();
  cJSON * json = NULL;

  ifail = psi_solver_options_to_json(&pso, &json);
  assert(ifail == 0);

  {
    /* We assume psi_solver_options_from_json() is independent ... */
    psi_solver_options_t check = {PSI_POISSON_SOLVER_INVALID};
    ifail = psi_solver_options_from_json(json, &check);
    assert(ifail == 0);
    if (check.psolver != PSI_POISSON_SOLVER_SOR) ifail += 1;
    assert(ifail == 0);
  }

  cJSON_Delete(json);

  return ifail;
}

/*****************************************************************************
 *
 *  test_psi_solver_options_from_json
 *
 *****************************************************************************/

int test_psi_solver_options_from_json(void) {

  int ifail = 0;
  const char * jstr = "{\"Solver type\":        \"petsc\","
                      "\"Maximum iterations\":  200,"
                      "\"Level of verbosity\":  2,"
                      "\"Frequency of output\": 20,"
                      "\"Stencil points\":      19,"
                      "\"Relative tolerance\":  0.01,"
                      "\"Absolute tolerance\":  0.02}";

  cJSON * json = cJSON_Parse(jstr);
  assert(json);

  {
    /* Check result ... */
    psi_solver_options_t pso = {PSI_POISSON_SOLVER_INVALID};
    ifail = psi_solver_options_from_json(json, &pso);
    assert(ifail == 0);

    assert(pso.psolver  == PSI_POISSON_SOLVER_PETSC);
    assert(pso.maxits   == 200);
    assert(pso.verbose  == 2);
    assert(pso.nfreq    == 20);
    assert(pso.nstencil == 19);

    assert(fabs(pso.reltol - 0.01) < DBL_EPSILON);
    assert(fabs(pso.abstol - 0.02) < DBL_EPSILON);
  }

  return ifail;
}


