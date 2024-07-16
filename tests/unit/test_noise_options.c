/*****************************************************************************
 *
 *  test_noise_options.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2024 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>

#include "noise_options.h"

int test_noise_options_default(void);
int test_noise_options_seed(void);
int test_noise_options_seed_nextra(void);
int test_noise_options_valid(void);

/*****************************************************************************
 *
 *  test_noise_options_suite
 *
 *****************************************************************************/

int test_noise_options_suite(void) {

  int ifail = 0;
  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_noise_options_default();
  test_noise_options_seed();
  test_noise_options_seed_nextra();
  test_noise_options_valid();

  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_options_default
 *
 *****************************************************************************/

int test_noise_options_default(void) {

  int ifail = 0;

  noise_options_t opts = noise_options_default();
  assert(opts.seed   == 13);
  assert(opts.nextra == 1);
  assert(strcmp(opts.filestub, "noise") == 0);

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_options_seed
 *
 *****************************************************************************/

int test_noise_options_seed(void) {

  int ifail = 0;

  /* A value */
  {
    unsigned int seed = 17;
    noise_options_t opts = noise_options_seed(seed);
    assert(opts.seed   == seed);
    assert(opts.nextra == 1);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_options_seed_nextra
 *
 *****************************************************************************/

int test_noise_options_seed_nextra(void) {

  int ifail = 0;

  /* Non-default values */
  {
    unsigned int seed = 37;
    int nextra = 2;
    noise_options_t opts = noise_options_seed_nextra(seed, nextra);
    assert(opts.seed   == seed);
    assert(opts.nextra == nextra);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_noise_options_valid
 *
 *****************************************************************************/

int test_noise_options_valid(void) {

  int ifail = 0;

  /* Default */
  {
    noise_options_t opts = noise_options_default();
    ifail = noise_options_valid(&opts);
    assert(ifail != 0);
  }

  /* Improper initialisation of opts */
  {
    noise_options_t opts = {.seed = 13};
    ifail = noise_options_valid(&opts);
    assert(ifail == 0);
  }

  /* Bad seed */
  {
    noise_options_t opts = noise_options_seed(0);
    ifail = noise_options_valid(&opts);
    assert(ifail == 0);
  }

  /* Bad nextra */
  {
    noise_options_t opts = noise_options_seed_nextra(17, -1);
    ifail = noise_options_valid(&opts);
    assert(ifail == 0);
  }

  return ifail;
}
