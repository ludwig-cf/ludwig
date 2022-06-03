/*****************************************************************************
 *
 *  test_lc_anchoring.c
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

#include "pe.h"
#include "lc_anchoring.h"

int test_lc_anchoring_type_from_string(void);

/*****************************************************************************
 *
 *  test_lc_anchoring_suite
 *
 *****************************************************************************/

int test_lc_anchoring_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_lc_anchoring_type_from_string();

  pe_info(pe, "PASS     ./unit/test_lc_anchoring\n");

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_lc_anchoring_type_from_string
 *
 *****************************************************************************/

int test_lc_anchoring_type_from_string(void) {

  lc_anchoring_enum_t lc = LC_ANCHORING_INVALID;

  lc = lc_anchoring_type_from_string("normal");
  assert(lc == LC_ANCHORING_NORMAL);

  lc = lc_anchoring_type_from_string("planar");
  assert(lc == LC_ANCHORING_PLANAR);

  lc = lc_anchoring_type_from_string("fixed");
  assert(lc == LC_ANCHORING_FIXED);

  lc = lc_anchoring_type_from_string("rubbish");
  assert(lc == LC_ANCHORING_INVALID);

  return (lc != LC_ANCHORING_INVALID);
}
