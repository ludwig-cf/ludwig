/*****************************************************************************
 *
 *  test_ch.c
 *
 *  Cahn-Hilliard solver (surfactants)
 *
 *  Edinburgh Soft Matter and Statistics Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019-2024 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "pe.h"
#include "coords.h"
#include "cahn_hilliard.h"

int test_ch_create(pe_t * pe);

/*****************************************************************************
 *
 *  test_ch_suite
 *
 *****************************************************************************/

int test_ch_suite(void) {

  int ndevice;
  pe_t * pe = NULL;

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  if (ndevice) {
    pe_info(pe, "SKIP     ./unit/test_ch\n");
  }
  else {
    test_ch_create(pe);

    pe_info(pe, "PASS     ./unit/test_ch\n");
  }

  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_ch_create
 *
 *****************************************************************************/

int test_ch_create(pe_t * pe) {

  const int nfield = 2;  /* phi, psi   */ 
  const double m1 = 1.0; /* mobility 1 */
  const double m2 = 2.0; /* mobility 2 */

  cs_t * cs = NULL;
  ch_t * ch = NULL;
  ch_info_t info = {0};

  assert(pe);

  cs_create(pe, &cs);
  cs_init(cs);

  /* info */
  info.nfield = nfield;
  info.mobility[0] = m1;
  info.mobility[1] = m2;

  ch_create(pe, cs, info, &ch);
  assert(ch);
  assert(ch->pe);
  assert(ch->cs);
  assert(ch->flux);
  assert(ch->target);

  assert(ch->info->nfield == nfield);
  assert(fabs(ch->info->mobility[0] - m1) < DBL_EPSILON);
  assert(fabs(ch->info->mobility[1] - m2) < DBL_EPSILON);

  ch_free(ch);
  cs_free(cs);

  return 0;
}
