/*****************************************************************************
 *
 *  test_colloid_state_io.c
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <string.h>

#include "pe.h"
#include "colloid_state_io.h"

int test_colloid_state_io_write_buf(void);
int test_colloid_state_io_read_buf(void);
int test_colloid_state_io_write_buf_ascii(void);
int test_colloid_state_io_read_buf_ascii(void);


colloid_state_t util_test_colloid_state(void);
int util_test_colloid_state_same(const colloid_state_t * s1,
				 const colloid_state_t * s2);

/*****************************************************************************
 *
 *  test_colloid_state_io_suite
 *
 *****************************************************************************/

int test_colloid_state_io_suite(void) {

  pe_t * pe = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  test_colloid_state_io_write_buf();
  test_colloid_state_io_read_buf();
  test_colloid_state_io_write_buf_ascii();
  test_colloid_state_io_read_buf();


  pe_info(pe, "%-9s %s\n", "PASS", __FILE__);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_colloid_state_io_write_buf
 *
 *****************************************************************************/

int test_colloid_state_io_write_buf(void) {

  int ifail = 0;

  /* Minimal test. */
  {
    colloid_state_t sw = util_test_colloid_state();
    colloid_state_t sr = {};
    char buf[sizeof(colloid_state_t)] = {0};
    int same = 0;

    ifail = colloid_state_io_write_buf(&sw, buf);
    assert(ifail == 0);

    memcpy(&sr, buf, sizeof(colloid_state_t));

    same = util_test_colloid_state_same(&sw, &sr);
    if (same == 0) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_state_io_read_buf
 *
 *****************************************************************************/

int test_colloid_state_io_read_buf(void) {

  int ifail = 0;

  {
    colloid_state_t sw = util_test_colloid_state();
    colloid_state_t sr = {};
    char buf[sizeof(colloid_state_t)] = {0};
    int same = 0;

    memcpy(buf, &sw, sizeof(colloid_state_t));

    ifail = colloid_state_io_read_buf(&sr, buf);
    assert(ifail == 0);
    same  = util_test_colloid_state_same(&sw, &sr);
    if (same == 0) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_state_io_write_buf_ascii
 *
 *****************************************************************************/

int test_colloid_state_io_write_buf_ascii(void) {

  int ifail = 0;

  {
    colloid_state_t sw = util_test_colloid_state();
    char buf[1 + COLLOID_BUFSZ] = {0};

    ifail = colloid_state_io_write_buf_ascii(&sw, buf);
    assert(ifail == 0);
    assert(strlen(buf) == COLLOID_BUFSZ);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  test_colloid_state_io_read_buf_ascii
 *
 *****************************************************************************/

int test_colloid_state_io_read_buf_ascii(void) {

  int ifail = 0;

  {
    colloid_state_t sw = util_test_colloid_state();
    colloid_state_t sr = {};
    char buf[1 + COLLOID_BUFSZ] = {0};
    int same = 0;

    ifail = colloid_state_io_write_buf(&sw, buf);
    assert(ifail == 0);
    ifail = colloid_state_io_read_buf(&sr, buf);
    assert(ifail == 0);
    same = util_test_colloid_state_same(&sw, &sr);
    if (same == 0) ifail = -1;
    assert(ifail == 0);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_test_colloid_state
 *
 *  Utility to provide some test data.
 *
 *****************************************************************************/

colloid_state_t util_test_colloid_state(void) {

  colloid_state_t s = {
    .index       =  1,
    .rebuild     =  2,
    .nbonds      =  3,
    .nangles     =  4,
    .isfixedr    =  5,
    .isfixedv    =  6,
    .isfixedw    =  7,
    .isfixeds    =  8,
    .type        =  9,
    .bond        = {10, 11},
    .rng         = 12,
    .isfixedrxyz = {13, 14, 15},
    .isfixedvxyz = {16, 17, 18},
    .inter_type  = 19,
    .ioversion   = 20,
    .bc          = 21,
    .shape       = 22,
    .active      = 23,
    .magnetic    = 24,
    .attr        = 25,
    .intpad      = {26, 27, 28, 29, 30, 31, 32}, /* Pads to 32 */
    .a0          = 33.0,
    .ah          = 34.0,
    .r           = {35.0, 36.0, 37.0},
    .v           = {38.0, 39.0, 40.0},
    .w           = {41.0, 42.0, 43.0},
    .s           = {44.0, 45.0, 46.0},
    .m           = {47.0, 48.0, 49.0},
    .b1          = 50.0,
    .b2          = 51.0,
    .c           = 52.0,
    .h           = 53.0,
    .dr          = {54.0, 55.0, 56.0},
    .deltaphi    = 57.0,
    .q0          = 58.0,
    .q1          = 59.0,
    .epsilon     = 60.0,
    .deltaq0     = 61.0,
    .deltaq1     = 62.0,
    .sa          = 63.0,
    .saf         = 64.0,
    .al          = 65.0,
    .elabc       = {66.0, 67.0, 68.0},
    .quat        = {69.0, 70.0, 71.0, 72.0},
    .quatold     = {73.0, 74.0, 75.0, 76.0},
    .dpad        = {77.0, 78.0, 79.0, 80.0}      /* Pads to 80 */
  };

  return s;
}

/*****************************************************************************
 *
 *  util_test_colloid_state_same
 *
 *  Identical (incl. floating point equality).
 *
 *****************************************************************************/

int util_test_colloid_state_same(const colloid_state_t * s1,
				 const colloid_state_t * s2) {
  int same = 0;
  int sum  = 0;

  sum += (s1->index          == s2->index);
  sum += (s1->rebuild        == s2->rebuild);
  sum += (s1->nbonds         == s2->nbonds);
  sum += (s1->nangles        == s2->nangles);
  sum += (s1->isfixedr       == s2->isfixedr);
  sum += (s1->isfixedv       == s2->isfixedv);
  sum += (s1->isfixedw       == s2->isfixedw);
  sum += (s1->isfixeds       == s2->isfixeds);
  sum += (s1->type           == s2->type);
  sum += (s1->bond[0]        == s2->bond[0]);
  sum += (s1->bond[1]        == s2->bond[1]);
  sum += (s1->rng            == s2->rng);
  sum += (s1->isfixedrxyz[0] == s2->isfixedrxyz[0]);
  sum += (s1->isfixedrxyz[1] == s2->isfixedrxyz[1]);
  sum += (s1->isfixedrxyz[2] == s2->isfixedrxyz[2]);
  sum += (s1->isfixedvxyz[0] == s2->isfixedvxyz[0]);
  sum += (s1->isfixedvxyz[1] == s2->isfixedvxyz[1]);
  sum += (s1->isfixedvxyz[2] == s2->isfixedvxyz[2]);
  sum += (s1->inter_type     == s2->inter_type);
  sum += (s1->ioversion      == s2->ioversion);
  sum += (s1->bc             == s2->bc);
  sum += (s1->shape          == s2->shape);
  sum += (s1->active         == s2->active);
  sum += (s1->magnetic       == s2->magnetic);
  sum += (s1->attr           == s2->attr);
  sum += (s1->intpad[0]      == s2->intpad[0]);
  sum += (s1->intpad[1]      == s2->intpad[1]);
  sum += (s1->intpad[2]      == s2->intpad[2]);
  sum += (s1->intpad[3]      == s2->intpad[3]);
  sum += (s1->intpad[4]      == s2->intpad[4]);
  sum += (s1->intpad[5]      == s2->intpad[5]);
  sum += (s1->intpad[6]      == s2->intpad[6]);

  sum += (fabs(s1->a0         - s2->a0)         < DBL_EPSILON);
  sum += (fabs(s1->ah         - s2->ah)         < DBL_EPSILON);
  sum += (fabs(s1->r[0]       - s2->r[0])       < DBL_EPSILON);
  sum += (fabs(s1->r[1]       - s2->r[1])       < DBL_EPSILON);
  sum += (fabs(s1->r[2]       - s2->r[2])       < DBL_EPSILON);
  sum += (fabs(s1->v[0]       - s2->v[0])       < DBL_EPSILON);
  sum += (fabs(s1->v[1]       - s2->v[1])       < DBL_EPSILON);
  sum += (fabs(s1->v[2]       - s2->v[2])       < DBL_EPSILON);
  sum += (fabs(s1->w[0]       - s2->w[0])       < DBL_EPSILON);
  sum += (fabs(s1->w[1]       - s2->w[1])       < DBL_EPSILON);
  sum += (fabs(s1->w[2]       - s2->w[2])       < DBL_EPSILON);
  sum += (fabs(s1->s[0]       - s2->s[0])       < DBL_EPSILON);
  sum += (fabs(s1->s[1]       - s2->s[1])       < DBL_EPSILON);
  sum += (fabs(s1->s[2]       - s2->s[2])       < DBL_EPSILON);
  sum += (fabs(s1->m[0]       - s2->m[0])       < DBL_EPSILON);
  sum += (fabs(s1->m[1]       - s2->m[1])       < DBL_EPSILON);
  sum += (fabs(s1->m[2]       - s2->m[2])       < DBL_EPSILON);
  sum += (fabs(s1->b1         - s2->b1)         < DBL_EPSILON);
  sum += (fabs(s1->b2         - s2->b2)         < DBL_EPSILON);
  sum += (fabs(s1->c          - s2->c)          < DBL_EPSILON);
  sum += (fabs(s1->h          - s2->h)          < DBL_EPSILON);
  sum += (fabs(s1->dr[0]      - s2->dr[0])      < DBL_EPSILON);
  sum += (fabs(s1->dr[1]      - s2->dr[1])      < DBL_EPSILON);
  sum += (fabs(s1->dr[2]      - s2->dr[2])      < DBL_EPSILON);
  sum += (fabs(s1->deltaphi   - s2->deltaphi)   < DBL_EPSILON);
  sum += (fabs(s1->q0         - s2->q0)         < DBL_EPSILON);
  sum += (fabs(s1->q1         - s2->q1)         < DBL_EPSILON);
  sum += (fabs(s1->epsilon    - s2->epsilon)    < DBL_EPSILON);
  sum += (fabs(s1->deltaq0    - s2->deltaq0)    < DBL_EPSILON);
  sum += (fabs(s1->deltaq1    - s2->deltaq1)    < DBL_EPSILON);
  sum += (fabs(s1->sa         - s2->sa)         < DBL_EPSILON);
  sum += (fabs(s1->saf        - s2->saf)        < DBL_EPSILON);
  sum += (fabs(s1->al         - s2->al)         < DBL_EPSILON);
  sum += (fabs(s1->elabc[0]   - s2->elabc[0])   < DBL_EPSILON);
  sum += (fabs(s1->elabc[1]   - s2->elabc[1])   < DBL_EPSILON);
  sum += (fabs(s1->elabc[2]   - s2->elabc[2])   < DBL_EPSILON);
  sum += (fabs(s1->quat[0]    - s2->quat[0])    < DBL_EPSILON);
  sum += (fabs(s1->quat[1]    - s2->quat[1])    < DBL_EPSILON);
  sum += (fabs(s1->quat[2]    - s2->quat[2])    < DBL_EPSILON);
  sum += (fabs(s1->quat[3]    - s2->quat[3])    < DBL_EPSILON);
  sum += (fabs(s1->quatold[0] - s2->quatold[0]) < DBL_EPSILON);
  sum += (fabs(s1->quatold[1] - s2->quatold[1]) < DBL_EPSILON);
  sum += (fabs(s1->quatold[2] - s2->quatold[2]) < DBL_EPSILON);
  sum += (fabs(s1->quatold[3] - s2->quatold[3]) < DBL_EPSILON);
  sum += (fabs(s1->dpad[0]    - s2->dpad[0])    < DBL_EPSILON);
  sum += (fabs(s1->dpad[1]    - s2->dpad[1])    < DBL_EPSILON);
  sum += (fabs(s1->dpad[2]    - s2->dpad[2])    < DBL_EPSILON);
  sum += (fabs(s1->dpad[3]    - s2->dpad[3])    < DBL_EPSILON);

  if (sum == 80) same = 1;

  return same;
}
