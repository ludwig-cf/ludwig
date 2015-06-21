/*****************************************************************************
 *
 *  fe_null.c
 *
 *  All quantities return zero. One can actaully call the routines
 *  (except create and free) with a NULL fe_t pointer.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2015 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "fe_s.h"
#include "fe_null.h"

struct fe_null_s {
  fe_t super;
};

static fe_vtable_t fe_null_vtable = {
  (fe_free_ft)    fe_null_free,
  (fe_fed_ft)     fe_null_fed,
  (fe_mu_ft)      fe_null_mu,
  (fe_str_ft)     fe_null_str,
  (fe_mu_solv_ft) fe_null_mu_solv,
  (fe_hvector_ft) fe_null_hvector,
  (fe_htensor_ft) fe_null_htensor
};

/*****************************************************************************
 *
 *  fe_null_create
 *
 *****************************************************************************/

__host__ int fe_null_create(fe_null_t ** pobj) {

  fe_null_t * fe = NULL;

  assert(pobj);

  fe = (fe_null_t *) calloc(1, sizeof(fe_null_t));
  if (fe == NULL) fatal("calloc(fe_null_t) failed\n");

  fe->super.vtable = &fe_null_vtable;

  *pobj = fe;

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_free
 *
 *****************************************************************************/

__host__ int fe_null_free(fe_null_t * fe) {

  assert(fe);

  free(fe);

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_fed
 *
 *****************************************************************************/

__host__ __device__
int fe_null_fed(fe_null_t * fe, int index, double * fed) {

  *fed = 0.0;

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_mu
 *
 *****************************************************************************/

__host__ __device__
int fe_null_mu(fe_null_t * fe, int index, double * mu) {

  *mu = 0.0;

  return 0;
}

/****************************************************************************
 *
 *  fe_null_str
 *
 *  Default chemical stress is zero.
 *
 ****************************************************************************/

__host__ __device__
int fe_null_str(fe_null_t * fe, int index, double s[3][3]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_mu_solv
 *
 *****************************************************************************/

__host__ __device__
int fe_null_mu_solv(fe_null_t * fe, int index, int nt, double * mu) {

  int n;

  for (n = 0; n < nt; n++) {
    mu[n] = 0.0;
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_hvector
 *
 *****************************************************************************/

__host__ __device__
int fe_null_hvector(fe_null_t * fe, int index, double h[3]) {

  h[0] = 0.0;
  h[1] = 0.0;
  h[2] = 0.0;

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_htensor
 *
 *****************************************************************************/

__host__ __device__
int fe_null_htensor(fe_null_t * fe, int index, double h[3][3]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] = 0.0;
    }
  }

  return 0;
}

