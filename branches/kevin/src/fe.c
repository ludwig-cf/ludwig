/*****************************************************************************
 *
 *  free_energy
 *
 *  This is an 'abstract' free energy interface.
 *
 *  Any concrete implementation should implement one or more of the
 *  following functions to override the default 'null' free energy:
 *
 *  fe_fed_ft      computes free energy density
 *  fe_mu_ft       computes one or more chemical potentials
 *  fe_str_ft      computes thermodynamic stress
 *  fe_mu_solv_ft  computes one or more solvation chemical potentials
 *  fe_hvector_ft  computes molecular field (vector order parameters)
 *  fe_htensor_ft  computes molecular field (tensor order parameters)
 *
 *  The choice of free energy is also related to the form of the order
 *  parameter, and sets the highest order of spatial derivatives of
 *  the order parameter required for the free energy calculations.
 *
 *  $Id: free_energy.c,v 1.16 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009-2014 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>

#include "fe_s.h"

static int fe_fed_null(fe_t * fe, int index, double * fed);
static int fe_mu_null(fe_t * fe,  int index, double * mu);
static int fe_pth_null(fe_t * fe, int index, double s[3][3]);
static int fe_mu_solv_null(fe_t * fe, int index, int n, double * mu);
static int fe_hvector_null(fe_t * fe, int index, double h[3]);
static int fe_htensor_null(fe_t * fe, int index, double h[3][3]);

/****************************************************************************
 *
 *  fe_create
 *
 ****************************************************************************/

int fe_create(fe_t ** p) {

  fe_t * fe = NULL;

  fe = (fe_t *) calloc(1, sizeof(fe_t));
  if (fe == NULL) fatal("calloc(fe_t) failed\n");

  fe->fed     = fe_fed_null;
  fe->mu      = fe_mu_null;
  fe->stress  = fe_pth_null;
  fe->mu_solv = fe_mu_solv_null;
  fe->hvector = fe_hvector_null;
  fe->htensor = fe_htensor_null;

  *p = fe;

  return 0;
}

/*****************************************************************************
 *
 *  fe_free
 *
 *****************************************************************************/

int fe_free(fe_t * fe) {

  assert(fe);

  free(fe);

  return 0;
}

/****************************************************************************
 *
 *  fe_register_cb
 *
 *  'Override' call back function pointers if non-NULL provided.
 *
 ****************************************************************************/

int fe_register_cb(fe_t * fe, void * child, fe_fed_ft fed, fe_mu_ft mu,
		   fe_str_ft str, fe_mu_solv_ft mu_solv,
		   fe_hvector_ft hvector, fe_htensor_ft htensor) {

  assert(fe);
  assert(child);

  fe->child = child;
  if (fed) fe->fed = fed;
  if (mu) fe->mu = mu;
  if (str) fe->stress = str;
  if (mu_solv) fe->mu_solv = mu_solv;
  if (hvector) fe->hvector = hvector;
  if (htensor) fe->htensor = htensor;

  return 0;
}

/*****************************************************************************
 *
 *  fe_child
 *
 *  Caller is responsible for dealing with child pointer correctly.
 *
 *****************************************************************************/

int fe_child(fe_t * fe, void ** child) {

  assert(fe);
  assert(child);

  *child = fe->child;

  return 0;
}

/*****************************************************************************
 *
 *  fe_fed
 *
 *****************************************************************************/

int fe_fed(fe_t * fe, int index, double * fed) {

  assert(fe);

  return fe->fed(fe, index, fed);
}

/*****************************************************************************
 *
 *  fe_mu
 *
 *****************************************************************************/

int fe_mu(fe_t * fe, int index, double * mu) {

  assert(fe);

  return fe->mu(fe, index, mu);
}

/*****************************************************************************
 *
 *  fe_str
 *
 *****************************************************************************/

int fe_str(fe_t * fe, int index, double s[3][3]) {

  assert(fe);

  return fe->stress(fe, index, s);
}

/*****************************************************************************
 *
 *  fe_hvector
 *
 *****************************************************************************/

int fe_hvector(fe_t * fe, int index, double h[3]) {

  assert(fe);

  return fe->hvector(fe, index, h);
}

/*****************************************************************************
 *
 *  fe_fed_null
 *
 *****************************************************************************/

int fe_fed_null(fe_t * fe, int index, double * fed) {

  assert(fe);

  *fed = 0.0;

  return 0;
}

/*****************************************************************************
 *
 *  fe_mu_null
 *
 *****************************************************************************/

int fe_mu_null(fe_t * fe, int index, double * mu) {

  assert(fe);

  *mu = 0.0;

  return 0;
}

/****************************************************************************
 *
 *  fe_pth_null
 *
 *  Default chemical stress is zero.
 *
 ****************************************************************************/

static int fe_pth_null(fe_t * fe, int index, double s[3][3]) {

  int ia, ib;

  assert(fe);

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_mu_solv_set
 *
 *****************************************************************************/

static int fe_mu_solv_null(fe_t * fe, int index, int nt, double * mu) {

  int n;

  assert(fe);

  for (n = 0; n < nt; n++) {
    mu[n] = 0.0;
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_hvector_null
 *
 *****************************************************************************/

static int fe_hvector_null(fe_t * fe, int index, double h[3]) {

  h[0] = 0.0;
  h[1] = 0.0;
  h[2] = 0.0;

  return 0;
}

/*****************************************************************************
 *
 *  fe_htensor_null
 *
 *****************************************************************************/

static int fe_htensor_null(fe_t * fe, int index, double h[3][3]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      h[ia][ib] = 0.0;
    }
  }

  return 0;
}
