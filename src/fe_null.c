/****************************************************************************
 *
 *  fe_null.c
 *
 *  A 'null' free energy. Everything is zero.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <stdlib.h>


#include "pe.h"
#include "fe_null.h"

static fe_vt_t fe_null_hvt = {
  (fe_free_ft)      fe_null_free,
  (fe_target_ft)    fe_null_target,
  (fe_fed_ft)       fe_null_fed,
  (fe_mu_ft)        fe_null_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_null_str,
  (fe_str_ft)       fe_null_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_null_str_v,
  (fe_stress_v_ft)  fe_null_str_v,
  (fe_stress_v_ft)  NULL
};

static  __constant__ fe_vt_t fe_null_dvt = {
  (fe_free_ft)      NULL,
  (fe_target_ft)    NULL,
  (fe_fed_ft)       fe_null_fed,
  (fe_mu_ft)        fe_null_mu,
  (fe_mu_solv_ft)   NULL,
  (fe_str_ft)       fe_null_str,
  (fe_str_ft)       fe_null_str,
  (fe_str_ft)       NULL,
  (fe_hvector_ft)   NULL,
  (fe_htensor_ft)   NULL,
  (fe_htensor_v_ft) NULL,
  (fe_stress_v_ft)  fe_null_str_v,
  (fe_stress_v_ft)  fe_null_str_v,
  (fe_stress_v_ft)  NULL
};

/****************************************************************************
 *
 *  fe_null_create
 *
 ****************************************************************************/

__host__ int fe_null_create(pe_t * pe, fe_null_t ** p) {

  int ndevice;
  fe_null_t * fe = NULL;

  assert(pe);

  fe = (fe_null_t *) calloc(1, sizeof(fe_null_t));
  assert(fe);
  if (fe == NULL) pe_fatal(pe, "calloc(fe_null_t) failed\n");

  fe->pe = pe;
  fe->super.func = &fe_null_hvt;
  fe->super.id = FE_NULL;

  /* Allocate target memory, or alias */

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    fe->target = fe;
  }
  else {
    fe_vt_t * vt = NULL;

    tdpGetSymbolAddress((void **) &vt, tdpSymbol(fe_null_dvt));
    tdpAssert(tdpMemcpy(&fe->target->super.func, &vt, sizeof(fe_vt_t *),
			tdpMemcpyHostToDevice));
  }

  *p = fe;

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_free
 *
 *****************************************************************************/

__host__ int fe_null_free(fe_null_t * fe) {

  int ndevice;

  assert(fe);

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) tdpFree(fe->target);
  free(fe);

  return 0;
}
/****************************************************************************
 *
 *  fe_null_target
 *
 ****************************************************************************/

__host__ int fe_null_target(fe_null_t * fe, fe_t ** target) {

  assert(fe);
  assert(target);

  *target = (fe_t *) fe->target;

  return 0;
}

/****************************************************************************
 *
 *  fe_null_fed
 *
 *  The free energy density is zero.
 *
 ****************************************************************************/

__host__ __device__ int fe_null_fed(fe_null_t * fe, int index, double * fed) {

  assert(fe);

  *fed = 0.0*index;

  return 0;
}

/*****************************************************************************
 *
 *  fe_null_mu
 *
 *****************************************************************************/

__host__ __device__ int fe_null_mu(fe_null_t * fe, int index, double * mu) {

  *mu = 0.0*index;

  return 0;
}

/****************************************************************************
 *
 *  fe_null_str
 *
 ****************************************************************************/

__host__ __device__
int fe_null_str(fe_null_t * fe, int index,  double s[3][3]) {

  assert(fe);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.0*index;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_symm_str_v
 *
 *****************************************************************************/

__host__ __device__
void fe_null_str_v(fe_null_t * fe, int index, double s[3][3][NSIMDVL]) {

  assert(fe);

  for (int ia = 0; ia < 3; ia++) {
    for (int ib = 0; ib < 3; ib++) {
      int iv = 0;
      for_simd_v(iv, NSIMDVL) {
	s[ia][ib][iv] = 0.0*index;
      }
    }
  }

  return;
}
