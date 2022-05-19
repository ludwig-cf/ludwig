/*****************************************************************************
 *
 *  test_gradient_d3q27.c
 *
 *  Test of field gradient computation.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>

#include "field_grad.h"
#include "gradient_d3q27.h"

__host__ int test_gradient_d3q27_d2(pe_t * pe, cs_t * cs);

/*****************************************************************************
 *
 *  test_gradient_d3q27_suite
 *
 *****************************************************************************/

__host__ int test_gradient_d3q27_suite() {

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  cs_create(pe, &cs);
  cs_init(cs);

  test_gradient_d3q27_d2(pe, cs);

  cs_free(cs);
  pe_free(pe);

  return 0;
}

/*****************************************************************************
 *
 *  test_gradient_d3q27_d2
 *
 *****************************************************************************/

__host__ int test_gradient_d3q27_d2(pe_t * pe, cs_t * cs) {

  field_t * f = NULL;
  field_grad_t * fg = NULL;

  field_options_t opts = field_options_default();

  assert(pe);
  assert(cs);

  field_create(pe, cs, NULL, "test", &opts, &f);
  field_grad_create(pe, f, 2, &fg);

  /* Gradient */
  {
    /* Probe location */
    int ic = 2;
    int jc = 2;
    int kc = 2;

    double grad[3] = {0};
    double delsq = 0.0;
    field_scalar_set(f, cs_index(cs, ic+1, jc+1, kc+1), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc,   kc+1), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc-1, kc+1), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc+1, kc  ), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc,   kc  ), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc-1, kc  ), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc+1, kc-1), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc,   kc-1), 1.0);
    field_scalar_set(f, cs_index(cs, ic+1, jc-1, kc-1), 1.0);

    /* Some test values */

    gradient_d3q27_d2(fg);

    /* Results at probe location. */

    field_grad_scalar_grad(fg, cs_index(cs, ic, jc, kc), grad);
    assert(fabs(grad[X] - 0.5) < DBL_EPSILON);
    assert(fabs(grad[Y] - 0.0) < DBL_EPSILON);
    assert(fabs(grad[Z] - 0.0) < DBL_EPSILON);

    field_grad_scalar_delsq(fg, cs_index(cs, ic, jc, kc), &delsq);
    assert(fabs(delsq   - 1.0) < DBL_EPSILON);
  }

  field_grad_free(fg);
  field_free(f);

  return 0;
}
