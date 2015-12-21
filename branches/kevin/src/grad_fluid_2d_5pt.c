/*****************************************************************************
 *
 *  grad_fluid_2d_5pt.c
 *
 *  Gradient operations for 2D five point stencil.
 *
 *                    (ic, jc+1)
 *         (ic-1, jc) (ic, jc  ) (ic+1, jc)
 *                    (ic, jc-1)
 *
 *  d_x phi = [phi(ic+1,jc) - phi(ic-1,jc)] / 2
 *  d_y phi = [phi(ic,jc+1) - phi(ic,jc-1)] / 2
 *
 *  nabla^2 phi = phi(ic+1,jc) + phi(ic-1,jc) + phi(ic,jc+1) + phi(ic,jc-1)
 *              - 4 phi(ic,jc)
 *
 *  Corrections for Lees-Edwards planes in X are included.
 *
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) The University of Edinburgh (2010-2015)
 *  Contributing authors
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "field_s.h"
#include "field_grad_s.h"
#include "grad_compute_s.h"
#include "grad_fluid_2d_5pt.h"

static gc_vtable_t grad_vtable = {
  (grad_compute_free_ft) grad_fluid_2d_5pt_free,
  (grad_computer_ft)     grad_fluid_2d_5pt_computer
};

struct grad_fluid_2d_5pt_s {
  grad_compute_t super;           /* Superclass */
  le_t * le;                      /* LE reference */
  int level;                      /* What is required */
  grad_fluid_2d_5pt_t * target;   /* Target copy */
};

__host__ static int grad_d2(grad_fluid_2d_5pt_t * gc, int nf,
			    double * data,  double * grad, double * delsq);
__host__ static int grad_le_correct(grad_fluid_2d_5pt_t * gc, int nf,
				    double * data, double * grad,
				    double * delsq);

/*****************************************************************************
 *
 *  grad_fluid_2d_5pt_create
 *
 *****************************************************************************/

__host__ int grad_fluid_2d_5pt_create(le_t * le, grad_fluid_2d_5pt_t ** pobj) {

  grad_fluid_2d_5pt_t * gc = NULL;

  assert(le);
  assert(pobj);

  gc = (grad_fluid_2d_5pt_t *) calloc(1, sizeof(grad_fluid_2d_5pt_t));
  if (gc == NULL) fatal("calloc(grad_fluid_2d_5pt) failed\n");

  gc->super.vtable = &grad_vtable;
  gc->le = le;
  le_retain(le);

  *pobj = gc;

  return 0;
}

/*****************************************************************************
 *
 *  grad_fluid_2d_5pt_free
 *
 *****************************************************************************/

__host__ int grad_fluid_2d_5pt_free(grad_fluid_2d_5pt_t * gc) {

  assert(gc);

  le_free(gc->le);
  free(gc);

  return 0;
}

/*****************************************************************************
 *
 *  grad_fluid_2d_5pt_computer
 *
 *****************************************************************************/

__host__ int grad_fluid_2d_5pt_computer(grad_fluid_2d_5pt_t * gc,
					field_t * field,
					field_grad_t * grad) {
  assert(gc);
  assert(field);
  assert(grad);

  grad_d2(gc, field->nf, field->data, grad->grad, grad->delsq);
  grad_le_correct(gc, field->nf, field->data, grad->grad, grad->delsq);

  assert(0);
  /* dab */

  return 0;
}

/*****************************************************************************
 *
 *  grad_d2
 *
 *****************************************************************************/

__host__ static int grad_d2(grad_fluid_2d_5pt_t * gc, int nf,
			    double * data,  double * grad, double * delsq) {
  int nlocal[3];
  int n;
  int nhalo, nextra;
  int ic, jc;
  int zs, ys, xs;
  int icm1, icp1;
  int index, indexm1, indexp1;

  le_t * le;

  le = gc->le;
  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);
  le_nhalo(le, &nhalo);

  nextra = nhalo - 1;
  assert(nextra >= 0);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(le, ic, -1);
    icp1 = le_index_real_to_buffer(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index = le_site_index(le, ic, jc, 1);
      indexm1 = le_site_index(le, icm1, jc, 1);
      indexp1 = le_site_index(le, icp1, jc, 1);

      for (n = 0; n < nf; n++) {
	grad[3*(nf*index + n) + X]
	  = 0.5*(data[nf*indexp1 + n] - data[nf*indexm1 + n]);
	grad[3*(nf*index + n) + Y]
	  = 0.5*(data[nf*(index + ys) + n] - data[nf*(index - ys) + n]);
	grad[3*(nf*index + n) + Z] = 0.0;
	delsq[nf*index + n]
	  = data[nf*indexp1 + n] + data[nf*indexm1 + n]
	  + data[nf*(index + ys) + n] + data[nf*(index - ys) + n]
	  - 4.0*data[nf*index + n];
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_le_correct
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

static int grad_le_correct(grad_fluid_2d_5pt_t * gc, int nf,
			   double * data, double * grad, double * delsq) {

  int nplane;                             /* Number LE planes */
  int nlocal[3];
  int nh;                                 /* counter over halo extent */
  int n, np;
  int nhalo, nextra;
  int ic, jc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int xs, ys, zs;                         /* stride for 1d address */

  le_t * le;

  le = gc->le;

  le_nplane_local(le, &nplane);
  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);

  le_nhalo(le, &nhalo);
  nextra = nhalo - 1;

  assert(nlocal[Z] == 1);
  assert(nextra >= 0);

  for (np = 0; np < nplane; np++) {

    ic = le_plane_location(le, np);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(le, ic, nh-1);
      ic1 = le_index_real_to_buffer(le, ic, nh  );
      ic2 = le_index_real_to_buffer(le, ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = le_site_index(le, ic0, jc, 1);
	index   = le_site_index(le, ic1, jc, 1);
	indexp1 = le_site_index(le, ic2, jc, 1);

	for (n = 0; n < nf; n++) {
	  grad[3*(nf*index + n) + X]
	    = 0.5*(data[nf*indexp1 + n] - data[nf*indexm1 + n]);
	  grad[3*(nf*index + n) + Y]
	    = 0.5*(data[nf*(index + ys) + n] - data[nf*(index - ys) + n]);
	  grad[3*(nf*index + n) + Z] = 0.0;
	  delsq[nf*index + n]
	    = data[nf*indexp1 + n] + data[nf*indexm1 + n]
	    + data[nf*(index + ys) + n] + data[nf*(index - ys) + n]
	    - 4.0*data[nf*index + n];
	}
      }
    }

    /* Looking across the plane in the -ve x-direction. */
    ic += 1;

    for (nh = 1; nh <= nextra; nh++) {
      ic2 = le_index_real_to_buffer(le, ic, -nh+1);
      ic1 = le_index_real_to_buffer(le, ic, -nh  );
      ic0 = le_index_real_to_buffer(le, ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	indexm1 = le_site_index(le, ic0, jc, 1);
	index   = le_site_index(le, ic1, jc, 1);
	indexp1 = le_site_index(le, ic2, jc, 1);

	for (n = 0; n < nf; n++) {
	  grad[3*(nf*index + n) + X]
	    = 0.5*(data[nf*indexp1 + n] - data[nf*indexm1 + n]);
	  grad[3*(nf*index + n) + Y]
	    = 0.5*(data[nf*(index + ys) + n] - data[nf*(index - ys) + n]);
	  grad[3*(nf*index + n) + Z] = 0.0;
	  delsq[nf*index + n]
	    = data[nf*indexp1 + n] + data[nf*indexm1 + n]
	    + data[nf*(index + ys) + n] + data[nf*(index - ys) + n]
	    - 4.0*data[nf*index + n];
	}
      }
    }
    /* Next plane */
  }

  return 0;
}
