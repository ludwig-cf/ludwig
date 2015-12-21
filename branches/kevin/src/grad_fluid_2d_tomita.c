/*****************************************************************************
 *
 *  grad_fluid_2d_tomita.c
 *
 *  Gradient operations for 2D stencil following Tomita Prog. Theor. Phys.
 *  {\bf 85} 47 (1991).
 *
 *  del^2 phi(r) = (1 / 1 + 2epsilon) [
 *                 \sum phi(r + a) + epsilon \sum phi(r + b)
 *                 - 4(1 + epsilon) phi(r) ]
 *
 *  where a represent 4 nearest neighbours, b represent 4 next-nearest
 *  neighbours, and epsilon is a parameter. If epsilon = 0, this
 *  collapses to a five-point stencil. The optimum value epsilon is
 *  reckoned to be epsilon = 0.5.
 *
 *  In the same spirit, the gradient is approximated as:
 *
 *  d_x phi = 0.5*(1 / 1 + 2epsilon) * [
 *            epsilon * ( phi(ic+1, jc-1) - phi(ic-1, jc-1) )
 *                    + ( phi(ic+1, jc  ) - phi(ic-1, jc  ) )
 *          + epsilon * ( phi(ic+1, jc+1) - phi(ic-1, jc+1) ) ]
 *
 *  and likewise for d_y phi. Again, this gives 5-point stencil for
 *  epsilon = 0. For the gradient, the optimum value is epsilon = 0.25,
 *  which gives weights consistent with Wolfram J. Stat. Phys {\bf 45},
 *  471 (1986). See also Shan PRE {\bf 73} 047701 (2006).
 *
 *  Corrections for Lees-Edwards planes and plane wall in X are included.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "field_s.h"
#include "field_grad_s.h"
#include "grad_compute_s.h"
#include "grad_fluid_2d_tomita.h"

#define EPSILON_DEFAULT (0.5)
#define EPSILON1_DEFAULT (0.25)

static gc_vtable_t grad_vtable = {
  (grad_compute_free_ft) grad_fluid_2d_tomita_free,
  (grad_computer_ft)     grad_fluid_2d_tomita_computer
};

struct grad_fluid_2d_tomita_s {
  grad_compute_t super;              /* Superclass */
  le_t * le;                         /* LE reference */
  double epsilon;                    /* Parameter */
  double epsilon1;                   /* Parameter */
  grad_fluid_2d_tomita_t * target;   /* Target copy */
};

__host__ static int grad_2d(grad_fluid_2d_tomita_t * gc, int nf,
			    double * data, double * grad, double * delsq);
__host__ static int grad_le_correct(grad_fluid_2d_tomita_t * gc, int nf,
				    double * data, double * grad,
				    double * delsq);

/*****************************************************************************
 *
 *  grad_fluid_2d_tomita_create
 *
 *****************************************************************************/

__host__ int grad_fluid_2d_tomita_create(le_t * le,
					 grad_fluid_2d_tomita_t ** pobj) {
  grad_fluid_2d_tomita_t * gc = NULL;

  assert(le);
  assert(pobj);

  gc = (grad_fluid_2d_tomita_t *) calloc(1, sizeof(grad_fluid_2d_tomita_t));
  if (gc == NULL) fatal("calloc(grad_fluid_2d_tomita_t) failed\n");

  gc->super.vtable = &grad_vtable;
  gc->epsilon  = EPSILON_DEFAULT;
  gc->epsilon1 = EPSILON1_DEFAULT;

  gc->le = le;
  le_retain(le);

  *pobj = gc;

  return 0;
}

/*****************************************************************************
 *
 *  grad_fluid_2d_tomita_free
 *
 *****************************************************************************/

__host__ int grad_fluid_2d_tomita_free(grad_fluid_2d_tomita_t * gc) {

  assert(gc);

  le_free(gc->le);
  free(gc);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_2d_tomita_fluid_d2
 *
 *****************************************************************************/

__host__ int grad_fluid_2d_tomita_computer(grad_fluid_2d_tomita_t * gc,
					   field_t * field,
					   field_grad_t * grad) {
  assert(gc);
  assert(field);
  assert(grad);

  grad_2d(gc, field->nf, field->data, grad->grad, grad->delsq);
  grad_le_correct(gc, field->nf, field->data, grad->grad, grad->delsq);

  assert(0);
  /* etc */

  return 0;
}

/*****************************************************************************
 *
 *  grad_d2
 *
 *****************************************************************************/

__host__ static int grad_2d(grad_fluid_2d_tomita_t * gc, int nf, double * data,
			    double * grad, double * delsq) {

  int nlocal[3];
  int n;
  int nhalo, nextra;
  int ic, jc;
  int zs, ys, xs;
  int icm1, icp1;
  int index, indexm1, indexp1;

  double rfactor;
  double rfactor1;

  le_t * le;

  assert(gc);

  le = gc->le;

  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);
  le_nhalo(le, &nhalo);
  nextra = nhalo - 1;
  assert(nextra >= 0);

  rfactor  = 1.0 / (1.0 + 2.0*gc->epsilon);
  rfactor1 = 1.0 / (1.0 + 2.0*gc->epsilon1);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(le, ic, -1);
    icp1 = le_index_real_to_buffer(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      index   = le_site_index(le, ic, jc, 1);
      indexm1 = le_site_index(le, icm1, jc, 1);
      indexp1 = le_site_index(le, icp1, jc, 1);

      for (n = 0; n < nf; n++) {
	grad[3*(nf*index + n) + X] = 0.5*rfactor1*
	  (data[nf*indexp1 + n] - data[nf*indexm1 + n]
	   + gc->epsilon1*
	   (data[nf*(indexp1 + ys) + n] - data[nf*(indexm1 + ys) + n] +
	    data[nf*(indexp1 - ys) + n] - data[nf*(indexm1 - ys) + n]));
	grad[3*(nf*index + n) + Y] = 0.5*rfactor1*
	  (data[nf*(index + ys) + n] - data[nf*(index - ys) + n]
	   + gc->epsilon1*
	   (data[nf*(indexp1 + ys) + n] - data[nf*(indexp1 - ys) + n] +
	    data[nf*(indexm1 + ys) + n] - data[nf*(indexm1 - ys) + n]));
	grad[3*(nf*index + n) + Z] = 0.0;
	delsq[nf*index + n] =
	  rfactor*(data[nf*indexp1 + n] +
		   data[nf*indexm1 + n] +
		   data[nf*(index + ys) + n] +
		   data[nf*(index - ys) + n] +
		   gc->epsilon*(data[nf*(indexp1 + ys) + n] +
			       data[nf*(indexp1 - ys) + n] +
			       data[nf*(indexm1 + ys) + n] +
			       data[nf*(indexm1 - ys) + n])
		   - 4.0*(1.0 + gc->epsilon)*data[nf*index + n]);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_le_correcton
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

__host__ static int grad_le_correct(grad_fluid_2d_tomita_t * gc, int nf,
				    double * data, double * grad,
				    double * delsq) {

  int nplane;                             /* Number LE planes */
  int nlocal[3];
  int nh;                                 /* counter over halo extent */
  int nhalo, nextra;
  int n, np;
  int ic, jc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int zs, ys, xs;                         /* strides for 1d address */

  double rfactor;

  le_t * le;

  assert(gc);

  le = gc->le;
  le_nplane_local(le, &nplane);
  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);
  le_nhalo(le, &nhalo);
  nextra = nhalo - 1;

  assert(nlocal[Z] == 1);
  assert(nextra >= 0);

  rfactor = 1.0 / (1.0 + 2.0*gc->epsilon);

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
	  grad[3*(nf*index + n) + X] = 0.5*rfactor*
	    (data[nf*indexp1 + n] - data[nf*indexm1 + n]
	     + gc->epsilon*
	     (data[nf*(indexp1 + ys) + n] - data[nf*(indexm1 + ys) + n] +
	      data[nf*(indexp1 - ys) + n] - data[nf*(indexm1 - ys) + n]));
	  grad[3*(nf*index + n) + Y] = 0.5*rfactor*
	    (data[nf*(index + ys) + n] - data[nf*(index - ys) + n]
	     + gc->epsilon*
	     (data[nf*(indexp1 + ys) + n] - data[nf*(indexp1 - ys) + n] +
	      data[nf*(indexm1 + ys) + n] - data[nf*(indexm1 - ys) + n]));
	  grad[3*(nf*index + n) + Z] = 0.0;

	  delsq[nf*index + n] =
	    rfactor*(data[nf*indexp1 + n] +
		     data[nf*indexm1 + n] +
		     data[nf*(index + ys) + n] +
		     data[nf*(index - ys) + n] +
		     gc->epsilon*(data[nf*(indexp1 + ys) + n] +
				  data[nf*(indexp1 - ys) + n] +
				  data[nf*(indexm1 + ys) + n] +
				  data[nf*(indexm1 - ys) + n])
		     - 4.0*(1.0 + gc->epsilon)*data[nf*index + n]);
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
	  grad[3*(nf*index + n) + X] = 0.5*rfactor*
	    (data[nf*indexp1 + n] - data[nf*indexm1 + n]
	     + gc->epsilon*
	     (data[nf*(indexp1 + ys) + n] - data[nf*(indexm1 + ys) + n] +
	      data[nf*(indexp1 - ys) + n] - data[nf*(indexm1 - ys) + n]));
	  grad[3*(nf*index + n) + Y] = 0.5*rfactor*
	    (data[nf*(index + ys) + n] - data[nf*(index - ys) + n]
	     + gc->epsilon*
	     (data[nf*(indexp1 + ys) + n] - data[nf*(indexp1 - ys) + n] +
	      data[nf*(indexm1 + ys) + n] - data[nf*(indexm1 - ys) + n]));
	  grad[3*(nf*index + n) + Z] = 0.0;

	  delsq[nf*index + n] =
	    rfactor*(data[nf*indexp1 + n] +
		     data[nf*indexm1 + n] +
		     data[nf*(index + ys) + n] +
		     data[nf*(index - ys) + n] +
		     gc->epsilon*(data[nf*(indexp1 + ys) + n] +
				  data[nf*(indexp1 - ys) + n] +
				  data[nf*(indexm1 + ys) + n] +
				  data[nf*(indexm1 - ys) + n])
		     - 4.0*(1.0 + gc->epsilon)*data[nf*index + n]);

	}
      }
    }
    /* Next plane */
  }

  return 0;
}
