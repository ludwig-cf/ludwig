/*****************************************************************************
 *
 *  grad_fluid_3d_7pt.c
 *
 *  Gradient operations for 3D seven point stencil.
 *
 *                        (ic, jc+1, kc)
 *         (ic-1, jc, kc) (ic, jc  , kc) (ic+1, jc, kc)
 *                        (ic, jc-1, kc)
 *
 *  ...and so in z-direction
 *
 *  d_x phi = [phi(ic+1,jc,kc) - phi(ic-1,jc,kc)] / 2
 *  d_y phi = [phi(ic,jc+1,kc) - phi(ic,jc-1,kc)] / 2
 *  d_z phi = [phi(ic,jc,kc+1) - phi(ic,jc,kc-1)] / 2
 *
 *  nabla^2 phi = phi(ic+1,jc,kc) + phi(ic-1,jc,kc)
 *              + phi(ic,jc+1,kc) + phi(ic,jc-1,kc)
 *              + phi(ic,jc,kc+1) + phi(ic,jc,kc-1)
 *              - 6 phi(ic,jc,kc)
 *
 *  Corrections for Lees-Edwards planes in X are included.
 *
 * 
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "field_s.h"
#include "field_grad_s.h"
#include "leesedwards.h"
#include "grad_fluid_3d_7pt.h"

static gc_vtable_t grad_vtable = {
  (grad_compute_free_ft) grad_fluid_3d_7pt_free,
  (grad_computer_ft)     grad_fluid_3d_7pt_computer
};

struct grad_fluid_3d_7pt_s {
  grad_compute_t super;
  le_t * le;
  int level;
  grad_fluid_3d_7pt_t * target;
};


__host__ static int grad_d2();
__host__ static int grad_d2_le_correct();

__host__ static int grad_dab(int nf, const double * field, double * dab);
__host__ static int grad_dab_le_correct(int nf, const double * field, double * dab);

/*****************************************************************************
 *
 *  grad_fluid_3d_7pt_create
 *
 *****************************************************************************/

__host__ int grad_fluid_3d_7pt_create(le_t * le, grad_fluid_3d_7pt_t ** pobj) {

  grad_fluid_3d_7pt_t * obj;

  obj = (grad_fluid_3d_7pt_t *) calloc(1, sizeof(grad_fluid_3d_7pt_t));
  if (obj == NULL) fatal("calloc(grad_fluid_3d_7pt_t) failed\n");

  obj->super.vtable = &grad_vtable;

  obj->le = le;
  le_retain(le);

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  grad_fluid_3d_7pt_free
 *
 *****************************************************************************/

__host__ int grad_fluid_3d_7pt_free(grad_fluid_3d_7pt_t * gc) {

  assert(gc);

  le_free(gc->le);
  free(gc);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_d2
 *
 *****************************************************************************/

__host__ int grad_fluid_3d_7pt_computer(grad_fluid_3d_7pt_t * gc,
					field_t * field, field_grad_t * grad) {
  assert(gc);
  assert(field);
  assert(grad);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_dab
 *
 *  This is the full gradient tensor, which actually requires more
 *  than the 7-point stencil advertised.
 *
 *  d_x d_x phi = phi(ic+1,jc,kc) - 2phi(ic,jc,kc) + phi(ic-1,jc,kc)
 *  d_x d_y phi = 0.25*[ phi(ic+1,jc+1,kc) - phi(ic+1,jc-1,kc)
 *                     - phi(ic-1,jc+1,kc) + phi(ic-1,jc-1,kc) ]
 *  d_x d_z phi = 0.25*[ phi(ic+1,jc,kc+1) - phi(ic+1,jc,kc-1)
 *                     - phi(ic-1,jc,kc+1) + phi(ic-1,jc,kc-1) ]
 *  and so on.
 *
 *  The tensor is symmetric. The 1-d compressed storage is
 *      dab[NSYMM*index + XX] etc.
 *
 *****************************************************************************/


/*****************************************************************************
 *
 *  grad_d2_driver
 *
 *  Kernel driver
 *
 *****************************************************************************/

__host__ static int grad_d2_driver(grad_fluid_3d_7pt_t * gc, field_t * field,
				   field_grad_t * grad) {
  int nlocal[3];
  int nhalo;
  int ic, jc, kc;
  int index;
  int Nall[3];

  assert(gc);
  assert(field);
  assert(grad);

  le_nhalo(le, &nhalo);
  le_nlocal(le, nlocal);

  Nall[X]=nlocal[X]+2*nhalo;
  Nall[Y]=nlocal[Y]+2*nhalo;
  Nall[Z]=nlocal[Z]+2*nhalo;

  int nSites=Nall[X]*Nall[Y]*Nall[Z];
  int nFields=nop;

  /*
  copyConstantInt1DArrayToTarget( (int*) tc_Nall, Nall, 3*sizeof(int)); 
  copyToTarget(t_field,field,nSites*nFields*sizeof(double)); 

  grad_d2_lattice TARGET_LAUNCH(nSites)(nop, t_field, t_grad, t_del2);
  */

  /* GPU version, we leave the results on the target for the next kernel.
   * C version, we bring back the results to the host (for now).
   * ultimitely GPU and C versions will follow the same pattern */

#ifndef CUDA
  copyFromTarget(grad,t_grad,3*nSites*nFields*sizeof(double)); 
  copyFromTarget(del2,t_del2,nSites*nFields*sizeof(double)); 
#endif

  return 0;
}

/*****************************************************************************
 *
 *  grad_d2_kernel
 *
 *****************************************************************************/

__global__ static void grad_d2_kernel(le_t * le,
				      const double * data,
				      double * grad,
				      double * delsq) {

  int nSites=tc_Nall[X]*tc_Nall[Y]*tc_Nall[Z];
  int index;

  TARGET_TLP(index, nSites) {
    grad_d2_site(le_t * le, data, grad, delsq, index);
  }

  return;
}

/*****************************************************************************
 *
 *  grad_d2_site
 *
 *  Called by the kernel per lattice site.
 *
 *****************************************************************************/

__device__ static void grad_d2_site(le_t * le,
				    const double * data,
				    double * grad,
				    double * delsq, 
				    const int index){

  int ic, jc, kc;

  /* Convert from index to coords */
  /* Is the coordinate within nextra ? */

  le_kernel_index_to_ijk(le, index, &ic, &jc, &kc);
  /* targetCoords3D(coords, tc_Nall, index);*/

  /* If not halo site */
  /*if (coords[0] >= (tc_nhalo-tc_nextra) && 
      coords[1] >= (tc_nhalo-tc_nextra) && 
      coords[2] >= (tc_nhalo-tc_nextra) &&
      coords[0] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&  
      coords[1] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&  
      coords[2] < tc_Nall[Z]-(tc_nhalo-tc_nextra) ){ */

  if (le_is_kernel(le, ic, jc, kc, nextra)) {

    targetCoords3D(coords,tc_Nall,index);

    /* get index +1 and -1 in X dirn */
    int indexm1 = targetIndex3D(coords[0]-1,coords[1],coords[2],tc_Nall);
    int indexp1 = targetIndex3D(coords[0]+1,coords[1],coords[2],tc_Nall);

    im1 = le_index_xneighbour(le, ic, -1);
    ip1 = le_index_xneighbour(le, ic, +1);

    im1 = le_kernel_index(le, im1, jc, kc);
    ip1 = le_kernel_index(le, ip1, jc, kc);

    int n;
    int ys=tc_Nall[Z];

    for (n = 0; n < nop; n++) {
      grad[3*(nop*index + n) + X]
	= 0.5*(data[nop*indexp1 + n] - data[nop*indexm1 + n]);
      grad[3*(nop*index + n) + Y]
	= 0.5*(data[nop*(index + ys) + n] - data[nop*(index - ys) + n]);
      grad[3*(nop*index + n) + Z]
	= 0.5*(data[nop*(index + 1) + n] - data[nop*(index - 1) + n]);
      delsq[nop*index + n]
	= data[nop*indexp1      + n] + data[nop*indexm1      + n]
	+ data[nop*(index + ys) + n] + data[nop*(index - ys) + n]
	+ data[nop*(index + 1)  + n] + data[nop*(index - 1)  + n]
	- 6.0*data[nop*index + n];
    }
  }
  
  return;
}


/*****************************************************************************
 *
 *  grad_d2_le_correct
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

__host__ static int grad_d2_le_correct(grad_fluid_3d_7pt_t * gc, int nf,
				       const double * data,
				       double * grad,
				       double * del2) {
  int nlocal[3];
  int nh;                                 /* counter over halo extent */
  int n, np;
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int zs, ys, xs;                         /* strides for 1d address */

  le_t * le;

  le = gc->le;
  le_nplane_local(le, &nplane);
  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);

  for (np = 0; np < nplane; np++) {

    ic = le_plane_location(le, np);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(le, ic, nh-1);
      ic1 = le_index_real_to_buffer(le, ic, nh  );
      ic2 = le_index_real_to_buffer(le, ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(le, ic0, jc, kc);
	  index   = le_site_index(le, ic1, jc, kc);
	  indexp1 = le_site_index(le, ic2, jc, kc);

	  for (n = 0; n < nop; n++) {
	    grad[3*(nop*index + n) + X]
	      = 0.5*(field[nop*indexp1 + n] - field[nop*indexm1 + n]);
	    grad[3*(nop*index + n) + Y]
	      = 0.5*(field[nop*(index + ys) + n]
		     - field[nop*(index - ys) + n]);
	    grad[3*(nop*index + n) + Z]
	      = 0.5*(field[nop*(index + 1) + n] - field[nop*(index - 1) + n]);
	    del2[nop*index + n]
	      = field[nop*indexp1      + n] + field[nop*indexm1      + n]
	      + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	      + field[nop*(index + 1)  + n] + field[nop*(index - 1)  + n]
	      - 6.0*field[nop*index + n];
	  }
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
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(le, ic0, jc, kc);
	  index   = le_site_index(le, ic1, jc, kc);
	  indexp1 = le_site_index(le, ic2, jc, kc);

	  for (n = 0; n < nop; n++) {
	    grad[3*(nop*index + n) + X]
	      = 0.5*(field[nop*indexp1 + n] - field[nop*indexm1 + n]);
	    grad[3*(nop*index + n) + Y]
	      = 0.5*(field[nop*(index + ys) + n]
		     - field[nop*(index - ys) + n]);
	    grad[3*(nop*index + n) + Z]
	      = 0.5*(field[nop*(index + 1) + n] - field[nop*(index - 1) + n]);
	    del2[nop*index + n]
	      = field[nop*indexp1      + n] + field[nop*indexm1      + n]
	      + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	      + field[nop*(index + 1)  + n] + field[nop*(index - 1)  + n]
	      - 6.0*field[nop*index + n];
	  }
	}
      }
    }
    /* Next plane */
  }

  return 0;
}

/*****************************************************************************
 *
 *  gradient_dab_compute
 *
 *****************************************************************************/

__host__ static int grad_dab(grad_fluid_3d_7pt_t * gc, int nf,
			     const double * data, double * dab) {

  int nlocal[3];
  int nhalo;
  int nextra;
  int n;
  int ic, jc, kc;
  int zs, ys, xs;
  int icm1, icp1;
  int index, indexm1, indexp1;

  le_t * le;

  assert(nf == 1); /* PENDING is this required? */
  assert(data);
  assert(dab);

  le = gc->le;

  le_nhalo(le, &nhalo);
  nextra = nhalo - 1;
  assert(nextra >= 0);

  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(le, ic, -1);
    icp1 = le_index_real_to_buffer(le, ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = le_site_index(le, ic, jc, kc);
	indexm1 = le_site_index(le, icm1, jc, kc);
	indexp1 = le_site_index(le, icp1, jc, kc);

	for (n = 0; n < nf; n++) {
	  dab[NSYMM*(nf*index + n) + XX]
	    = data[nf*indexp1 + n] + data[nf*indexm1 + n]
	    - 2.0*data[nf*index + n];
	  dab[NSYMM*(nf*index + n) + XY] = 0.25*
	    (data[nf*(indexp1 + ys) + n] - data[nf*(indexp1 - ys) + n]
	     - data[nf*(indexm1 + ys) + n] + data[nf*(indexm1 - ys) + n]);
	  dab[NSYMM*(nf*index + n) + XZ] = 0.25*
	    (data[nf*(indexp1 + 1) + n] - data[nf*(indexp1 - 1) + n]
	     - data[nf*(indexm1 + 1) + n] + data[nf*(indexm1 - 1) + n]);

	  dab[NSYMM*(nf*index + n) + YY]
	    = data[nf*(index + ys) + n] + data[nf*(index - ys) + n]
	    - 2.0*data[nf*index + n];
	  dab[NSYMM*(nf*index + n) + YZ] = 0.25*
	    (data[nf*(index + ys + 1) + n] - data[nf*(index + ys - 1) + n]
	   - data[nf*(index - ys + 1) + n] + data[nf*(index - ys - 1) + n]
	     );

	  dab[NSYMM*(nf*index + n) + ZZ]
	    = data[nf*(index + 1)  + n] + data[nf*(index - 1)  + n]
	    - 2.0*data[nf*index + n];
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  grad_dab_le_correct
 *
 *****************************************************************************/

__host__ static int grad_dab_le_correct(grad_fluid_3d_7pt_t * gc, int nf,
					const double * data,
					double * dab) {

  int nlocal[3];
  int nhalo;
  int nextra;
  int nh;                                 /* counter over halo extent */
  int n, np;
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int zs, ys, xs;                         /* strides for 1d address */

  le_t * le;

  assert(gc);

  le = gc->le;

  le_nplane_local(le, &nplane);
  le_nhalo(le, &nhalo);
  le_nlocal(le, nlocal);
  le_strides(le, &xs, &ys, &zs);

  nextra = nhalo - 1;
  assert(nextra >= 0);

  for (np = 0; np < nplane; np++) {

    ic = le_plane_location(le, np);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(le, ic, nh-1);
      ic1 = le_index_real_to_buffer(le, ic, nh  );
      ic2 = le_index_real_to_buffer(le, ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(le, ic0, jc, kc);
	  index   = le_site_index(le, ic1, jc, kc);
	  indexp1 = le_site_index(le, ic2, jc, kc);

	  for (n = 0; n < nf; n++) {
	    dab[NSYMM*(nf*index + n) + XX]
	      = data[nf*indexp1 + n] + data[nf*indexm1 + n]
	      - 2.0*data[nf*index + n];
	    dab[NSYMM*(nf*index + n) + XY] = 0.25*
	      (data[nf*(indexp1 + ys) + n] - data[nf*(indexp1 - ys) + n]
	       - data[nf*(indexm1 + ys) + n] + data[nf*(indexm1 - ys) + n]);
	    dab[NSYMM*(nf*index + n) + XZ] = 0.25*
	      (data[nf*(indexp1 + 1) + n] - data[nf*(indexp1 - 1) + n]
	       - data[nf*(indexm1 + 1) + n] + data[nf*(indexm1 - 1) + n]);

	    dab[NSYMM*(nf*index + n) + YY]
	      = data[nf*(index + ys) + n] + data[nf*(index - ys) + n]
	      - 2.0*data[nf*index + n];
	    dab[NSYMM*(nf*index + n) + YZ] = 0.25*
	      (data[nf*(index + ys + 1) + n] - data[nf*(index + ys - 1) + n]
	     - data[nf*(index - ys + 1) + n] + data[nf*(index - ys - 1) + n]
	       );

	    dab[NSYMM*(nf*index + n) + ZZ]
	      = data[nf*(index + 1)  + n] + data[nf*(index - 1)  + n]
	      - 2.0*data[nf*index + n];
	  }
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
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(le, ic0, jc, kc);
	  index   = le_site_index(le, ic1, jc, kc);
	  indexp1 = le_site_index(le, ic2, jc, kc);

	  for (n = 0; n < nf; n++) {
	    dab[NSYMM*(nf*index + n) + XX]
	      = data[nf*indexp1 + n] + data[nf*indexm1 + n]
	      - 2.0*data[nf*index + n];
	    dab[NSYMM*(nf*index + n) + XY] = 0.25*
	      (data[nf*(indexp1 + ys) + n] - data[nf*(indexp1 - ys) + n]
	       - data[nf*(indexm1 + ys) + n] + data[nf*(indexm1 - ys) + n]);
	    dab[NSYMM*(nf*index + n) + XZ] = 0.25*
	      (data[nf*(indexp1 + 1) + n] - data[nf*(indexp1 - 1) + n]
	       - data[nf*(indexm1 + 1) + n] + data[nf*(indexm1 - 1) + n]);

	    dab[NSYMM*(nf*index + n) + YY]
	      = data[nf*(index + ys) + n] + data[nf*(index - ys) + n]
	      - 2.0*data[nf*index + n];
	    dab[NSYMM*(nf*index + n) + YZ] = 0.25*
	      (data[nf*(index + ys + 1) + n] - data[nf*(index + ys - 1) + n]
	     - data[nf*(index - ys + 1) + n] + data[nf*(index - ys - 1) + n]
	       );

	    dab[NSYMM*(nf*index + n) + ZZ]
	      = data[nf*(index + 1)  + n] + data[nf*(index - 1)  + n]
	      - 2.0*data[nf*index + n];
	  }
	}
      }
    }
    /* Next plane */
  }

  return 0;
}
