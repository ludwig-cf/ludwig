/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid.c
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
 *  Corrections for Lees-Edwards planes and plane wall in X are included.
 *
 *  $Id: gradient_3d_7pt_fluid.c,v 1.2 2010-10-15 12:40:03 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "wall.h"
#include "gradient_3d_7pt_fluid.h"
#include "string.h"
#include "field_s.h"
#include "field_grad_s.h"
#include "targetDP.h"
#include "timer.h"

__targetHost__ static void gradient_3d_7pt_fluid_operator(const int nop, 
					   const double * field,
					   double * t_field,
					   double * grad,
					   double * t_grad,
					   double * delsq,
					   double * t_delsq,
			     const int nextra);
__targetHost__ static void gradient_3d_7pt_fluid_le_correction(const int nop,
						const double * field,
						double * grad,
						double * delsq,
						const int nextra);
__targetHost__ static void gradient_3d_7pt_fluid_wall_correction(const int nop,
						  const double * field,
						  double * grad,
						  double * delsq,
						  const int nextra);

__targetHost__ static int gradient_dab_le_correct(int nf, const double * field, double * dab);
__targetHost__ static int gradient_dab_compute(int nf, const double * field, double * dab);

/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_d2
 *
 *****************************************************************************/

__targetHost__ int gradient_3d_7pt_fluid_d2(const int nop, 
			     const double * field,
			     double * t_field,
			     double * grad,
			     double * t_grad,
			     double * delsq,
			     double * t_delsq
			     ) {

  int nextra;

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_3d_7pt_fluid_operator(nop, field, t_field, grad, t_grad,
				 delsq, t_delsq, nextra);
  gradient_3d_7pt_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_3d_7pt_fluid_wall_correction(nop, field, grad, delsq, nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_d4
 *
 *  Higher derivatives are obtained by using the same operation
 *  on appropriate field.
 *
 *****************************************************************************/

__targetHost__ int gradient_3d_7pt_fluid_d4(const int nop, 
			     const double * field,
			     double * t_field,
			     double * grad,
			     double * t_grad,
			     double * delsq,
			     double * t_delsq
			     ) {

  int nextra;

  nextra = coords_nhalo() - 2;
  assert(nextra >= 0);

  assert(field);
  assert(grad);
  assert(delsq);

  gradient_3d_7pt_fluid_operator(nop, field, t_field, grad, t_grad, delsq, t_delsq, nextra);
  gradient_3d_7pt_fluid_le_correction(nop, field, grad, delsq, nextra);
  gradient_3d_7pt_fluid_wall_correction(nop, field, grad, delsq, nextra);

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

__targetHost__ int gradient_3d_7pt_fluid_dab(const int nf, 
			     const double * field,
			      double * dab){

  assert(nf == 1); /* Scalars only */

  gradient_dab_compute(nf, field, dab);
  gradient_dab_le_correct(nf, field, dab);

  return 0;
}


//__targetConst__ int tc_Nall[3];
//__targetConst__ int tc_nhalo;
//__targetConst__ int tc_nextra;


/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_operator
 *
 *****************************************************************************/

static __target__ void gradient_3d_7pt_fluid_operator_site(const int nop,
					   const double * t_field,
					   double * t_grad,
						double * t_del2, 
						const int baseIndex){




  int iv=0;
  int i;

    int coordschunk[3][VVL];
    int coords[3];

    __targetILP__(iv){      
      for(i=0;i<3;i++){
	targetCoords3D(coords,tc_Nall,baseIndex+iv);
	coordschunk[i][iv]=coords[i];
      }      
    }

  
#if VVL == 1    
/*restrict operation to the interior lattice sites*/ 
    if (coords[0] >= (tc_nhalo-tc_nextra) && 
	coords[1] >= (tc_nhalo-tc_nextra) && 
	coords[2] >= (tc_nhalo-tc_nextra) &&
	coords[0] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&  
	coords[1] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&  
	coords[2] < tc_Nall[Z]-(tc_nhalo-tc_nextra) )
#endif

{ 

      /* work out which sites in this chunk should be included */
      int includeSite[VVL];
      __targetILP__(iv) includeSite[iv]=0;
      
      int coordschunk[3][VVL];
      
      __targetILP__(iv){
	for(i=0;i<3;i++){
	  targetCoords3D(coords,tc_Nall,baseIndex+iv);
	  coordschunk[i][iv]=coords[i];
	}
      }
      
      __targetILP__(iv){
	
	if ((coordschunk[0][iv] >= (tc_nhalo-tc_nextra) &&
	     coordschunk[1][iv] >= (tc_nhalo-tc_nextra) &&
	     coordschunk[2][iv] >= (tc_nhalo-tc_nextra) &&
	     coordschunk[0][iv] < tc_Nall[X]-(tc_nhalo-tc_nextra) &&
	     coordschunk[1][iv] < tc_Nall[Y]-(tc_nhalo-tc_nextra)  &&
	     coordschunk[2][iv] < tc_Nall[Z]-(tc_nhalo-tc_nextra)))
	  
	  includeSite[iv]=1;
      }



  int indexm1[VVL];
  int indexp1[VVL];

    //get index +1 and -1 in X dirn
    __targetILP__(iv) indexm1[iv] = targetIndex3D(coordschunk[0][iv]-1,coordschunk[1][iv],
						      coordschunk[2][iv],tc_Nall);
    __targetILP__(iv) indexp1[iv] = targetIndex3D(coordschunk[0][iv]+1,coordschunk[1][iv],
						      coordschunk[2][iv],tc_Nall);

      
    int n;
    int ys=tc_Nall[Z];
    for (n = 0; n < nop; n++) {

      __targetILP__(iv){ 
	if(includeSite[iv])
	  t_grad[FGRDADR(tc_nSites,nop,baseIndex+iv,n,X)]
	    = 0.5*(t_field[FLDADR(tc_nSites,nop,indexp1[iv],n)] - t_field[FLDADR(tc_nSites,nop,indexm1[iv],n)]); 
      }
      
      __targetILP__(iv){ 
	if(includeSite[iv])
	  t_grad[FGRDADR(tc_nSites,nop,baseIndex+iv,n,Y)]
	    = 0.5*(t_field[FLDADR(tc_nSites,nop,baseIndex+iv+ys,n)] - t_field[FLDADR(tc_nSites,nop,baseIndex+iv-ys,n)]);
      }
      
      __targetILP__(iv){ 
	if(includeSite[iv])
	  t_grad[FGRDADR(tc_nSites,nop,baseIndex+iv,n,Z)]
	    = 0.5*(t_field[FLDADR(tc_nSites,nop,baseIndex+iv+1,n)] - t_field[FLDADR(tc_nSites,nop,baseIndex+iv-1,n)]);
      }
      
      __targetILP__(iv){ 
	if(includeSite[iv])
	  t_del2[FLDADR(tc_nSites,nop,baseIndex+iv,n)]
	    = t_field[FLDADR(tc_nSites,nop,indexp1[iv],n)] + t_field[FLDADR(tc_nSites,nop,indexm1[iv],n)]
	    + t_field[FLDADR(tc_nSites,nop,baseIndex+iv+ys,n)] + t_field[FLDADR(tc_nSites,nop,baseIndex+iv-ys,n)]
	    + t_field[FLDADR(tc_nSites,nop,baseIndex+iv+1,n)] + t_field[FLDADR(tc_nSites,nop,baseIndex+iv-1,n)]
	    - 6.0*t_field[FLDADR(tc_nSites,nop,baseIndex+iv,n)];
      }
      
    }
    
  }
  
  return;
}


static __targetEntry__ void gradient_3d_7pt_fluid_operator_lattice(const int nop,
					   const double * t_field,
					   double * t_grad,
						double * t_del2){



  

  int baseIndex;
  __targetTLP__(baseIndex,tc_nSites){
    gradient_3d_7pt_fluid_operator_site(nop,t_field,t_grad,t_del2,baseIndex);
  }


}


static void gradient_3d_7pt_fluid_operator(const int nop,
					   const double * field,
					   double * t_field,
					   double * grad,
					   double * t_grad,
					   double * del2,
					   double * t_del2,
					   const int nextra) {
  int nlocal[3];
  int nhalo;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);


  int Nall[3];
  Nall[X]=nlocal[X]+2*nhalo;  Nall[Y]=nlocal[Y]+2*nhalo;  Nall[Z]=nlocal[Z]+2*nhalo;


  int nSites=Nall[X]*Nall[Y]*Nall[Z];

  int nFields=nop;

  //start constant setup
  copyConstToTarget(tc_Nall,Nall, 3*sizeof(int)); 
  copyConstToTarget(&tc_nhalo,&nhalo, sizeof(int)); 
  copyConstToTarget(&tc_nextra,&nextra, sizeof(int)); 
  copyConstToTarget(&tc_nSites,&nSites, sizeof(int));

  //end constant setup

  #ifndef KEEPFIELDONTARGET
  copyToTarget(t_field,field,nSites*nFields*sizeof(double)); 
  #endif

  TIMER_start(TIMER_PHI_GRAD_KERNEL);	       
   gradient_3d_7pt_fluid_operator_lattice __targetLaunch__(nSites) 
  (nop,t_field,t_grad,t_del2);
  targetSynchronize();
  TIMER_stop(TIMER_PHI_GRAD_KERNEL);	       
   

  //for GPU version, we leave the results on the target for the next kernel.
  //for C version, we bring back the results to the host (for now).
  //ultimitely GPU and C versions will follow the same pattern
  #ifndef KEEPFIELDONTARGET
  copyFromTarget(grad,t_grad,3*nSites*nFields*sizeof(double)); 
  copyFromTarget(del2,t_del2,nSites*nFields*sizeof(double)); 
  #endif

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_le_correction
 *
 *  Additional gradient calculations near LE planes to account for
 *  sliding displacement.
 *
 *****************************************************************************/

__targetHost__ static void gradient_3d_7pt_fluid_le_correction(const int nop,
						const double * field,
						double * grad,
						double * del2,
						const int nextra) {
  int nlocal[3];
  int nhalo;
  int nh;                                 /* counter over halo extent */
  int n;
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  ys = (nlocal[Z] + 2*nhalo);

  for (nplane = 0; nplane < le_get_nplane_local(); nplane++) {

    ic = le_plane_location(nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(ic, nh-1);
      ic1 = le_index_real_to_buffer(ic, nh  );
      ic2 = le_index_real_to_buffer(ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(ic0, jc, kc);
	  index   = le_site_index(ic1, jc, kc);
	  indexp1 = le_site_index(ic2, jc, kc);

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
      ic2 = le_index_real_to_buffer(ic, -nh+1);
      ic1 = le_index_real_to_buffer(ic, -nh  );
      ic0 = le_index_real_to_buffer(ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(ic0, jc, kc);
	  index   = le_site_index(ic1, jc, kc);
	  indexp1 = le_site_index(ic2, jc, kc);

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

  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_wall_correction
 *
 *  Correct the gradients near the X boundary wall, if necessary.
 *
 *****************************************************************************/

__targetHost__ static  void gradient_3d_7pt_fluid_wall_correction(const int nop,
						  const double * field,
						  double * grad,
						  double * del2,
						  const int nextra) {
  int nlocal[3];
  int nhalo;
  int n;
  int jc, kc;
  int index;
  int xs, ys;

  double fb;                    /* Extrapolated value of field at boundary */
  double gradm1, gradp1;        /* gradient terms */
  double rk;                    /* Fluid free energy parameter (reciprocal) */
  double * c;                   /* Solid free energy parameters C */
  double * h;                   /* Solid free energy parameters H */

  if (wall_at_edge(X) == 0) return;

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = (nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  assert(wall_at_edge(Y) == 0);
  assert(wall_at_edge(Z) == 0);

  /* This enforces C = 0 and H = 0, ie., neutral wetting, as there
   * is currently no mechanism to obtain the free energy parameters. */

  c = (double *) malloc(nop*sizeof(double));
  h = (double *) malloc(nop*sizeof(double));

  if (c == NULL) fatal("malloc(c) failed\n");
  if (h == NULL) fatal("malloc(h) failed\n");

  for (n = 0; n < nop; n++) {
    c[n] = 0.0;
    h[n] = 0.0;
  }
  rk = 0.0;

  if (cart_coords(X) == 0) {

    /* Correct the lower wall */

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(1, jc, kc);

	for (n = 0; n < nop; n++) {
	  gradp1 = field[nop*(index + xs) + n] - field[nop*index + n];
	  fb = field[nop*index + n] - 0.5*gradp1;
	  gradm1 = -(c[n]*fb + h[n])*rk;
	  grad[3*(nop*index + n) + X] = 0.5*(gradp1 - gradm1);
	  del2[nop*index + n]
	    = gradp1 - gradm1
	    + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	    + field[nop*(index + 1 ) + n] + field[nop*(index - 1 ) + n] 
	    - 4.0*field[nop*index + n];
	}

	/* Next site */
      }
    }
  }

  if (cart_coords(X) == cart_size(X) - 1) {

    /* Correct the upper wall */

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = coords_index(nlocal[X], jc, kc);

	for (n = 0; n < nop; n++) {
	  gradm1 = field[nop*index + n] - field[nop*(index - xs) + n];
	  fb = field[nop*index + n] + 0.5*gradm1;
	  gradp1 = -(c[n]*fb + h[n])*rk;
	  grad[3*(nop*index + n) + X] = 0.5*(gradp1 - gradm1);
	  del2[nop*index + n]
	    = gradp1 - gradm1
	    + field[nop*(index + ys) + n] + field[nop*(index - ys) + n]
	    + field[nop*(index + 1 ) + n] + field[nop*(index - 1 ) + n]
	    - 4.0*field[nop*index + n];
	}
	/* Next site */
      }
    }
  }

  free(c);
  free(h);

  return;
}

/*****************************************************************************
 *
 *  gradient_dab_compute
 *
 *****************************************************************************/

__targetHost__ static int gradient_dab_compute(int nf, const double * field, double * dab) {

  int nlocal[3];
  int nhalo;
  int nextra;
  int n;
  int ic, jc, kc;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  assert(nf == 1);
  assert(field);
  assert(dab);

  nextra = coords_nhalo() - 1;
  assert(nextra >= 0);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  ys = nlocal[Z] + 2*nhalo;

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    icm1 = le_index_real_to_buffer(ic, -1);
    icp1 = le_index_real_to_buffer(ic, +1);
    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	index = le_site_index(ic, jc, kc);
	indexm1 = le_site_index(icm1, jc, kc);
	indexp1 = le_site_index(icp1, jc, kc);

	for (n = 0; n < nf; n++) {
	  dab[NSYMM*(nf*index + n) + XX]
	    = field[nf*indexp1 + n] + field[nf*indexm1 + n]
	    - 2.0*field[nf*index + n];
	  dab[NSYMM*(nf*index + n) + XY] = 0.25*
	    (field[nf*(indexp1 + ys) + n] - field[nf*(indexp1 - ys) + n]
	     - field[nf*(indexm1 + ys) + n] + field[nf*(indexm1 - ys) + n]);
	  dab[NSYMM*(nf*index + n) + XZ] = 0.25*
	    (field[nf*(indexp1 + 1) + n] - field[nf*(indexp1 - 1) + n]
	     - field[nf*(indexm1 + 1) + n] + field[nf*(indexm1 - 1) + n]);

	  dab[NSYMM*(nf*index + n) + YY]
	    = field[nf*(index + ys) + n] + field[nf*(index - ys) + n]
	    - 2.0*field[nf*index + n];
	  dab[NSYMM*(nf*index + n) + YZ] = 0.25*
	    (field[nf*(index + ys + 1) + n] - field[nf*(index + ys - 1) + n]
	   - field[nf*(index - ys + 1) + n] + field[nf*(index - ys - 1) + n]
	     );

	  dab[NSYMM*(nf*index + n) + ZZ]
	    = field[nf*(index + 1)  + n] + field[nf*(index - 1)  + n]
	    - 2.0*field[nf*index + n];
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  gradient_dab_le_correct
 *
 *****************************************************************************/

__targetHost__ static int gradient_dab_le_correct(int nf, const double * field,
				   double * dab) {

  int nlocal[3];
  int nhalo;
  int nextra;
  int nh;                                 /* counter over halo extent */
  int n;
  int nplane;                             /* Number LE planes */
  int ic, jc, kc;
  int ic0, ic1, ic2;                      /* x indices involved */
  int index, indexm1, indexp1;            /* 1d addresses involved */
  int ys;                                 /* y-stride for 1d address */

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  ys = (nlocal[Z] + 2*nhalo);

  nextra = nhalo - 1;
  assert(nextra >= 0);

  for (nplane = 0; nplane < le_get_nplane_local(); nplane++) {

    ic = le_plane_location(nplane);

    /* Looking across in +ve x-direction */
    for (nh = 1; nh <= nextra; nh++) {
      ic0 = le_index_real_to_buffer(ic, nh-1);
      ic1 = le_index_real_to_buffer(ic, nh  );
      ic2 = le_index_real_to_buffer(ic, nh+1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(ic0, jc, kc);
	  index   = le_site_index(ic1, jc, kc);
	  indexp1 = le_site_index(ic2, jc, kc);

	  for (n = 0; n < nf; n++) {
	    dab[NSYMM*(nf*index + n) + XX]
	      = field[nf*indexp1 + n] + field[nf*indexm1 + n]
	      - 2.0*field[nf*index + n];
	    dab[NSYMM*(nf*index + n) + XY] = 0.25*
	      (field[nf*(indexp1 + ys) + n] - field[nf*(indexp1 - ys) + n]
	       - field[nf*(indexm1 + ys) + n] + field[nf*(indexm1 - ys) + n]);
	    dab[NSYMM*(nf*index + n) + XZ] = 0.25*
	      (field[nf*(indexp1 + 1) + n] - field[nf*(indexp1 - 1) + n]
	       - field[nf*(indexm1 + 1) + n] + field[nf*(indexm1 - 1) + n]);

	    dab[NSYMM*(nf*index + n) + YY]
	      = field[nf*(index + ys) + n] + field[nf*(index - ys) + n]
	      - 2.0*field[nf*index + n];
	    dab[NSYMM*(nf*index + n) + YZ] = 0.25*
	      (field[nf*(index + ys + 1) + n] - field[nf*(index + ys - 1) + n]
	     - field[nf*(index - ys + 1) + n] + field[nf*(index - ys - 1) + n]
	       );

	    dab[NSYMM*(nf*index + n) + ZZ]
	      = field[nf*(index + 1)  + n] + field[nf*(index - 1)  + n]
	      - 2.0*field[nf*index + n];
	  }
	}
      }
    }

    /* Looking across the plane in the -ve x-direction. */
    ic += 1;

    for (nh = 1; nh <= nextra; nh++) {
      ic2 = le_index_real_to_buffer(ic, -nh+1);
      ic1 = le_index_real_to_buffer(ic, -nh  );
      ic0 = le_index_real_to_buffer(ic, -nh-1);

      for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
	for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	  indexm1 = le_site_index(ic0, jc, kc);
	  index   = le_site_index(ic1, jc, kc);
	  indexp1 = le_site_index(ic2, jc, kc);

	  for (n = 0; n < nf; n++) {
	    dab[NSYMM*(nf*index + n) + XX]
	      = field[nf*indexp1 + n] + field[nf*indexm1 + n]
	      - 2.0*field[nf*index + n];
	    dab[NSYMM*(nf*index + n) + XY] = 0.25*
	      (field[nf*(indexp1 + ys) + n] - field[nf*(indexp1 - ys) + n]
	       - field[nf*(indexm1 + ys) + n] + field[nf*(indexm1 - ys) + n]);
	    dab[NSYMM*(nf*index + n) + XZ] = 0.25*
	      (field[nf*(indexp1 + 1) + n] - field[nf*(indexp1 - 1) + n]
	       - field[nf*(indexm1 + 1) + n] + field[nf*(indexm1 - 1) + n]);

	    dab[NSYMM*(nf*index + n) + YY]
	      = field[nf*(index + ys) + n] + field[nf*(index - ys) + n]
	      - 2.0*field[nf*index + n];
	    dab[NSYMM*(nf*index + n) + YZ] = 0.25*
	      (field[nf*(index + ys + 1) + n] - field[nf*(index + ys - 1) + n]
	     - field[nf*(index - ys + 1) + n] + field[nf*(index - ys - 1) + n]
	       );

	    dab[NSYMM*(nf*index + n) + ZZ]
	      = field[nf*(index + 1)  + n] + field[nf*(index - 1)  + n]
	      - 2.0*field[nf*index + n];
	  }
	}
      }
    }
    /* Next plane */
  }

  return 0;
}
