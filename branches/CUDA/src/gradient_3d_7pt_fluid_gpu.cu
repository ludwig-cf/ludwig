/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_gpu.c
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
 * Adapted to run on GPU: Alan Gray
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include "common_gpu.h"
#include "colloids.h"
#include "site_map.h"
#include "gradient_3d_7pt_fluid_gpu.h"
#include "util.h"

#define NOP 5
#define NITERATION 40

__constant__ double q_0_cd;
__constant__ double kappa0_cd;
__constant__ double kappa1_cd;
__constant__ double w_cd;
__constant__ double amplitude_cd;
__constant__ double e_cd[3][3][3];
__constant__ int noffset_cd[3];
__constant__ double d_cd[3][3];

extern "C" double blue_phase_q0(void);
extern "C" double blue_phase_kappa0(void);
extern "C" double blue_phase_kappa1(void);
extern "C" double colloids_q_tensor_w(void);
extern "C" void coords_nlocal_offset(int n[3]);
extern "C" double blue_phase_amplitude_compute(void);

__global__ void gradient_3d_7pt_solid_gpu_d(int nop, int nhalo, 
						     int N_d[3], 
						     const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     char * site_map_status_d,
					    char * colloid_map_d,
					    double * colloid_r_d,
					    int nextra);



void phi_gradients_compute_gpu()
{

  int nop,N[3],Ngradcalc[3],nhalo;



  nhalo = coords_nhalo();
  int nextra=nhalo-1;
  coords_nlocal(N); 
  nop = phi_nop();
  

  /* set up CUDA grid */
  
  Ngradcalc[X]=N[X]+2*nextra;
  Ngradcalc[Y]=N[Y]+2*nextra;
  Ngradcalc[Z]=N[Z]+2*nextra;
  
  int nblocks=(Ngradcalc[X]*Ngradcalc[Y]*Ngradcalc[Z]+DEFAULT_TPB-1)
    /DEFAULT_TPB;
  
  /* run the kernel */
  /* gradient_3d_7pt_fluid_operator_gpu_d<<<nblocks,DEFAULT_TPB>>> */
  /*   (nop, nhalo, N_d, phi_site_d,grad_phi_site_d,delsq_phi_site_d, */
  /*    le_index_real_to_buffer_d,nextra);  */


  // FROM blue_phase.c
  double q0_;        /* Pitch = 2pi / q0_ */
  double a0_;        /* Bulk free energy parameter A_0 */
  double gamma_;     /* Controls magnitude of order */
  double kappa0_;    /* Elastic constant \kappa_0 */
  double kappa1_;    /* Elastic constant \kappa_1 */
  
  double xi_;        /* effective molecular aspect ratio (<= 1.0) */
  double redshift_;  /* redshift parameter */
  double rredshift_; /* reciprocal */
  double zeta_;      /* Apolar activity parameter \zeta */
  
  double epsilon_; /* Dielectric anisotropy (e/12pi) */
  
  double electric_[3]; /* Electric field */



  double w;                                 /* Anchoring strength parameter */
  double amplitude;


  q0_=blue_phase_q0();
  kappa0_=blue_phase_kappa0();
  kappa1_=blue_phase_kappa1();
  w = colloids_q_tensor_w();
  amplitude = blue_phase_amplitude_compute();

  int noffset[3];
  coords_nlocal_offset(noffset);

 cudaMemcpyToSymbol(q_0_cd, &q0_, sizeof(double), 0, cudaMemcpyHostToDevice);  
cudaMemcpyToSymbol(kappa0_cd, &kappa0_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(kappa1_cd, &kappa1_, sizeof(double), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(e_cd, e_, 3*3*3*sizeof(double), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(noffset_cd, noffset, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 

 cudaMemcpyToSymbol(w_cd, &w, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(amplitude_cd, &amplitude, sizeof(double), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(d_cd, d_, 3*3*sizeof(double), 0, cudaMemcpyHostToDevice); 


  gradient_3d_7pt_solid_gpu_d<<<nblocks,DEFAULT_TPB>>>
    (nop, nhalo, N_d, phi_site_d,grad_phi_site_d,delsq_phi_site_d,
     site_map_status_d,colloid_map_d, colloid_r_d,nextra); 
  
  cudaThreadSynchronize();
  
  return;
  
  }


/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_operator_gpu_d
 *
 *****************************************************************************/
__global__ void gradient_3d_7pt_fluid_operator_gpu_d(int nop, int nhalo, 
						     int N_d[3], 
						     const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     int * le_index_real_to_buffer_d,
						     int nextra) {

  int n;
  int ys;
  int icm1, icp1;
  int index, indexm1, indexp1;

  int threadIndex, Nall[3], Ngradcalc[3],ii, jj, kk;

  ys = N_d[Z] + 2*nhalo;

  Nall[X]=N_d[X]+2*nhalo;
  Nall[Y]=N_d[Y]+2*nhalo;
  Nall[Z]=N_d[Z]+2*nhalo;

  int nsites=Nall[X]*Nall[Y]*Nall[Z];

  Ngradcalc[X]=N_d[X]+2*nextra;
  Ngradcalc[Y]=N_d[Y]+2*nextra;
  Ngradcalc[Z]=N_d[Z]+2*nextra;

  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  /* Avoid going beyond problem domain */
  if (threadIndex < Ngradcalc[X]*Ngradcalc[Y]*Ngradcalc[Z])
    {

      /* calculate index from CUDA thread index */

      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Ngradcalc);
      index = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra,
				     kk+nhalo-nextra,Nall);      
      

      /* icm1 = le_index_real_to_buffer(ic, -1); */
      /* icp1 = le_index_real_to_buffer(ic, +1); */
      /*le_index_real_to_buffer_d holds -1 then +1 translation values */
      icm1=le_index_real_to_buffer_d[ii+nhalo-nextra];
      icp1=le_index_real_to_buffer_d[Nall[X]+ii+nhalo-nextra];      

      indexm1 = get_linear_index_gpu_d(icm1,jj+nhalo-nextra,kk+nhalo-nextra,Nall);
      indexp1 = get_linear_index_gpu_d(icp1,jj+nhalo-nextra,kk+nhalo-nextra,Nall);


      for (n = 0; n < nop; n++) { 

	  grad_d[X*nsites*nop+n*nsites+index]
	    = 0.5*(field_d[nsites*n+indexp1] - field_d[nsites*n+indexm1]);
	  grad_d[Y*nsites*nop+n*nsites+index]
	    = 0.5*(field_d[nsites*n+(index + ys)] - field_d[nsites*n+(index - ys)]);
	  grad_d[Z*nsites*nop+n*nsites+index]
	    = 0.5*(field_d[nsites*n+(index + 1)] - field_d[nsites*n+(index - 1)]);
	  del2_d[n*nsites + index]
	    = field_d[nsites*n+indexp1] + field_d[nsites*n+indexm1]
	    + field_d[nsites*n+(index + ys)] + field_d[nsites*n+(index - ys)]
	    + field_d[nsites*n+(index + 1)] + field_d[nsites*n+(index - 1)]
	    - 6.0*field_d[nsites*n+index];
		  } 


   } 
  return;
}

__device__ static void gradient_bcs_gpu_d(const double kappa0, const double kappa1, const int dn[3],
			 double dq[NOP][3], double bc[NOP][NOP][3]) {

  double kappa2;

  kappa2 = kappa0 + kappa1;

  /* XX equation */

  bc[XX][XX][X] =  kappa0*dn[X]*dq[XX][X];
  bc[XX][XY][X] = -kappa1*dn[Y]*dq[XY][X];
  bc[XX][XZ][X] = -kappa1*dn[Z]*dq[XZ][X];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;

  bc[XX][XX][Y] = kappa1*dn[Y]*dq[XX][Y];
  bc[XX][XY][Y] = kappa0*dn[X]*dq[XY][Y];
  bc[XX][XZ][Y] = 0.0;
  bc[XX][YY][Y] = 0.0;
  bc[XX][YZ][Y] = 0.0;

  bc[XX][XX][Z] = kappa1*dn[Z]*dq[XX][Z];
  bc[XX][XY][Z] = 0.0;
  bc[XX][XZ][Z] = kappa0*dn[X]*dq[XZ][Z];
  bc[XX][YY][Z] = 0.0;
  bc[XX][YZ][Z] = 0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0*dn[Y]*dq[XX][X];
  bc[XY][XY][X] =  kappa2*dn[X]*dq[XY][X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1*dn[Y]*dq[YY][X];
  bc[XY][YZ][X] = -kappa1*dn[Z]*dq[YZ][X];

  bc[XY][XX][Y] = -kappa1*dn[X]*dq[XX][Y];
  bc[XY][XY][Y] =  kappa2*dn[Y]*dq[XY][Y];
  bc[XY][XZ][Y] = -kappa1*dn[Z]*dq[XZ][Y];
  bc[XY][YY][Y] =  kappa0*dn[X]*dq[YY][Y];
  bc[XY][YZ][Y] =  0.0;

  bc[XY][XX][Z] = 0.0;
  bc[XY][XY][Z] = 2.0*kappa1*dn[Z]*dq[XY][Z];
  bc[XY][XZ][Z] = kappa0*dn[Y]*dq[XZ][Z];
  bc[XY][YY][Z] = 0.0;
  bc[XY][YZ][Z] = kappa0*dn[X]*dq[YZ][Z];

  /* XZ equation */

  bc[XZ][XX][X] =  kappa2*dn[Z]*dq[XX][X];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2*dn[X]*dq[XZ][X];
  bc[XZ][YY][X] =  kappa1*dn[Z]*dq[YY][X];
  bc[XZ][YZ][X] = -kappa1*dn[Y]*dq[YZ][X];

  bc[XZ][XX][Y] = 0.0;
  bc[XZ][XY][Y] = kappa0*dn[Z]*dq[XY][Y];
  bc[XZ][XZ][Y] = 2.0*kappa1*dn[Y]*dq[XZ][Y];
  bc[XZ][YY][Y] = 0.0;
  bc[XZ][YZ][Y] = kappa0*dn[X]*dq[YZ][Y];

  bc[XZ][XX][Z] = -kappa2*dn[X]*dq[XX][Z];
  bc[XZ][XY][Z] = -kappa1*dn[Y]*dq[XY][Z];
  bc[XZ][XZ][Z] =  kappa2*dn[Z]*dq[XZ][Z];
  bc[XZ][YY][Z] = -kappa0*dn[X]*dq[YY][Z];
  bc[XZ][YZ][Z] =  0.0;

  /* YY equation */

  bc[YY][XX][X] = 0.0;
  bc[YY][XY][X] = kappa0*dn[Y]*dq[XY][X];
  bc[YY][XZ][X] = 0.0;
  bc[YY][YY][X] = kappa1*dn[X]*dq[YY][X];
  bc[YY][YZ][X] = 0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1*dn[X]*dq[XY][Y];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0*dn[Y]*dq[YY][Y];
  bc[YY][YZ][Y] = -kappa1*dn[Z]*dq[YZ][Y];

  bc[YY][XX][Z] = 0.0;
  bc[YY][XY][Z] = 0.0;
  bc[YY][XZ][Z] = 0.0;
  bc[YY][YY][Z] = kappa1*dn[Z]*dq[YY][Z];
  bc[YY][YZ][Z] = kappa0*dn[Y]*dq[YZ][Z];

  /* YZ equation */

  bc[YZ][XX][X] = 0.0;
  bc[YZ][XY][X] = kappa0*dn[Z]*dq[XY][X];
  bc[YZ][XZ][X] = kappa0*dn[Y]*dq[XZ][X];
  bc[YZ][YY][X] = 0.0;
  bc[YZ][YZ][X] = 2.0*kappa1*dn[X]*dq[YZ][X];

  bc[YZ][XX][Y] =  kappa1*dn[Z]*dq[XX][Y];
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1*dn[X]*dq[XZ][Y];
  bc[YZ][YY][Y] =  kappa2*dn[Z]*dq[YY][Y];
  bc[YZ][YZ][Y] =  kappa2*dn[Y]*dq[YZ][Y];

  bc[YZ][XX][Z] = -kappa0*dn[Y]*dq[XX][Z];
  bc[YZ][XY][Z] = -kappa1*dn[X]*dq[XY][Z];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa2*dn[Y]*dq[YY][Z];
  bc[YZ][YZ][Z] =  kappa2*dn[Z]*dq[YZ][Z];

  return;
}


__device__ static int util_gaussian_gpu_d(double a[NOP][NOP], double xb[NOP]) {

  int i, j, k;
  int ifail = 0;
  int iprow;
  int ipivot[NOP];

  double tmp;

  iprow = -1;
  for (k = 0; k < NOP; k++) {
    ipivot[k] = -1;
  }

  for (k = 0; k < NOP; k++) {

    /* Find pivot row */
    tmp = 0.0;
    for (i = 0; i < NOP; i++) {
      if (ipivot[i] == -1) {
	if (fabs(a[i][k]) >= tmp) {
	  tmp = fabs(a[i][k]);
	  iprow = i;
	}
      }
    }
    ipivot[k] = iprow;

    /* divide pivot row by the pivot element a[iprow][k] */

    //TO DO
    //if (a[iprow][k] == 0.0) {
    //  fatal("Gaussian elimination failed in gradient calculation\n");
    // }

    tmp = 1.0 / a[iprow][k];
    for (j = k; j < NOP; j++) {
      a[iprow][j] *= tmp;
    }
    xb[iprow] *= tmp;

    /* Subtract the pivot row (scaled) from remaining rows */

    for (i = 0; i < NOP; i++) {
      if (ipivot[i] == -1) {
	tmp = a[i][k];
	for (j = k; j < NOP; j++) {
	  a[i][j] -= tmp*a[iprow][j];
	}
	xb[i] -= tmp*xb[iprow];
      }
    }
  }

  /* Now do the back substitution */

  for (i = NOP - 1; i > -1; i--) {
    iprow = ipivot[i];
    tmp = xb[iprow];
    for (k = i + 1; k < NOP; k++) {
      tmp -= a[iprow][k]*xb[ipivot[k]];
    }
    xb[iprow] = tmp;
  }

  return ifail;
}


__device__ void colloids_q_boundary_normal_gpu_d(const int di[3],
						 double dn[3], int Nall[3], int nhalo, int nextra, int ii, int jj, int kk, char *site_map_status_d,					    char * colloid_map_d,
						 double * colloid_r_d) {
  int ia, index1;
  int index;
  double rd;

  index = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra,kk+nhalo-nextra,Nall);

  index1 = get_linear_index_gpu_d(ii+nhalo-nextra-di[X],jj+nhalo-nextra-di[Y],kk+nhalo-nextra-di[Z],Nall);


  if (site_map_status_d[index1] == COLLOID){


    //TO DO check this when nhalo is not 2
      dn[X] = 1.0*(noffset_cd[X] + ii);
      dn[Y] = 1.0*(noffset_cd[Y] + jj);
      dn[Z] = 1.0*(noffset_cd[Z] + kk);

      for (ia = 0; ia < 3; ia++) {
	//dn[ia] -= pc->s.r[ia];
	dn[ia] -= colloid_r_d[3*colloid_map_d[index]+ia];
      }

    rd=sqrt(dn[X]*dn[X] + dn[Y]*dn[Y] + dn[Z]*dn[Z]);
    rd = 1.0/rd;

    for (ia = 0; ia < 3; ia++) {
      dn[ia] *= rd;
    }
  }
  else {
    /* Assume di is the true outward normal (e.g., flat wall) */
    for (ia = 0; ia < 3; ia++) {
      dn[ia] = 1.0*di[ia];
    }
  }

  //if (index==127332) printf("YYY %f %f %f\n",dn[X],dn[Y],dn[Z]);

  //if (index==127332) printf("YYY %d %f\n",colloid_map_d[index],colloid_r_d[colloid_map_d[index]]);



  return;
}




__global__ void gradient_3d_7pt_solid_gpu_d(int nop, int nhalo, 
						     int N_d[3], 
						     const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     char * site_map_status_d,
					    char * colloid_map_d,
					    double * colloid_r_d,
					    int nextra
					    ) {
  int n;
  int ys;
  int icm1, icp1;
  int index, index1;
  int niterate;

  int threadIndex, Nall[3], Ngradcalc[3],str[3],ii, jj, kk, ia, ib, ig, ih;
 
  char status[6];
  int mask[6];
  int ns,n1,n2;

  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  const int normal[6] = {X, X, Y, Y, Z, Z};
  const int nsolid[6] = {0, 1, 0, 1, 0, 1};

 double c[6][3][3];                        /* Constant terms in BC. */
 double q0[3][3];                          /* Prefered surface Q_ab */
 double qs[3][3];                          /* 'Surface' Q_ab */
 double b[NOP];                            /* RHS / unknown */
 double a[NOP][NOP];                       /* Matrix for linear system */
 double dq[NOP][3];                        /* normal/tangential gradients */
 double bc[NOP][NOP][3];                   /* Terms in boundary condition */
 double gradn[NOP][3][2];                  /* Partial gradients */
  double dn[3];                             /* Unit normal. */
 double tmp;

  ys = N_d[Z] + 2*nhalo;

  Nall[X]=N_d[X]+2*nhalo;
  Nall[Y]=N_d[Y]+2*nhalo;
  Nall[Z]=N_d[Z]+2*nhalo;

  int nsites=Nall[X]*Nall[Y]*Nall[Z];

  Ngradcalc[X]=N_d[X]+2*nextra;
  Ngradcalc[Y]=N_d[Y]+2*nextra;
  Ngradcalc[Z]=N_d[Z]+2*nextra;

  str[Z] = 1;
  str[Y] = str[Z]*(N_d[Z] + 2*nhalo);
  str[X] = str[Y]*(N_d[Y] + 2*nhalo);


  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  /* Avoid going beyond problem domain */
  if (threadIndex < Ngradcalc[X]*Ngradcalc[Y]*Ngradcalc[Z])
    {

      /* calculate index from CUDA thread index */

      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Ngradcalc);
      index = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra,
				     kk+nhalo-nextra,Nall);      
      

      if (site_map_status_d[index] == FLUID){

	index1 = get_linear_index_gpu_d(ii+nhalo-nextra+1,jj+nhalo-nextra,
					kk+nhalo-nextra,Nall);      
	status[0]=site_map_status_d[index1];
	
	index1 = get_linear_index_gpu_d(ii+nhalo-nextra-1,jj+nhalo-nextra,
					kk+nhalo-nextra,Nall);      
	status[1]=site_map_status_d[index1];
	
	index1 = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra+1,
					kk+nhalo-nextra,Nall);      
	status[2]=site_map_status_d[index1];
	
	index1 = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra-1,
					kk+nhalo-nextra,Nall);      
	status[3]=site_map_status_d[index1];
	
	index1 = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra,
					kk+nhalo-nextra+1,Nall);      
	status[4]=site_map_status_d[index1];
	
	index1 = get_linear_index_gpu_d(ii+nhalo-nextra,jj+nhalo-nextra,
					kk+nhalo-nextra-1,Nall);      
	status[5]=site_map_status_d[index1];
	
	
	for (n1 = 0; n1 < nop; n1++) { 
	  for (ia = 0; ia < 3; ia++) {
	    gradn[n1][ia][0] =  
	      field_d[nsites*n1+index+str[ia]]- field_d[nsites*n1+index];
	    gradn[n1][ia][1] =
	      field_d[nsites*n1+index]- field_d[nsites*n1+index-str[ia]];
	  }
	}
	
	for (n1 = 0; n1 < nop; n1++) { 
	  del2_d[n1*nsites+index] = 0.0;
	  for (ia = 0; ia < 3; ia++) {
	    grad_d[ia*nsites*nop+n1*nsites+index]=
	      0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	    del2_d[n1*nsites+index]+= gradn[n1][ia][0] - gradn[n1][ia][1];
	  }
	}
	
	
	ns = 0;
	for (n = 0; n < 6; n++) {
	  mask[n] = (status[n] != FLUID);
	  ns += mask[n];
	}

	if (ns !=0 ){

	/* Solid boundary condition corrections are required. */

	//util_q5_to_qab(qs, field + NOP*index);

	  qs[X][X] = field_d[nsites*0+index];
	  qs[X][Y] = field_d[nsites*1+index]; 
	  qs[X][Z] = field_d[nsites*2+index]; 
	  qs[Y][X] = field_d[nsites*1+index]; 
	  qs[Y][Y] = field_d[nsites*3+index]; 
	  qs[Y][Z] = field_d[nsites*4+index]; 
	  qs[Z][X] = field_d[nsites*2+index]; 
	  qs[Z][Y] = field_d[nsites*4+index]; 
	  qs[Z][Z] = -field_d[nsites*0+index] - field_d[nsites*3+index]; 


	  for (n = 0; n < 6; n++) {
	  if (status[n] != FLUID) {


	    //TO DO
	    colloids_q_boundary_normal_gpu_d(bcs[n], dn, Nall, nhalo, nextra, ii, jj, kk, site_map_status_d,colloid_map_d, colloid_r_d);
	      //colloids_q_boundary(dn, qs, q0, status[n]);

	    for (ia = 0; ia < 3; ia++) {
	      for (ib = 0; ib < 3; ib++) {
		q0[ia][ib] = 0.5*amplitude_cd*(3.0*dn[ia]*dn[ib] - d_cd[ia][ib]);
	      }
	    }


	      /* Check for wall/colloid */
	    //TO DO: BOUNDARY
	      //if (status[n] == COLLOID) w = colloids_q_tensor_w();
	      //if (status[n] == BOUNDARY) w = wall_w_get();
	      //assert(status[n] == COLLOID || status[n] == BOUNDARY);
	      
	      /* Compute c[n][a][b] */
	      
	      for (ia = 0; ia < 3; ia++) {
	    	for (ib = 0; ib < 3; ib++) {
	    	  c[n][ia][ib] = 0.0;
	    	  for (ig = 0; ig < 3; ig++) {
	    	    for (ih = 0; ih < 3; ih++) {
	    	      c[n][ia][ib] -= kappa1_cd*q_0_cd*bcs[n][ig]*
	    	          (e_cd[ia][ig][ih]*qs[ih][ib] + e_cd[ib][ig][ih]*qs[ih][ia]);
	    	    }
	    	  }
	    	  c[n][ia][ib] -= w_cd*(qs[ia][ib] - q0[ia][ib]);
	    	}
	      }
	  }
	  }
	    
	    /* Set up initial approximation to grad using partial gradients
	 /* where solid sites are involved (or zero where none available) */

	for (n1 = 0; n1 < NOP; n1++) {
	  for (ia = 0; ia < 3; ia++) {
	    gradn[n1][ia][0] *= (1 - mask[2*ia]);
	    gradn[n1][ia][1] *= (1 - mask[2*ia + 1]);
	    grad_d[ia*nsites*nop+n1*nsites+index]  =
	      0.5*(1.0 + ((mask[2*ia] + mask[2*ia+1]) % 2))*
	      (gradn[n1][ia][0] + gradn[n1][ia][1]);
	  }
	}


	/* Iterate to a solution. */

	for (niterate = 0; niterate < NITERATION; niterate++) {
	  
	  for (n = 0; n < 6; n++) {
	    
	    if (status[n] != FLUID) {
	      
	      for (n1 = 0; n1 < NOP; n1++) {
	  	for (ia = 0; ia < 3; ia++) {
	  	  dq[n1][ia] = grad_d[ia*nsites*nop+n1*nsites+index];		  
	  	}
	  	dq[n1][normal[n]] = 1.0;
	      }
	      
	      /* Construct boundary condition terms. */
	      
	      gradient_bcs_gpu_d(kappa0_cd, kappa1_cd, bcs[n], dq, bc);


	      
	      for (n1 = 0; n1 < NOP; n1++) {
	  	b[n1] = 0.0;
	  	for (n2 = 0; n2 < NOP; n2++) {
	  	  a[n1][n2] = bc[n1][n2][normal[n]];
	  	  b[n1] -= bc[n1][n2][normal[n]];
	  	  for (ia = 0; ia < 3; ia++) {
	  	    b[n1] += bc[n1][n2][ia];
		    //b[n1] =0.01;
	  	  }
	  	}
	      }
	      
	      b[XX] = -(b[XX] +     c[n][X][X]);
	      b[XY] = -(b[XY] + 2.0*c[n][X][Y]);
	      b[XZ] = -(b[XZ] + 2.0*c[n][X][Z]);
	      b[YY] = -(b[YY] +     c[n][Y][Y]);
	      b[YZ] = -(b[YZ] + 2.0*c[n][Y][Z]);
	      
	      util_gaussian_gpu_d(a, b);
	      
	      for (n1 = 0; n1 < NOP; n1++) {
	  	gradn[n1][normal[n]][nsolid[n]] = b[n1];
	      }
	    }
	  }

	  /* Do not update gradients if solid neighbours in both directions */
	  for (ia = 0; ia < 3; ia++) {
	    tmp = 1.0*(1 - (mask[2*ia] && mask[2*ia+1]));
	    for (n1 = 0; n1 < NOP; n1++) {
	      gradn[n1][ia][0] *= tmp;
	      gradn[n1][ia][1] *= tmp;
	    }
	  }

	  /* Now recompute gradients */

	  for (n1 = 0; n1 < NOP; n1++) {
	    del2_d[n1*nsites+index] = 0.0;
	    for (ia = 0; ia < 3; ia++) {
	      grad_d[ia*nsites*nop+n1*nsites+index] =
	      0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
	      del2_d[n1*nsites+index] += gradn[n1][ia][0] - gradn[n1][ia][1];
	    }
	  }

	  /* No iteration required if only one boundary. */
	  if (ns < 2) break;
	}



	}



      }


      
    }
  return;
}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N_d[3])
{
  
  int yfac = N_d[Z];
  int xfac = N_d[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}

/* get 3d coordinates from the index on the accelerator */
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N_d[3])

{
  
  int yfac = N_d[Z];
  int xfac = N_d[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}
