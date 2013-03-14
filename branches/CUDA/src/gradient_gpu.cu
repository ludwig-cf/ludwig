/*****************************************************************************
 *
 * gradient_gpu.cu
 *
 * GPU versions of gradient schemes 
 * Alan Gray
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include "common_gpu.h"
#include "colloids.h"
#include "site_map.h"
#include "gradient_gpu.h"
#include "gradient_internal_gpu.h"
#include "util.h"



/* scheme in use */
static char gradient_gpu=-1;

__constant__ int Ngradcalc_cd[3];
__constant__ int nextra_cd;

void phi_gradients_compute_gpu()
{

  int nop,N[3],Ngradcalc[3],nhalo;


  nhalo = coords_nhalo();
  int nextra=nhalo-1;
  coords_nlocal(N); 
  nop = phi_nop();
  

  put_gradient_constants_on_gpu();

  /* set up CUDA grid */
  
  Ngradcalc[X]=N[X]+2*nextra;
  Ngradcalc[Y]=N[Y]+2*nextra;
  Ngradcalc[Z]=N[Z]+2*nextra;
  
  int nblocks=(Ngradcalc[X]*Ngradcalc[Y]*Ngradcalc[Z]+DEFAULT_TPB-1)
    /DEFAULT_TPB;
  

 if (gradient_gpu==OPTION_3D_7PT_FLUID){
  gradient_3d_7pt_fluid_operator_gpu_d<<<nblocks,DEFAULT_TPB>>>
    ( phi_site_d,grad_phi_site_d,delsq_phi_site_d,
     le_index_real_to_buffer_d); 
 }
 else if (gradient_gpu==OPTION_3D_7PT_SOLID){
  gradient_3d_7pt_solid_gpu_d<<<nblocks,DEFAULT_TPB>>>
    (nop, nhalo, N_d, phi_site_d,grad_phi_site_d,delsq_phi_site_d,
     site_map_status_d,colloid_map_d, colloid_r_d,nextra);
 }
 else
   {
    printf("The chosen gradient scheme is not yet supported in GPU mode. Exiting\n");
    exit(1);
  }
    
  
  cudaThreadSynchronize();

checkCUDAError("gradient_3d_7pt");  
  return;
}

/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_operator_gpu_d
 *
 *****************************************************************************/
__global__ void gradient_3d_7pt_fluid_operator_gpu_d(const double * field_d,
						     double * grad_d,
						     double * del2_d,
						     int * le_index_real_to_buffer_d) {

  int n, icm1, icp1;
  int index, indexm1, indexp1;

  int threadIndex,ii, jj, kk;

  int ys = N_cd[Z] + 2*nhalo_cd;

  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  /* Avoid going beyond problem domain */
  if (threadIndex < Ngradcalc_cd[X]*Ngradcalc_cd[Y]*Ngradcalc_cd[Z])
    {

      /* calculate index from CUDA thread index */

      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Ngradcalc_cd);
      index = get_linear_index_gpu_d(ii+nhalo_cd-nextra_cd,jj+nhalo_cd-nextra_cd,
				     kk+nhalo_cd-nextra_cd,Nall_cd);      
      

      /* icm1 = le_index_real_to_buffer(ic, -1); */
      /* icp1 = le_index_real_to_buffer(ic, +1); */
      /*le_index_real_to_buffer_d holds -1 then +1 translation values */
      icm1=le_index_real_to_buffer_d[ii+nhalo_cd-nextra_cd];
      icp1=le_index_real_to_buffer_d[Nall_cd[X]+ii+nhalo_cd-nextra_cd];      

      indexm1 = get_linear_index_gpu_d(icm1,jj+nhalo_cd-nextra_cd,kk+nhalo_cd-nextra_cd,Nall_cd);
      indexp1 = get_linear_index_gpu_d(icp1,jj+nhalo_cd-nextra_cd,kk+nhalo_cd-nextra_cd,Nall_cd);


      for (n = 0; n < nop_cd; n++) { 

	  grad_d[X*nsites_cd*nop_cd+n*nsites_cd+index]
	    = 0.5*(field_d[nsites_cd*n+indexp1] - field_d[nsites_cd*n+indexm1]);
	  grad_d[Y*nsites_cd*nop_cd+n*nsites_cd+index]
	    = 0.5*(field_d[nsites_cd*n+(index + ys)] - field_d[nsites_cd*n+(index - ys)]);
	  grad_d[Z*nsites_cd*nop_cd+n*nsites_cd+index]
	    = 0.5*(field_d[nsites_cd*n+(index + 1)] - field_d[nsites_cd*n+(index - 1)]);
	  del2_d[n*nsites_cd + index]
	    = field_d[nsites_cd*n+indexp1] + field_d[nsites_cd*n+indexm1]
	    + field_d[nsites_cd*n+(index + ys)] + field_d[nsites_cd*n+(index - ys)]
	    + field_d[nsites_cd*n+(index + 1)] + field_d[nsites_cd*n+(index - 1)]
	    - 6.0*field_d[nsites_cd*n+index];
		  } 


   } 
  return;
}


#define NITERATION 40


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
	      
	      //gradient_bcs_gpu_d(kappa0_cd, kappa1_cd, bcs[n], dq, bc);
	      /* XX equation */
	      
	      bc[XX][XX][X] =  kappa0_cd*bcs_cd[n][X]*dq[XX][X];
	      bc[XX][XY][X] = -kappa1_cd*bcs_cd[n][Y]*dq[XY][X];
	      bc[XX][XZ][X] = -kappa1_cd*bcs_cd[n][Z]*dq[XZ][X];
	      bc[XX][YY][X] =  0.0;
	      bc[XX][YZ][X] =  0.0;
	      
	      bc[XX][XX][Y] = kappa1_cd*bcs_cd[n][Y]*dq[XX][Y];
	      bc[XX][XY][Y] = kappa0_cd*bcs_cd[n][X]*dq[XY][Y];
	      bc[XX][XZ][Y] = 0.0;
	      bc[XX][YY][Y] = 0.0;
	      bc[XX][YZ][Y] = 0.0;
	      
	      bc[XX][XX][Z] = kappa1_cd*bcs_cd[n][Z]*dq[XX][Z];
	      bc[XX][XY][Z] = 0.0;
	      bc[XX][XZ][Z] = kappa0_cd*bcs_cd[n][X]*dq[XZ][Z];
	      bc[XX][YY][Z] = 0.0;
	      bc[XX][YZ][Z] = 0.0;
	      
	      /* XY equation */
	      
	      bc[XY][XX][X] =  kappa0_cd*bcs_cd[n][Y]*dq[XX][X];
	      bc[XY][XY][X] =  kappa2_cd*bcs_cd[n][X]*dq[XY][X];
	      bc[XY][XZ][X] =  0.0;
	      bc[XY][YY][X] = -kappa1_cd*bcs_cd[n][Y]*dq[YY][X];
	      bc[XY][YZ][X] = -kappa1_cd*bcs_cd[n][Z]*dq[YZ][X];
	      
	      bc[XY][XX][Y] = -kappa1_cd*bcs_cd[n][X]*dq[XX][Y];
	      bc[XY][XY][Y] =  kappa2_cd*bcs_cd[n][Y]*dq[XY][Y];
	      bc[XY][XZ][Y] = -kappa1_cd*bcs_cd[n][Z]*dq[XZ][Y];
	      bc[XY][YY][Y] =  kappa0_cd*bcs_cd[n][X]*dq[YY][Y];
	      bc[XY][YZ][Y] =  0.0;
	      
	      bc[XY][XX][Z] = 0.0;
	      bc[XY][XY][Z] = 2.0*kappa1_cd*bcs_cd[n][Z]*dq[XY][Z];
	      bc[XY][XZ][Z] = kappa0_cd*bcs_cd[n][Y]*dq[XZ][Z];
	      bc[XY][YY][Z] = 0.0;
	      bc[XY][YZ][Z] = kappa0_cd*bcs_cd[n][X]*dq[YZ][Z];
	      
	      /* XZ equation */
	      
	      bc[XZ][XX][X] =  kappa2_cd*bcs_cd[n][Z]*dq[XX][X];
	      bc[XZ][XY][X] =  0.0;
	      bc[XZ][XZ][X] =  kappa2_cd*bcs_cd[n][X]*dq[XZ][X];
	      bc[XZ][YY][X] =  kappa1_cd*bcs_cd[n][Z]*dq[YY][X];
	      bc[XZ][YZ][X] = -kappa1_cd*bcs_cd[n][Y]*dq[YZ][X];
	      
	      bc[XZ][XX][Y] = 0.0;
	      bc[XZ][XY][Y] = kappa0_cd*bcs_cd[n][Z]*dq[XY][Y];
	      bc[XZ][XZ][Y] = 2.0*kappa1_cd*bcs_cd[n][Y]*dq[XZ][Y];
	      bc[XZ][YY][Y] = 0.0;
	      bc[XZ][YZ][Y] = kappa0_cd*bcs_cd[n][X]*dq[YZ][Y];
	      
	      bc[XZ][XX][Z] = -kappa2_cd*bcs_cd[n][X]*dq[XX][Z];
	      bc[XZ][XY][Z] = -kappa1_cd*bcs_cd[n][Y]*dq[XY][Z];
	      bc[XZ][XZ][Z] =  kappa2_cd*bcs_cd[n][Z]*dq[XZ][Z];
	      bc[XZ][YY][Z] = -kappa0_cd*bcs_cd[n][X]*dq[YY][Z];
	      bc[XZ][YZ][Z] =  0.0;
	      
	      /* YY equation */
	      
	      bc[YY][XX][X] = 0.0;
	      bc[YY][XY][X] = kappa0_cd*bcs_cd[n][Y]*dq[XY][X];
	      bc[YY][XZ][X] = 0.0;
	      bc[YY][YY][X] = kappa1_cd*bcs_cd[n][X]*dq[YY][X];
	      bc[YY][YZ][X] = 0.0;
	      
	      bc[YY][XX][Y] =  0.0;
	      bc[YY][XY][Y] = -kappa1_cd*bcs_cd[n][X]*dq[XY][Y];
	      bc[YY][XZ][Y] =  0.0;
	      bc[YY][YY][Y] =  kappa0_cd*bcs_cd[n][Y]*dq[YY][Y];
	      bc[YY][YZ][Y] = -kappa1_cd*bcs_cd[n][Z]*dq[YZ][Y];
	      
	      bc[YY][XX][Z] = 0.0;
	      bc[YY][XY][Z] = 0.0;
	      bc[YY][XZ][Z] = 0.0;
	      bc[YY][YY][Z] = kappa1_cd*bcs_cd[n][Z]*dq[YY][Z];
	      bc[YY][YZ][Z] = kappa0_cd*bcs_cd[n][Y]*dq[YZ][Z];
	      
	      /* YZ equation */
	      
	      bc[YZ][XX][X] = 0.0;
	      bc[YZ][XY][X] = kappa0_cd*bcs_cd[n][Z]*dq[XY][X];
	      bc[YZ][XZ][X] = kappa0_cd*bcs_cd[n][Y]*dq[XZ][X];
	      bc[YZ][YY][X] = 0.0;
	      bc[YZ][YZ][X] = 2.0*kappa1_cd*bcs_cd[n][X]*dq[YZ][X];
	      
	      bc[YZ][XX][Y] =  kappa1_cd*bcs_cd[n][Z]*dq[XX][Y];
	      bc[YZ][XY][Y] =  0.0;
	      bc[YZ][XZ][Y] = -kappa1_cd*bcs_cd[n][X]*dq[XZ][Y];
	      bc[YZ][YY][Y] =  kappa2_cd*bcs_cd[n][Z]*dq[YY][Y];
	      bc[YZ][YZ][Y] =  kappa2_cd*bcs_cd[n][Y]*dq[YZ][Y];
	      
	      bc[YZ][XX][Z] = -kappa0_cd*bcs_cd[n][Y]*dq[XX][Z];
	      bc[YZ][XY][Z] = -kappa1_cd*bcs_cd[n][X]*dq[XY][Z];
	      bc[YZ][XZ][Z] =  0.0;
	      bc[YZ][YY][Z] = -kappa2_cd*bcs_cd[n][Y]*dq[YY][Z];
	      bc[YZ][YZ][Z] =  kappa2_cd*bcs_cd[n][Z]*dq[YZ][Z];
	      

	      
	      for (n1 = 0; n1 < NOP; n1++) {
	  	b[n1] = 0.0;
	  	for (n2 = 0; n2 < NOP; n2++) {
	  	  a[n1][n2] = bc[n1][n2][normal[n]];
	  	  b[n1] -= bc[n1][n2][normal[n]];
	  	  for (ia = 0; ia < 3; ia++) {
	  	    b[n1] += bc[n1][n2][ia];
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

__device__ static void gradient_bcs_gpu_d(const double kappa0, const double kappa1, const int dn[3],
			 double dq[NOP][3], double bc[NOP][NOP][3]) {


  /* XX equation */

  bc[XX][XX][X] =  kappa0_cd*dn[X]*dq[XX][X];
  bc[XX][XY][X] = -kappa1_cd*dn[Y]*dq[XY][X];
  bc[XX][XZ][X] = -kappa1_cd*dn[Z]*dq[XZ][X];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;

  bc[XX][XX][Y] = kappa1_cd*dn[Y]*dq[XX][Y];
  bc[XX][XY][Y] = kappa0_cd*dn[X]*dq[XY][Y];
  bc[XX][XZ][Y] = 0.0;
  bc[XX][YY][Y] = 0.0;
  bc[XX][YZ][Y] = 0.0;

  bc[XX][XX][Z] = kappa1_cd*dn[Z]*dq[XX][Z];
  bc[XX][XY][Z] = 0.0;
  bc[XX][XZ][Z] = kappa0_cd*dn[X]*dq[XZ][Z];
  bc[XX][YY][Z] = 0.0;
  bc[XX][YZ][Z] = 0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0_cd*dn[Y]*dq[XX][X];
  bc[XY][XY][X] =  kappa2_cd*dn[X]*dq[XY][X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1_cd*dn[Y]*dq[YY][X];
  bc[XY][YZ][X] = -kappa1_cd*dn[Z]*dq[YZ][X];

  bc[XY][XX][Y] = -kappa1_cd*dn[X]*dq[XX][Y];
  bc[XY][XY][Y] =  kappa2_cd*dn[Y]*dq[XY][Y];
  bc[XY][XZ][Y] = -kappa1_cd*dn[Z]*dq[XZ][Y];
  bc[XY][YY][Y] =  kappa0_cd*dn[X]*dq[YY][Y];
  bc[XY][YZ][Y] =  0.0;

  bc[XY][XX][Z] = 0.0;
  bc[XY][XY][Z] = 2.0*kappa1_cd*dn[Z]*dq[XY][Z];
  bc[XY][XZ][Z] = kappa0_cd*dn[Y]*dq[XZ][Z];
  bc[XY][YY][Z] = 0.0;
  bc[XY][YZ][Z] = kappa0_cd*dn[X]*dq[YZ][Z];

  /* XZ equation */

  bc[XZ][XX][X] =  kappa2_cd*dn[Z]*dq[XX][X];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2_cd*dn[X]*dq[XZ][X];
  bc[XZ][YY][X] =  kappa1_cd*dn[Z]*dq[YY][X];
  bc[XZ][YZ][X] = -kappa1_cd*dn[Y]*dq[YZ][X];

  bc[XZ][XX][Y] = 0.0;
  bc[XZ][XY][Y] = kappa0_cd*dn[Z]*dq[XY][Y];
  bc[XZ][XZ][Y] = 2.0*kappa1_cd*dn[Y]*dq[XZ][Y];
  bc[XZ][YY][Y] = 0.0;
  bc[XZ][YZ][Y] = kappa0_cd*dn[X]*dq[YZ][Y];

  bc[XZ][XX][Z] = -kappa2_cd*dn[X]*dq[XX][Z];
  bc[XZ][XY][Z] = -kappa1_cd*dn[Y]*dq[XY][Z];
  bc[XZ][XZ][Z] =  kappa2_cd*dn[Z]*dq[XZ][Z];
  bc[XZ][YY][Z] = -kappa0_cd*dn[X]*dq[YY][Z];
  bc[XZ][YZ][Z] =  0.0;

  /* YY equation */

  bc[YY][XX][X] = 0.0;
  bc[YY][XY][X] = kappa0_cd*dn[Y]*dq[XY][X];
  bc[YY][XZ][X] = 0.0;
  bc[YY][YY][X] = kappa1_cd*dn[X]*dq[YY][X];
  bc[YY][YZ][X] = 0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1_cd*dn[X]*dq[XY][Y];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0_cd*dn[Y]*dq[YY][Y];
  bc[YY][YZ][Y] = -kappa1_cd*dn[Z]*dq[YZ][Y];

  bc[YY][XX][Z] = 0.0;
  bc[YY][XY][Z] = 0.0;
  bc[YY][XZ][Z] = 0.0;
  bc[YY][YY][Z] = kappa1_cd*dn[Z]*dq[YY][Z];
  bc[YY][YZ][Z] = kappa0_cd*dn[Y]*dq[YZ][Z];

  /* YZ equation */

  bc[YZ][XX][X] = 0.0;
  bc[YZ][XY][X] = kappa0_cd*dn[Z]*dq[XY][X];
  bc[YZ][XZ][X] = kappa0_cd*dn[Y]*dq[XZ][X];
  bc[YZ][YY][X] = 0.0;
  bc[YZ][YZ][X] = 2.0*kappa1_cd*dn[X]*dq[YZ][X];

  bc[YZ][XX][Y] =  kappa1_cd*dn[Z]*dq[XX][Y];
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1_cd*dn[X]*dq[XZ][Y];
  bc[YZ][YY][Y] =  kappa2_cd*dn[Z]*dq[YY][Y];
  bc[YZ][YZ][Y] =  kappa2_cd*dn[Y]*dq[YZ][Y];

  bc[YZ][XX][Z] = -kappa0_cd*dn[Y]*dq[XX][Z];
  bc[YZ][XY][Z] = -kappa1_cd*dn[X]*dq[XY][Z];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa2_cd*dn[Y]*dq[YY][Z];
  bc[YZ][YZ][Z] =  kappa2_cd*dn[Z]*dq[YZ][Z];

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


void set_gradient_option_gpu(char option){

  gradient_gpu=option;

}

void put_gradient_constants_on_gpu(){
  // FROM blue_phase.c
  double q0_;        /* Pitch = 2pi / q0_ */
  double kappa0_;    /* Elastic constant \kappa_0 */
  double kappa1_;    /* Elastic constant \kappa_1 */
  double kappa2_;


  double w;                                 /* Anchoring strength parameter */
  double amplitude;

  int N[3],nhalo,Nall[3], Ngradcalc[3];
  
  nhalo = coords_nhalo();
  coords_nlocal(N); 


  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;


  int nextra=nhalo-1;
  Ngradcalc[X]=N[X]+2*nextra;
  Ngradcalc[Y]=N[Y]+2*nextra;
  Ngradcalc[Z]=N[Z]+2*nextra;


  
  int nsites=Nall[X]*Nall[Y]*Nall[Z];
  int nop = phi_nop();
  /* copy to constant memory on device */
  cudaMemcpyToSymbol(N_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Ngradcalc_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Nall_cd, Nall, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nhalo_cd, &nhalo, sizeof(int), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(nop_cd, &nop, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nsites_cd, &nsites, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nextra_cd, &nextra, sizeof(int), 0, cudaMemcpyHostToDevice); 



  q0_=blue_phase_q0();
  kappa0_=blue_phase_kappa0();
  kappa1_=blue_phase_kappa1();
  kappa2_=kappa0_+kappa1_;
  w = colloids_q_tensor_w();
  amplitude = blue_phase_amplitude_compute();

  int noffset[3];
  coords_nlocal_offset(noffset);

  char bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

 cudaMemcpyToSymbol(q_0_cd, &q0_, sizeof(double), 0, cudaMemcpyHostToDevice);  
cudaMemcpyToSymbol(kappa0_cd, &kappa0_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(kappa1_cd, &kappa1_, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(kappa2_cd, &kappa2_, sizeof(double), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(e_cd, e_, 3*3*3*sizeof(double), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(noffset_cd, noffset, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 

 cudaMemcpyToSymbol(w_cd, &w, sizeof(double), 0, cudaMemcpyHostToDevice); 
 cudaMemcpyToSymbol(amplitude_cd, &amplitude, sizeof(double), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(d_cd, d_, 3*3*sizeof(double), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(bcs_cd, bcs, 6*3*sizeof(char), 0, cudaMemcpyHostToDevice); 
cudaFuncSetCacheConfig(gradient_3d_7pt_solid_gpu_d,cudaFuncCachePreferL1);

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
