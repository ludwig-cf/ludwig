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
#include "utilities_gpu.h"
#include "utilities_internal_gpu.h"
#include "util.h"

#include "blue_phase.h"
#include "colloids_Q_tensor.h"

enum gradent_options_gpu {OPTION_3D_7PT_FLUID,OPTION_3D_7PT_SOLID};

/* scheme in use */
static char gradient_gpu=-1;

__constant__ double e_cd[3][3][3];
__constant__ double d_cd[3][3];

extern double * phi_site_d;
extern double * grad_phi_site_d;
extern double * delsq_phi_site_d;
extern char * site_map_status_d;
extern char * colloid_map_d;
extern double * colloid_r_d;

#define NQAB 5
#define NSYMM 6

typedef struct gradient_gpu_d_s gradient_gpu_t;

struct gradient_gpu_d_s {
  int nhalo;
  int nextra;
  int npoints;
  int nxtent[3];    /* kernel extent 1 - nextra ... nlocal[] + nextra */
  int nall[3];      /* local extent incl. halos:  nlocal[] + 2*nhalo */
  int nsites;       /* nall[X]*nall[Y]*nall[Z] */
  int nlocal[3];
  int noffset[3];

  int ntype_coll;   /* Colloid anchoring type */
  int ntype_wall;   /* Wall anchoring type */

  double amp;       /* surface anchoring */
  double kappa0;    /* elastic constant */
  double kappa1;    /* elastic constant */
  double q0;
  double w1_coll;
  double w2_coll;
  double w1_wall;
  double w2_wall;

  double a6inv[3][6];
  double a12inv[3][12][12];
  double a18inv[18][18];
};

static __constant__ gradient_gpu_t dev;
static gradient_gpu_t host;

__global__ void gradient_fluid_d(const double* __restrict__ field_d,
                                 double * __restrict__ grad_d,
                                 double* __restrict__ del2_d);

#ifdef KEVIN_GPU
extern coll_array_t * carry_d;
__global__ void gradient_solid_d(const double * __restrict__ field_d,
                                 double * __restrict__ grad_d,
		 		 double * __restrict__ del2_d,
			 	 char * __restrict__ site_map_status_d,
                                 coll_array_t * __restrict__ carry_d);
#else
__global__ void gradient_solid_d(const double * __restrict__ field_d,
                                 double * __restrict__ grad_d,
		 		 double * __restrict__ del2_d,
			 	 char * __restrict__ site_map_status_d,
                                 char * __restrict__ colloid_map_d,
                                 double * __restrict__ colloid_r_d);
#endif
__device__ static int coords_index_gpu_d(int ic, int jc, int kc, int * index);
__device__ static void coords_from_index_gpu_d(int index,
	int *ic, int *jc, int *kc);

__host__ __device__ static void gradient_bcs6x6_coeff_d(double kappa0,
	                                       double kappa1, const int dn[3],
                                               double bc[6][6][3]);

#ifdef KEVIN_GPU
__device__ void q_boundary_constants_d(int ic, int jc, int kc, double qs[3][3],
                                       const int di[3],
                                       int status,
                                       coll_array_t * carry_d, double c[3][3]);
#else
__device__ void q_boundary_constants_d(int ic, int jc, int kc, double qs[3][3],
                                       const int di[3],
                                       int status, char * colloid_map_d,
                                       double * colloid_r_d, double c[3][3]);
#endif
/*****************************************************************************
 *
 *  phi_gradients_compute_gpu
 *
 *****************************************************************************/

int phi_gradients_compute_gpu() {

  int nblocks;
  int ndefault;
  dim3 nblock;

  /* Points required and thread blocks */

  ndefault = DEFAULT_TPB/2;
  nblocks = (host.npoints + ndefault - 1) / ndefault;
  nblock.x = nblocks; nblock.y = 1; nblock.z = 1;

  if (gradient_gpu == OPTION_3D_7PT_FLUID) {
    gradient_fluid_d<<<nblocks,ndefault>>>(phi_site_d,
                                     	      grad_phi_site_d,
		 		              delsq_phi_site_d);


  }
#ifdef KEVIN_GPU
  if (gradient_gpu == OPTION_3D_7PT_SOLID) {
    gradient_solid_d<<<nblocks,ndefault>>>(phi_site_d,
                                     	      grad_phi_site_d,
		 		              delsq_phi_site_d,
			 	              site_map_status_d,
                                              carry_d);
  }
#else
  if (gradient_gpu == OPTION_3D_7PT_SOLID) {
    gradient_solid_d<<<nblocks,ndefault>>>(phi_site_d,
                                     	      grad_phi_site_d,
		 		              delsq_phi_site_d,
			 	              site_map_status_d,
                                              colloid_map_d,
                                              colloid_r_d);
  }
#endif
  cudaThreadSynchronize();
  checkCUDAError("gradient_3d_7pt");  

  return 0;
}

/*****************************************************************************
 *
 *  gradient_gpu_init_h
 *
 *****************************************************************************/

int gradient_gpu_init_h(void) {

  int ia, n1, n2;
  const int bcs[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  double bc[6][6][3];
  double ** a12inv[3];
  double ** a18inv;

  host.nhalo = coords_nhalo();
  host.nextra = host.nhalo - 1;
  coords_nlocal(host.nlocal);
  coords_nlocal_offset(host.noffset);

  host.nxtent[X] = host.nlocal[X] + 2*host.nextra;
  host.nxtent[Y] = host.nlocal[Y] + 2*host.nextra;
  host.nxtent[Z] = host.nlocal[Z] + 2*host.nextra;
  host.npoints = host.nxtent[X]*host.nxtent[Y]*host.nxtent[Z];

  host.nall[X] = host.nlocal[X] + 2*host.nhalo;
  host.nall[Y] = host.nlocal[Y] + 2*host.nhalo;
  host.nall[Z] = host.nlocal[Z] + 2*host.nhalo;
  host.nsites = host.nall[X]*host.nall[Y]*host.nall[Z];

  /* Anchoring */
  host.amp    = blue_phase_amplitude_compute();
  host.kappa0 = blue_phase_kappa0();
  host.kappa1 = blue_phase_kappa1();
  host.q0     = blue_phase_q0();

  host.ntype_coll = colloids_q_anchoring();
  host.ntype_wall = wall_anchoring();
  host.w1_coll = colloids_q_tensor_w1();
  host.w2_coll = colloids_q_tensor_w2();
  host.w1_wall = wall_w1();
  host.w2_wall = wall_w2();

  /* Boundary condition inverse matrices */

  util_matrix_create(2*NSYMM, 2*NSYMM, &(a12inv[0]));
  util_matrix_create(2*NSYMM, 2*NSYMM, &(a12inv[1]));
  util_matrix_create(2*NSYMM, 2*NSYMM, &(a12inv[2]));
  util_matrix_create(3*NSYMM, 3*NSYMM, &a18inv);

  for (ia = 0; ia < 3; ia ++) {
    gradient_bcs6x6_coeff_d(host.kappa0, host.kappa1, bcs[ia], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      host.a6inv[ia][n1] = 1.0/bc[n1][n1][ia];
    }

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {
        a18inv[ia*NSYMM+n1][X*NSYMM+n2] = 0.5*(1.0 + d_[ia][X])*bc[n1][n2][X];
        a18inv[ia*NSYMM+n1][Y*NSYMM+n2] = 0.5*(1.0 + d_[ia][Y])*bc[n1][n2][Y];
        a18inv[ia*NSYMM+n1][Z*NSYMM+n2] = 0.5*(1.0 + d_[ia][Z])*bc[n1][n2][Z];
      }
    }
  }

  /* Set up the 2-unknown cases, which are the relevant blocks from the
   * full 3-unknown case. */

  /* XY [0][][] and YZ [2][][] */

  for (n1 = 0; n1 < 2*NSYMM; n1++) {
    for (n2 = 0; n2 < 2*NSYMM; n2++) {
      a12inv[0][n1][n2] = a18inv[n1][n2];
      a12inv[2][n1][n2] = a18inv[NSYMM+n1][NSYMM+n2];
    }
  }

  /* XZ [1][][] */

  for (n1 = 0; n1 < NSYMM; n1++) {
    for (n2 = 0; n2 < NSYMM; n2++) {
      a12inv[1][n1][n2] = a18inv[n1][n2];
      a12inv[1][n1][NSYMM + n2] = a18inv[n1][2*NSYMM + n2];
    }
  }

  for (n1 = NSYMM; n1 < 2*NSYMM; n1++) {
    for (n2 = 0; n2 < NSYMM; n2++) {
      a12inv[1][n1][n2] = a18inv[NSYMM + n1][n2];
      a12inv[1][n1][NSYMM + n2] = a18inv[NSYMM + n1][2*NSYMM + n2];
    }
  }

  /* Now compute the inverses and store */

  ia = util_matrix_invert(2*NSYMM, a12inv[0]);
  assert(ia == 0);

  ia = util_matrix_invert(2*NSYMM, a12inv[1]);
  assert(ia == 0);

  ia = util_matrix_invert(2*NSYMM, a12inv[2]);
  assert(ia == 0);

  ia = util_matrix_invert(3*NSYMM, a18inv);
  assert(ia == 0);

  for (ia = 0; ia < 3; ia++) {
    for (n1 = 0; n1 < 2*NSYMM; n1++) {
      for (n2 = 0; n2 < 2*NSYMM; n2++) {
        host.a12inv[ia][n1][n2] = a12inv[ia][n1][n2];
      }
    }
  }

  for (n1 = 0; n1 < 3*NSYMM; n1++) {
    for (n2 = 0; n2 < 3*NSYMM; n2++) {
      host.a18inv[n1][n2] = a18inv[n1][n2];
    }
  }

  util_matrix_free(3*NSYMM, &a18inv);
  util_matrix_free(2*NSYMM, &(a12inv[2]));
  util_matrix_free(2*NSYMM, &(a12inv[1]));
  util_matrix_free(2*NSYMM, &(a12inv[0]));

  cudaMemcpyToSymbol(dev, &host, sizeof(gradient_gpu_t), 0,
	             cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(e_cd, e_, 27*sizeof(double), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(d_cd, d_, 9*sizeof(double), 0, cudaMemcpyHostToDevice); 

  return 0;
}

/*****************************************************************************
 *
 *  gradient_fluid_d
 *
 *  Sanity check for Kevin
 *
 *****************************************************************************/

__global__ void gradient_fluid_d(const double* __restrict__ field_d,
                                 double* __restrict__ grad_d,
				 double* __restrict__ del2_d) {
  int n;
  int index;
  int threadIndex, ic, jc, kc;
  int ys = dev.nall[Z];
  int xs = ys*dev.nall[Y];
  int nstr;

  threadIndex = blockIdx.x*blockDim.x + threadIdx.x;

  if (threadIndex >= dev.npoints) return;

  /* calculate index from CUDA thread index */

  coords_from_index_gpu_d(threadIndex, &ic, &jc, &kc);
  coords_index_gpu_d(ic, jc, kc, &index);


  for (n = 0; n < NQAB; n++) {
    nstr = dev.nsites*n; 

    grad_d[X*dev.nsites*NQAB + nstr + index]
	= 0.5*(field_d[nstr + index + xs] - field_d[nstr + index - xs]);
    grad_d[Y*dev.nsites*NQAB + nstr + index]
	= 0.5*(field_d[nstr + index + ys] - field_d[nstr + index - ys]);
    grad_d[Z*dev.nsites*NQAB + nstr + index]
	 = 0.5*(field_d[nstr + index + 1] - field_d[nstr + index - 1]);

    del2_d[nstr + index]
	  = field_d[nstr + index + xs] + field_d[nstr + index - xs]
	  + field_d[nstr + index + ys] + field_d[nstr + index - ys]
	  + field_d[nstr + index + 1] + field_d[nstr + index - 1]
	  - 6.0*field_d[nstr + index];
  }

  return;
}

/*****************************************************************************
 *
 *  gradient_solid_d
 *
 *  Isotropic, and non-iterative, version. Based on 6x6 Gaussian
 *  elimination followed by explicit control for trace.
 *
 *****************************************************************************/
#ifdef KEVIN_GPU
__global__ void gradient_solid_d(const double * __restrict__ field_d,
                                 double * __restrict__ grad_d,
		 		 double * __restrict__ del2_d,
			 	 char * __restrict__ site_map_status_d,
                                 coll_array_t * __restrict__ carry_d) {
#else
__global__ void gradient_solid_d(const double * __restrict__ field_d,
                                 double * __restrict__ grad_d,
		 		 double * __restrict__ del2_d,
			 	 char * __restrict__ site_map_status_d,
				 char * __restrict__ colloid_map_d,
                                 double * __restrict__ colloid_r_d) {
#endif
  int ic, jc, kc, index;
  int threadIndex;
  int str[3];
  int ia, ib, n1, n2;
  int ih, ig;
  int noff, noffv;
  int n, nunknown;
  int status[6];
  int normal[3];
  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  const double bcsign[6] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};

  double gradn[6][3][2];          /* one-sided partial gradients */
  double dq;
  double qs[3][3];
  double c[3][3];

  double bc[6][6][3];
  double b18[18];
  double x18[18];
  double tr;
  const double r3 = (1.0/3.0);

  threadIndex = blockIdx.x*blockDim.x + threadIdx.x;
  if (threadIndex >= dev.npoints) return;

  str[Z] = 1;
  str[Y] = str[Z]*dev.nall[Z];
  str[X] = str[Y]*dev.nall[Y];

  coords_from_index_gpu_d(threadIndex, &ic, &jc, &kc);
  coords_index_gpu_d(ic, jc, kc, &index);

  if (site_map_status_d[index] != FLUID) return;

  noffv = NQAB*dev.nsites;
  nunknown = 0;

  for (ia = 0; ia < 3; ia++) {

    normal[ia] = ia;

    /* Look for outward normals in bcs[] */
    /* -ve direction (+ve outward normal) */

    ib = 2*ia + 1;
    ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
    status[2*ia] = site_map_status_d[index + ib];

    /* +ve direction (-ve outward normal) */
    ib = 2*ia;
    ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
    status[2*ia + 1] = site_map_status_d[index + ib];

    ig = (status[2*ia    ] != FLUID);
    ih = (status[2*ia + 1] != FLUID);

    /* Calculate half-gradients assuming they are all knowns */

    for (n1 = 0; n1 < NQAB; n1++) {
      n = index + dev.nsites*n1;
      gradn[n1][ia][0] = field_d[n + str[ia]] - field_d[n];
      gradn[n1][ia][1] = field_d[n] - field_d[n - str[ia]];
    }

    gradn[ZZ][ia][0] = -gradn[XX][ia][0] - gradn[YY][ia][0];
    gradn[ZZ][ia][1] = -gradn[XX][ia][1] - gradn[YY][ia][1];


    /* Set unknown, with direction, or treat as known (zero grad) */

    if (ig + ih == 1) {
      normal[nunknown] = 2*ia + ih;
      nunknown += 1;
    }
    else if (ig && ih) {
      for (n1 = 0; n1 < NSYMM; n1++) {
        gradn[n1][ia][0] = 0.0;
        gradn[n1][ia][1] = 0.0;
      }
    }
  }

  /* Boundary condition constant terms */

  if (nunknown > 0) {

    /* Fluid Qab at surface */

    qs[X][X] = field_d[dev.nsites*XX + index];
    qs[X][Y] = field_d[dev.nsites*XY + index]; 
    qs[X][Z] = field_d[dev.nsites*XZ + index]; 
    qs[Y][X] = qs[X][Y]; 
    qs[Y][Y] = field_d[dev.nsites*YY + index]; 
    qs[Y][Z] = field_d[dev.nsites*YZ + index]; 
    qs[Z][X] = qs[X][Z]; 
    qs[Z][Y] = qs[Y][Z]; 
    qs[Z][Z] = -qs[X][X] - qs[Y][Y];
#ifdef KEVIN_GPU
    q_boundary_constants_d(ic, jc, kc, qs, bcs[normal[0]], status[normal[0]],
	                   carry_d, c);
#else
    q_boundary_constants_d(ic, jc, kc, qs, bcs[normal[0]], status[normal[0]],
	colloid_map_d, colloid_r_d, c);
#endif
    /* Constant terms all move to RHS (hence -ve sign). Factors
     * of two in off-diagonals agree with matrix coefficients. */

    b18[XX] = -1.0*c[X][X];
    b18[XY] = -2.0*c[X][Y];
    b18[XZ] = -2.0*c[X][Z];
    b18[YY] = -1.0*c[Y][Y];
    b18[YZ] = -2.0*c[Y][Z];
    b18[ZZ] = -1.0*c[Z][Z];

    /* Fill in a known value in the unknown gradient at this
     * boundary so that we can always compute the full gradient
     * as 0.5*(gradn[][][0] + gradn[][][1]) */

    ig = normal[0] / 2;
    ih = normal[0] % 2;
    for (n1 = 0; n1 < NSYMM; n1++) {
      gradn[n1][ig][ih] = gradn[n1][ig][1-ih];
    }
  }

  if (nunknown > 1) {
#ifdef KEVIN_GPU
    q_boundary_constants_d(ic, jc, kc, qs, bcs[normal[1]], status[normal[1]],
	                   carry_d, c);
#else
    q_boundary_constants_d(ic, jc, kc, qs, bcs[normal[1]], status[normal[1]],
	colloid_map_d, colloid_r_d, c);
#endif
    b18[1*NSYMM + XX] = -1.0*c[X][X];
    b18[1*NSYMM + XY] = -2.0*c[X][Y];
    b18[1*NSYMM + XZ] = -2.0*c[X][Z];
    b18[1*NSYMM + YY] = -1.0*c[Y][Y];
    b18[1*NSYMM + YZ] = -2.0*c[Y][Z];
    b18[1*NSYMM + ZZ] = -1.0*c[Z][Z];

    ig = normal[1] / 2;
    ih = normal[1] % 2;
    for (n1 = 0; n1 < NSYMM; n1++) {
      gradn[n1][ig][ih] = gradn[n1][ig][1-ih];
    }

  }

  if (nunknown > 2) {
#ifdef KEVIN_GPU
    q_boundary_constants_d(ic, jc, kc, qs, bcs[normal[2]], status[normal[2]],
			   carry_d, c);
#else
    q_boundary_constants_d(ic, jc, kc, qs, bcs[normal[2]], status[normal[2]],
	colloid_map_d, colloid_r_d, c);
#endif
    b18[2*NSYMM + XX] = -1.0*c[X][X];
    b18[2*NSYMM + XY] = -2.0*c[X][Y];
    b18[2*NSYMM + XZ] = -2.0*c[X][Z];
    b18[2*NSYMM + YY] = -1.0*c[Y][Y];
    b18[2*NSYMM + YZ] = -2.0*c[Y][Z];
    b18[2*NSYMM + ZZ] = -1.0*c[Z][Z];

    ig = normal[2] / 2;
    ih = normal[2] % 2;
    for (n1 = 0; n1 < NSYMM; n1++) {
      gradn[n1][ig][ih] = gradn[n1][ig][1-ih];
    }
  }

  /* Now solve */

  if (nunknown == 1) {

    /* Set up right hand side (two non-normal directions) */
    /* Special case A matrix is diagonal. */

    gradient_bcs6x6_coeff_d(dev.kappa0, dev.kappa1, bcs[normal[0]], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {

	for (ia = 0; ia < 3; ia++) {
	  dq = 0.5*(gradn[n2][ia][0] + gradn[n2][ia][1]);
          b18[n1] -= bc[n1][n2][ia]*dq;
        }

	dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
	b18[n1] += bc[n1][n2][normal[0]/2]*dq;
      }
      b18[n1] *= bcsign[normal[0]];
      x18[n1] = dev.a6inv[normal[0]/2][n1]*b18[n1];
    }
  }

  if (nunknown == 2) {

    if (normal[0]/2 == X && normal[1]/2 == Y) normal[2] = Z;
    if (normal[0]/2 == X && normal[1]/2 == Z) normal[2] = Y;
    if (normal[0]/2 == Y && normal[1]/2 == Z) normal[2] = X;

    gradient_bcs6x6_coeff_d(dev.kappa0, dev.kappa1, bcs[normal[0]], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {

	dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
        b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;

	dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
        b18[n1] -= bc[n1][n2][normal[2]]*dq;
      }
    }

    gradient_bcs6x6_coeff_d(dev.kappa0, dev.kappa1, bcs[normal[1]], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {

	dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
        b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;

	dq = 0.5*(gradn[n2][normal[2]][0] + gradn[n2][normal[2]][1]);
        b18[NSYMM + n1] -= bc[n1][n2][normal[2]]*dq;
      }
    }

    /* Solve x = A^-1 b */

    ia = normal[0]/2 + normal[1]/2 - 1;

    for (n1 = 0; n1 < 2*NSYMM; n1++) {
      x18[n1] = 0.0;
      for (n2 = 0; n2 < NSYMM; n2++) {
        x18[n1] += bcsign[normal[0]]*dev.a12inv[ia][n1][n2]*b18[n2];
      }
      for (n2 = NSYMM; n2 < 2*NSYMM; n2++) {
        x18[n1] += bcsign[normal[1]]*dev.a12inv[ia][n1][n2]*b18[n2];
      }
    }
  }

  if (nunknown == 3) {

    gradient_bcs6x6_coeff_d(dev.kappa0, dev.kappa1, bcs[normal[0]], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {
        dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
        b18[n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;

	dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	b18[n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
      }
      b18[n1] *= bcsign[normal[0]];
    }

    gradient_bcs6x6_coeff_d(dev.kappa0, dev.kappa1, bcs[normal[1]], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {
	dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
        b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;

	dq = 0.5*(gradn[n2][normal[2]/2][0] + gradn[n2][normal[2]/2][1]);
	b18[NSYMM + n1] -= 0.5*bc[n1][n2][normal[2]/2]*dq;
      }
      b18[NSYMM + n1] *= bcsign[normal[1]];
    }

    gradient_bcs6x6_coeff_d(dev.kappa0, dev.kappa1, bcs[normal[2]], bc);

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < NSYMM; n2++) {
	dq = 0.5*(gradn[n2][normal[0]/2][0] + gradn[n2][normal[0]/2][1]);
        b18[2*NSYMM + n1] -= 0.5*bc[n1][n2][normal[0]/2]*dq;

	dq = 0.5*(gradn[n2][normal[1]/2][0] + gradn[n2][normal[1]/2][1]);
	b18[2*NSYMM + n1] -= 0.5*bc[n1][n2][normal[1]/2]*dq;
      }
      b18[2*NSYMM + n1] *= bcsign[normal[2]];
    }

    /* Solve x = A^-1 b */

    for (n1 = 0; n1 < 3*NSYMM; n1++) {
      x18[n1] = 0.0;
      for (n2 = 0; n2 < 3*NSYMM; n2++) {
        x18[n1] += dev.a18inv[n1][n2]*b18[n2];
      }
    }
  }

  for (n = 0; n < nunknown; n++) {

    /* Fix trace (don't care about Qzz in the end) */

    tr = r3*(x18[NSYMM*n + XX] + x18[NSYMM*n + YY] + x18[NSYMM*n + ZZ]);
    x18[NSYMM*n + XX] -= tr;
    x18[NSYMM*n + YY] -= tr;

    /* Store missing half gradients */

    for (n1 = 0; n1 < NQAB; n1++) {
      gradn[n1][normal[n]/2][normal[n] % 2] = x18[NSYMM*n + n1];
    }
  }

  /* The final answer is the sum of partial gradients */

  for (n1 = 0; n1 < NQAB; n1++) {
    noff = dev.nsites*n1 + index;
    del2_d[noff] = 0.0;
    for (ia = 0; ia < 3; ia++) {
      grad_d[ia*noffv + noff] = 0.5*(gradn[n1][ia][0] + gradn[n1][ia][1]);
      del2_d[noff] += (gradn[n1][ia][0] - gradn[n1][ia][1]);
    }
  } 

  return;
}

/*****************************************************************************
 *
 *  q_boundary_constants_d
 *
 *  Compute constant terms in boundary condition equation.
 *
 *****************************************************************************/
#ifdef KEVIN_GPU
__device__ void q_boundary_constants_d(int ic, int jc, int kc, double qs[3][3],
                                       const int di[3],
                                       int status,
               coll_array_t * carry_d, double c[3][3]) {
#else
__device__ void q_boundary_constants_d(int ic, int jc, int kc, double qs[3][3],
                                       const int di[3],
                                       int status, char * colloid_map_d,
                                       double * colloid_r_d, double c[3][3]) {
#endif
  int index;
  int cid;

  int ia, ib, ig, ih;
  int anchor;

  double w1, w2;
  double dnhat[3];
  double qtilde[3][3];
  double q0[3][3];
  double q2 = 0.0;
  double rd;

  coords_index_gpu_d(ic - di[X], jc - di[Y], kc - di[Z], &index);

  /* Default -> outward normal, ie., flat wall */

  w1 = dev.w1_wall;
  w2 = dev.w2_wall;
  anchor = dev.ntype_wall;

  dnhat[X] = 1.0*di[X];
  dnhat[Y] = 1.0*di[Y];
  dnhat[Z] = 1.0*di[Z];

  if (status == COLLOID) {

    w1 = dev.w1_coll;
    w2 = dev.w2_coll;
    anchor = dev.ntype_coll;
#ifdef KEVIN_GPU
    cid = carry_d->mapd[index];
    if (cid == -1) printf("index: bad %3d %3d %3d %d\n", ic - di[X], jc - di[Y], kc - di[Z], index);
    dnhat[X] = 1.0*(dev.noffset[X] + ic) - carry_d->s[cid].r[X];
    dnhat[Y] = 1.0*(dev.noffset[Y] + jc) - carry_d->s[cid].r[Y];
    dnhat[Z] = 1.0*(dev.noffset[Z] + kc) - carry_d->s[cid].r[Z];
#else
    cid = colloid_map_d[index];
    dnhat[X] = 1.0*(dev.noffset[X] + ic) - colloid_r_d[3*cid + X];
    dnhat[Y] = 1.0*(dev.noffset[Y] + jc) - colloid_r_d[3*cid + Y];
    dnhat[Z] = 1.0*(dev.noffset[Z] + kc) - colloid_r_d[3*cid + Z];
#endif
    /* unit vector */
    rd = 1.0/sqrt(dnhat[X]*dnhat[X] + dnhat[Y]*dnhat[Y] + dnhat[Z]*dnhat[Z]);
    dnhat[X] *= rd;
    dnhat[Y] *= rd;
    dnhat[Z] *= rd;
  }

  if (anchor == ANCHORING_NORMAL) {

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        q0[ia][ib] = 0.5*dev.amp*(3.0*dnhat[ia]*dnhat[ib] - d_cd[ia][ib]);
        qtilde[ia][ib] = 0.0;
      }
    }

  }
  else { /* PLANAR */

    q2 = 0.0;
    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        qtilde[ia][ib] = qs[ia][ib] + 0.5*dev.amp*d_cd[ia][ib];
        q2 += qtilde[ia][ib]*qtilde[ia][ib];
      }
    }

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        q0[ia][ib] = 0.0;
        for (ig = 0; ig < 3; ig++) {
          for (ih = 0; ih < 3; ih++) {
            q0[ia][ib] += (d_cd[ia][ig] - dnhat[ia]*dnhat[ig])*qtilde[ig][ih]
              *(d_cd[ih][ib] - dnhat[ih]*dnhat[ib]);
          }
        }

        q0[ia][ib] -= 0.5*dev.amp*d_cd[ia][ib];
      }
    }

  }


  /* Compute c[a][b] */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {

      c[ia][ib] = 0.0;

      for (ig = 0; ig < 3; ig++) {
        for (ih = 0; ih < 3; ih++) {
          c[ia][ib] -= dev.kappa1*dev.q0*di[ig]*
                (e_cd[ia][ig][ih]*qs[ih][ib] + e_cd[ib][ig][ih]*qs[ih][ia]);
        }
      }

      /* Normal anchoring: w2 must be zero and q0 is preferred Q
       * Planar anchoring: in w1 term q0 is effectively
       *                   (Qtilde^perp - 0.5S_0) while in w2 we
       *                   have Qtilde appearing explicitly.
       *                   See colloids_q_boundary() etc */

      c[ia][ib] +=
              -w1*(qs[ia][ib] - q0[ia][ib])
              -w2*(2.0*q2 - 4.5*dev.amp*dev.amp)*qtilde[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  coords_index_gpu_d
 *
 *  For lattice position (ic, jc, kc), map to memory index allowing
 *  for halo and kernel extent.
 *
 *****************************************************************************/

__device__
static int coords_index_gpu_d(int ic, int jc, int kc, int * index) {

  int zs = 1;
  int ys = zs*dev.nall[Z];
  int xs = ys*dev.nall[Y];

  *index = xs*(ic + dev.nhalo - dev.nextra)
         + ys*(jc + dev.nhalo - dev.nextra)
         + zs*(kc + dev.nhalo - dev.nextra);

  return 0;
}

/*****************************************************************************
 *
 *  gradient_bcs6x6_block_gpu
 *
 *  Boundary condition coefficients for block dimension id
 *
 *****************************************************************************/

__host__ __device__
static void gradient_bcs6x6_coeff_d(double kappa0, double kappa1,
                                    const int dn[3], double bc[6][6][3]) {
  double kappa2;

  kappa2 = kappa0 + kappa1;

  /* XX equation */

  bc[XX][XX][X] =  kappa0*dn[X];
  bc[XX][XY][X] = -kappa1*dn[Y];
  bc[XX][XZ][X] = -kappa1*dn[Z];
  bc[XX][YY][X] =  0.0;
  bc[XX][YZ][X] =  0.0;
  bc[XX][ZZ][X] =  0.0;

  bc[XX][XX][Y] =  kappa1*dn[Y];
  bc[XX][XY][Y] =  kappa0*dn[X];
  bc[XX][XZ][Y] =  0.0;
  bc[XX][YY][Y] =  0.0;
  bc[XX][YZ][Y] =  0.0;
  bc[XX][ZZ][Y] =  0.0;

  bc[XX][XX][Z] =  kappa1*dn[Z];
  bc[XX][XY][Z] =  0.0;
  bc[XX][XZ][Z] =  kappa0*dn[X];
  bc[XX][YY][Z] =  0.0;
  bc[XX][YZ][Z] =  0.0;
  bc[XX][ZZ][Z] =  0.0;

  /* XY equation */

  bc[XY][XX][X] =  kappa0*dn[Y];
  bc[XY][XY][X] =  kappa2*dn[X];
  bc[XY][XZ][X] =  0.0;
  bc[XY][YY][X] = -kappa1*dn[Y];
  bc[XY][YZ][X] = -kappa1*dn[Z];
  bc[XY][ZZ][X] =  0.0;

  bc[XY][XX][Y] = -kappa1*dn[X];
  bc[XY][XY][Y] =  kappa2*dn[Y];
  bc[XY][XZ][Y] = -kappa1*dn[Z];
  bc[XY][YY][Y] =  kappa0*dn[X];
  bc[XY][YZ][Y] =  0.0;
  bc[XY][ZZ][Y] =  0.0;

  bc[XY][XX][Z] =  0.0;
  bc[XY][XY][Z] =  2.0*kappa1*dn[Z];
  bc[XY][XZ][Z] =  kappa0*dn[Y];
  bc[XY][YY][Z] =  0.0;
  bc[XY][YZ][Z] =  kappa0*dn[X];
  bc[XY][ZZ][Z] =  0.0;

  /* XZ equation */

  bc[XZ][XX][X] =  kappa0*dn[Z];
  bc[XZ][XY][X] =  0.0;
  bc[XZ][XZ][X] =  kappa2*dn[X];
  bc[XZ][YY][X] =  0.0;
  bc[XZ][YZ][X] = -kappa1*dn[Y];
  bc[XZ][ZZ][X] = -kappa1*dn[Z];

  bc[XZ][XX][Y] =  0.0;
  bc[XZ][XY][Y] =  kappa0*dn[Z];
  bc[XZ][XZ][Y] =  2.0*kappa1*dn[Y];
  bc[XZ][YY][Y] =  0.0;
  bc[XZ][YZ][Y] =  kappa0*dn[X];
  bc[XZ][ZZ][Y] =  0.0;

  bc[XZ][XX][Z] = -kappa1*dn[X];
  bc[XZ][XY][Z] = -kappa1*dn[Y];
  bc[XZ][XZ][Z] =  kappa2*dn[Z];
  bc[XZ][YY][Z] =  0.0;
  bc[XZ][YZ][Z] =  0.0;
  bc[XZ][ZZ][Z] =  kappa0*dn[X];

  /* YY equation */

  bc[YY][XX][X] =  0.0;
  bc[YY][XY][X] =  kappa0*dn[Y];
  bc[YY][XZ][X] =  0.0;
  bc[YY][YY][X] =  kappa1*dn[X];
  bc[YY][YZ][X] =  0.0;
  bc[YY][ZZ][X] =  0.0;

  bc[YY][XX][Y] =  0.0;
  bc[YY][XY][Y] = -kappa1*dn[X];
  bc[YY][XZ][Y] =  0.0;
  bc[YY][YY][Y] =  kappa0*dn[Y];
  bc[YY][YZ][Y] = -kappa1*dn[Z];
  bc[YY][ZZ][Y] =  0.0;

  bc[YY][XX][Z] =  0.0;
  bc[YY][XY][Z] =  0.0;
  bc[YY][XZ][Z] =  0.0;
  bc[YY][YY][Z] =  kappa1*dn[Z];
  bc[YY][YZ][Z] =  kappa0*dn[Y];
  bc[YY][ZZ][Z] =  0.0;

  /* YZ equation */

  bc[YZ][XX][X] =  0.0;
  bc[YZ][XY][X] =  kappa0*dn[Z];
  bc[YZ][XZ][X] =  kappa0*dn[Y];
  bc[YZ][YY][X] =  0.0;
  bc[YZ][YZ][X] =  2.0*kappa1*dn[X];
  bc[YZ][ZZ][X] =  0.0;

  bc[YZ][XX][Y] =  0.0;
  bc[YZ][XY][Y] =  0.0;
  bc[YZ][XZ][Y] = -kappa1*dn[X];
  bc[YZ][YY][Y] =  kappa0*dn[Z];
  bc[YZ][YZ][Y] =  kappa2*dn[Y];
  bc[YZ][ZZ][Y] = -kappa1*dn[Z];

  bc[YZ][XX][Z] =  0.0;
  bc[YZ][XY][Z] = -kappa1*dn[X];
  bc[YZ][XZ][Z] =  0.0;
  bc[YZ][YY][Z] = -kappa1*dn[Y];
  bc[YZ][YZ][Z] =  kappa2*dn[Z];
  bc[YZ][ZZ][Z] =  kappa0*dn[Y];

  /* ZZ equation */

  bc[ZZ][XX][X] =  0.0;
  bc[ZZ][XY][X] =  0.0;
  bc[ZZ][XZ][X] =  kappa0*dn[Z];
  bc[ZZ][YY][X] =  0.0;
  bc[ZZ][YZ][X] =  0.0;
  bc[ZZ][ZZ][X] =  kappa1*dn[X];
  
  bc[ZZ][XX][Y] =  0.0;
  bc[ZZ][XY][Y] =  0.0;
  bc[ZZ][XZ][Y] =  0.0;
  bc[ZZ][YY][Y] =  0.0;
  bc[ZZ][YZ][Y] =  kappa0*dn[Z];
  bc[ZZ][ZZ][Y] =  kappa1*dn[Y];
  
  bc[ZZ][XX][Z] =  0.0;
  bc[ZZ][XY][Z] =  0.0;
  bc[ZZ][XZ][Z] = -kappa1*dn[X];
  bc[ZZ][YY][Z] =  0.0;
  bc[ZZ][YZ][Z] = -kappa1*dn[Y];
  bc[ZZ][ZZ][Z] =  kappa0*dn[Z];

  return;
}

void set_gradient_option_gpu(char option){

  gradient_gpu=option;

}

/* get 3d coordinates from the index on the accelerator */

__device__
static void coords_from_index_gpu_d(int index, int *ic, int *jc, int *kc) {

  int yfac = dev.nxtent[Z];
  int xfac = dev.nxtent[Y]*yfac;
  
  *ic = index/xfac;
  *jc = (index - xfac*(*ic))/yfac;
  *kc = index - (*ic)*xfac - (*jc)*yfac;

  return;

}
