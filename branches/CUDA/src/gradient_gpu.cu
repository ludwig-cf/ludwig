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

#include "colloids_Q_tensor.h"

/* scheme in use */
static char gradient_gpu=-1;

__constant__ int Ngradcalc_cd[3];
__constant__ int nextra_cd;

#ifdef KEVIN_GPU

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
};

__global__ void gradient_fluid_d(const double* __restrict__ field_d,
                                 double * __restrict__ grad_d,
                                 double* __restrict__ del2_d,
				 gradient_gpu_t * cdevice);


__global__ void gradient_solid_d(const double * __restrict__ field_d,
                                 double * __restrict__ grad_d,
		 		 double * __restrict__ del2_d,
			 	 char * __restrict__ site_map_status_d,
                                 char * __restrict__ colloid_map_d,
                                 double * __restrict__ colloid_r_d,
			         gradient_gpu_t * cdevice);

__device__ static int coords_index_gpu_d(gradient_gpu_t * cdevice,
	int ic, int jc, int kc, int * index);
__device__ static void gradient_bcs6x6_coeff_d(double kappa0, double kappa1,
                                               const int dn[3],
                                               double bc[6][6][3]);
__device__ static void util_gauss_solve_d(int mrow, double a[][18], double * x,
	                                  int * pivot);
__device__ static void q_boundary_normal_gpu_d(
	gradient_gpu_t * cdev,
	int ic, int jc, int kc,
	const int di[3],
        int status,
        char * __restrict__ colloid_map_d,
        double * __restrict__ colloid_r_d,
	double dn[3]);
__device__ void q_boundary_gpu_d(gradient_gpu_t * cdev, double nhat[3],
	double qs[3][3], double q0[3][3], int status);

int phi_gradients_compute_gpu() {

  int nblocks;
  int ndefault;
  dim3 nblock;
  dim3 ntpb;


  gradient_gpu_t host;
  gradient_gpu_t * device;

  put_gradient_constants_on_gpu(); /* for d_cd, e_cd constants */

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

  cudaMalloc((void **) &device, sizeof(gradient_gpu_t));
  cudaMemcpy(device, &host, sizeof(gradient_gpu_t), cudaMemcpyHostToDevice);

  /* Points required and thread blocks */

  ndefault = DEFAULT_TPB;
  nblocks = (host.npoints + ndefault - 1) / ndefault;
  nblock.x = nblocks; nblock.y = 1; nblock.z = 1;
  ntpb.x = ndefault; ntpb.y = 1; ntpb.z = 1;

  if (gradient_gpu == OPTION_3D_7PT_FLUID) {
    cudaConfigureCall(nblock, ntpb, 0, 0);
    cudaLaunch("gradient_fluid_d");
  }
  if (gradient_gpu == OPTION_3D_7PT_SOLID) {
    gradient_solid_d<<<nblocks,ndefault>>>(phi_site_d,
                                     	      grad_phi_site_d,
		 		              delsq_phi_site_d,
			 	              site_map_status_d,
                                              colloid_map_d,
                                              colloid_r_d,
			                      device);
  }

  cudaThreadSynchronize();
  checkCUDAError("gradient_3d_7pt");  
  cudaFree(device);

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
				 double* __restrict__ del2_d,
			         gradient_gpu_t * cdevice) {
  int n;
  int index;
  int threadIndex,ii, jj, kk;
  int ys = cdevice->nall[Z];
  int xs = ys*cdevice->nall[Y];
  int nstr;

  threadIndex = blockIdx.x*blockDim.x + threadIdx.x;

  if (threadIndex >= cdevice->npoints) return;

  /* calculate index from CUDA thread index */

  get_coords_from_index_gpu_d(&ii, &jj, &kk, threadIndex, cdevice->nxtent);
  index = get_linear_index_gpu_d(ii + cdevice->nhalo - cdevice->nextra,
	                         jj + cdevice->nhalo - cdevice->nextra,
				 kk + cdevice->nhalo - cdevice->nextra,
				 cdevice->nall);      

  for (n = 0; n < NQAB; n++) {
    nstr = cdevice->nsites*n; 

    grad_d[X*cdevice->nsites*NQAB + nstr + index]
	= 0.5*(field_d[nstr + index + xs] - field_d[nstr + index - xs]);
    grad_d[Y*cdevice->nsites*NQAB + nstr + index]
	= 0.5*(field_d[nstr + index + ys] - field_d[nstr + index - ys]);
    grad_d[Z*cdevice->nsites*NQAB + nstr + index]
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

__global__ void gradient_solid_d(const double * __restrict__ field_d,
                                 double * __restrict__ grad_d,
		 		 double * __restrict__ del2_d,
			 	 char * __restrict__ site_map_status_d,
				 char * __restrict__ colloid_map_d,
                                 double * __restrict__ colloid_r_d,
			         gradient_gpu_t * cdevice) {

  int ic, jc, kc, index;
  int threadIndex;
  int str[3];
  int ia, ib, n1, n2;
  int ih, ig;
  int noff, noffv;
  int n, nunknown;
  int status[6];
  int normal[3];
  int known[3];
  int idb, jcol;
  int pivot18[18];
  const int bcs[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};

  double gradn[6][3][2];          /* one-sided partial gradients */
  double dq;
  double q2;
  double qs[3][3];
  double q0[3][3];
  double qtilde[3][3];
  double c[3][3];
  double dn[3];
  double w1, w2;

  double a18[18][18];
  double xb18[18];
  double bc[NSYMM][NSYMM][3];
  double tr;
  double unkn;
  double diag;
  const double r3 = (1.0/3.0);

  threadIndex = blockIdx.x*blockDim.x + threadIdx.x;

  if (threadIndex >= cdevice->npoints) return;

  str[Z] = 1;
  str[Y] = str[Z]*cdevice->nall[Z];
  str[X] = str[Y]*cdevice->nall[Y];

  get_coords_from_index_gpu_d(&ic, &jc, &kc, threadIndex, cdevice->nxtent);
  coords_index_gpu_d(cdevice, ic, jc, kc, &index);

  if (site_map_status_d[index] != FLUID) return;

  noffv = NQAB*cdevice->nsites;
  nunknown = 0;

  for (ia = 0; ia < 3; ia++) {

    known[ia] = 1;
    normal[ia] = ia;

    /* Look for outward normals is bcs[] */

    ib = 2*ia + 1;
    ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
    status[2*ia] = site_map_status_d[index + ib];

    ib = 2*ia;
    ib = bcs[ib][X]*str[X] + bcs[ib][Y]*str[Y] + bcs[ib][Z]*str[Z];
    status[2*ia + 1] = site_map_status_d[index + ib];

    ig = (status[2*ia    ] != FLUID);
    ih = (status[2*ia + 1] != FLUID);

    /* Calculate half-gradients assuming they are all knowns */

    for (n1 = 0; n1 < NQAB; n1++) {
      n = index + cdevice->nsites*n1;
      gradn[n1][ia][0] = field_d[n + str[ia]] - field_d[n];
      gradn[n1][ia][1] = field_d[n] - field_d[n - str[ia]];
    }

    gradn[ZZ][ia][0] = -gradn[XX][ia][0] - gradn[YY][ia][0];
    gradn[ZZ][ia][1] = -gradn[XX][ia][1] - gradn[YY][ia][1];


    /* Set unknown, with direction, or treat as known (zero grad) */

    if (ig + ih == 1) {
      known[ia] = 0;
      normal[nunknown] = 2*ia + ih;
      nunknown += 1;
    }
    else if (ig && ih) {
      for (n1 = 0; n1 < NQAB; n1++) {
        gradn[n1][ia][0] = 0.0;
        gradn[n1][ia][1] = 0.0;
      }
    }
  }

  /* For planar anchoring we require qtilde_ab of Fournier and
   * Galatola, and its square */

  qs[X][X] = field_d[cdevice->nsites*XX + index];
  qs[X][Y] = field_d[cdevice->nsites*XY + index]; 
  qs[X][Z] = field_d[cdevice->nsites*XZ + index]; 
  qs[Y][X] = qs[X][Y]; 
  qs[Y][Y] = field_d[cdevice->nsites*YY + index]; 
  qs[Y][Z] = field_d[cdevice->nsites*YZ + index]; 
  qs[Z][X] = qs[X][Z]; 
  qs[Z][Y] = qs[Y][Z]; 
  qs[Z][Z] = -qs[X][X] - qs[Y][Y];

  q2 = 0.0;
  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      qtilde[ia][ib] = qs[ia][ib] + 0.5*cdevice->amp*d_cd[ia][ib];
      q2 += qtilde[ia][ib]*qtilde[ia][ib]; 
    }
  }

  /* For each solid boundary, set up the boundary condition terms */

  for (n = 0; n < nunknown; n++) {

    q_boundary_normal_gpu_d(cdevice, ic, jc, kc, bcs[normal[n]],
	status[normal[n]], colloid_map_d, colloid_r_d, dn);
    q_boundary_gpu_d(cdevice, dn, qs, q0, status[normal[n]]);


    /* Check for wall/colloid */
    if (status[normal[n]] == COLLOID) {
      w1 = cdevice->w1_coll;
      w2 = cdevice->w2_coll;
    }

    if (status[normal[n]] == BOUNDARY) {
      w1 = cdevice->w1_wall;
      w2 = cdevice->w2_wall;
    }

    /* Compute c[a][b] */

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {

        c[ia][ib] = 0.0;

        for (ig = 0; ig < 3; ig++) {
          for (ih = 0; ih < 3; ih++) {
            c[ia][ib] -= cdevice->kappa1*cdevice->q0*bcs[normal[n]][ig]*
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
              -w2*(2.0*q2 - 4.5*cdevice->amp*cdevice->amp)*qtilde[ia][ib];
      }
    }

    /* Now set up the system */
    /* Initialise whole rows of A and b */

    for (n1 = 0; n1 < NSYMM; n1++) {
      for (n2 = 0; n2 < 3*NSYMM; n2++) {
        a18[NSYMM*n + n1][n2] = 0.0;
      }
      xb18[NSYMM*n + n1] = 0.0;
    }

    gradient_bcs6x6_coeff_d(cdevice->kappa0, cdevice->kappa1,
	bcs[normal[n]], bc);

    /* Three blocks of columns for each row; note that the index
     * ib is used to compute known terms, while idb is the
     * appropriate normal direction for unknowns (counted by jcol) */

    jcol = 0;

    for (ib = 0; ib < 3; ib++) {

      diag = d_cd[normal[n]/2][ib]; /* block diagonal? */
      unkn = 1.0 - known[ib];       /* is ib unknown direction? */
      idb = normal[jcol]/2;         /* normal direction for this unknown */

      for (n1 = 0; n1 < NSYMM; n1++) {
        for (n2 = 0; n2 < NSYMM; n2++) {

          /* Unknown diagonal blocks contribute full bc to A */
          /* Unknown off-diagonal blocks contribute (1/2) bc
           * to A and (1/2) known dq to RHS.
           * Known off-diagonals to RHS */

          a18[NSYMM*n + n1][NSYMM*jcol + n2] += unkn*diag*bc[n1][n2][idb];
          a18[NSYMM*n + n1][NSYMM*jcol + n2] += 0.5*unkn*(1.0 - diag)*bc[n1][n2][idb];

          dq = gradn[n2][idb][1 - (normal[jcol] % 2)];
          xb18[NSYMM*n + n1] += 0.5*unkn*(1.0 - diag)*bc[n1][n2][idb]*dq;

          dq = 0.5*known[ib]*(gradn[n2][ib][0] + gradn[n2][ib][1]);
          xb18[NSYMM*n + n1] += (1.0 - diag)*bc[n1][n2][ib]*dq;
        }
      }

      jcol += (1 - known[ib]);
    }

    /* Constant terms all move to RHS (hence -ve sign). Factors
     * of two in off-diagonals agree with coefficients. */

    xb18[NSYMM*n + XX] = -(xb18[NSYMM*n + XX] +     c[X][X]);
    xb18[NSYMM*n + XY] = -(xb18[NSYMM*n + XY] + 2.0*c[X][Y]);
    xb18[NSYMM*n + XZ] = -(xb18[NSYMM*n + XZ] + 2.0*c[X][Z]);
    xb18[NSYMM*n + YY] = -(xb18[NSYMM*n + YY] +     c[Y][Y]);
    xb18[NSYMM*n + YZ] = -(xb18[NSYMM*n + YZ] + 2.0*c[Y][Z]);
    xb18[NSYMM*n + ZZ] = -(xb18[NSYMM*n + ZZ] +     c[Z][Z]);
  }


  if (nunknown > 0) util_gauss_solve_d(NSYMM*nunknown, a18, xb18, pivot18);

  for (n = 0; n < nunknown; n++) {

    /* Fix trace (don't care about Qzz in the end) */

    tr = r3*(xb18[NSYMM*n + XX] + xb18[NSYMM*n + YY] + xb18[NSYMM*n + ZZ]);
    xb18[NSYMM*n + XX] -= tr;
    xb18[NSYMM*n + YY] -= tr;

    /* Store missing half gradients */

    for (n1 = 0; n1 < NQAB; n1++) {
      gradn[n1][normal[n]/2][normal[n] % 2] = xb18[NSYMM*n + n1];
    }
  }

  /* The final answer is the sum of partial gradients */

  for (n1 = 0; n1 < NQAB; n1++) {
    noff = cdevice->nsites*n1 + index;
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
 *  coords_index_gpu_d
 *
 *  For lattice position (ic, jc, kc), map to memory index allowing
 *  for halo and kernel extent.
 *
 *****************************************************************************/

__device__ static int coords_index_gpu_d(gradient_gpu_t * cdevice,
	int ic, int jc, int kc, int * index) {

  int zs = 1;
  int ys = zs*cdevice->nall[Z];
  int xs = ys*cdevice->nall[Y];

  *index = xs*(ic + cdevice->nhalo - cdevice->nextra)
         + ys*(jc + cdevice->nhalo - cdevice->nextra)
         + zs*(kc + cdevice->nhalo - cdevice->nextra);

  return 0;
}

/*****************************************************************************
 *
 *  q_boundary_normal_gpu_d
 *
 *****************************************************************************/

static __device__ void q_boundary_normal_gpu_d(
	gradient_gpu_t * cdev,
	int ic, int jc, int kc,
	const int di[3],
        int status,
        char * __restrict__ colloid_map_d,
        double * __restrict__ colloid_r_d,
	double dn[3]) {

  int index;
  int cid;
  double rd;

  coords_index_gpu_d(cdev, ic - di[X], jc - di[Y], kc - di[Z], &index);

  /* Default -> outward normal, ie., flat wall */
  dn[X] = 1.0*di[X];
  dn[Y] = 1.0*di[Y];
  dn[Z] = 1.0*di[Z];

  if (status == COLLOID) {

    cid = colloid_map_d[index];

    dn[X] = 1.0*(cdev->noffset[X] + ic) - colloid_r_d[3*cid + X];
    dn[Y] = 1.0*(cdev->noffset[Y] + jc) - colloid_r_d[3*cid + Y];
    dn[Z] = 1.0*(cdev->noffset[Z] + kc) - colloid_r_d[3*cid + Z];

    /* unit vector */
    rd = 1.0/sqrt(dn[X]*dn[X] + dn[Y]*dn[Y] + dn[Z]*dn[Z]);
    dn[X] *= rd;
    dn[Y] *= rd;
    dn[Z] *= rd;
  }

  return;
}

/*****************************************************************************
 *
 *  q_boundary_gpu_d
 *
 *  No fixed anchoring.
 *
 *****************************************************************************/

__device__ void q_boundary_gpu_d(gradient_gpu_t * cdev, double nhat[3],
	double qs[3][3], double q0[3][3], int status) {

  int ia, ib, ic, id;
  int anchor;
  double qtilde[3][3];

  anchor = cdev->ntype_wall;
  if (status == COLLOID) anchor = cdev->ntype_coll;

  if (anchor == ANCHORING_NORMAL) {

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        q0[ia][ib] = 0.5*cdev->amp*(3.0*nhat[ia]*nhat[ib] - d_cd[ia][ib]);
      }
    }

  }
  else { /* PLANAR */

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        qtilde[ia][ib] = qs[ia][ib] + 0.5*cdev->amp*d_cd[ia][ib];
      }
    }

    for (ia = 0; ia < 3; ia++) {
      for (ib = 0; ib < 3; ib++) {
        q0[ia][ib] = 0.0;
        for (ic = 0; ic < 3; ic++) {
          for (id = 0; id < 3; id++) {
            q0[ia][ib] += (d_cd[ia][ic] - nhat[ia]*nhat[ic])*qtilde[ic][id]
              *(d_cd[id][ib] - nhat[id]*nhat[ib]);
          }
        }
        /* Return Q^0_ab = ~Q_ab - (1/2) A d_ab */
        q0[ia][ib] -= 0.5*cdev->amp*d_cd[ia][ib];
      }
    }


  }

  return;
}

/*****************************************************************************
 *
 *  gradient_bcs6x6_block_gpu
 *
 *  Boundary condition coefficients for block dimension id
 *
 *****************************************************************************/

static __device__ void gradient_bcs6x6_coeff_d(double kappa0, double kappa1,
                                               const int dn[3],
                                               double bc[6][6][3]) {
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

/*****************************************************************************
 *
 *  util_gauss_solve
 *
 *  Solve linear system via Gaussian elimination. For the problems in this
 *  file, we only need to exchange rows, ie., have a partial pivot.
 *
 *  We solve Ax = b for A[MROW][MROW].
 *  x[MROW] is RHS on entry, and solution on exit.
 *  A is destroyed.
 *  Workspace for the pivot rows must be supplied: pivot[MROW].
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

__device__ static void util_gauss_solve_d(int mrow, double a[][18], double * x,
	                                  int * pivot) {
  int i, j, k;
  int iprow;
  double tmp;

  iprow = -1;
  for (k = 0; k < mrow; k++) {
    pivot[k] = -1;
  }

  for (k = 0; k < mrow; k++) {

    /* Find pivot row */
    tmp = 0.0;
    for (i = 0; i < mrow; i++) {
      if (pivot[i] == -1) {
        if (fabs(a[i][k]) >= tmp) {
          tmp = fabs(a[i][k]);
          iprow = i;
        }
      }
    }
    pivot[k] = iprow;

    /* divide pivot row by the pivot element a[iprow][k] */
    /* There is no check for zero pivot element */

    tmp = 1.0 / a[iprow][k];
    for (j = k; j < mrow; j++) {
      a[iprow][j] *= tmp;
    }
    x[iprow] *= tmp;

    /* Subtract the pivot row (scaled) from remaining rows */

    for (i = 0; i < mrow; i++) {
      if (pivot[i] == -1) {
        tmp = a[i][k];
        for (j = k; j < mrow; j++) {
          a[i][j] -= tmp*a[iprow][j];
        }
        x[i] -= tmp*x[iprow];
      }
    }
  }
  /* Now do the back substitution */

  for (i = mrow - 1; i > -1; i--) {
    iprow = pivot[i];
    tmp = x[iprow];
    for (k = i + 1; k < mrow; k++) {
      tmp -= a[iprow][k]*x[pivot[k]];
    }
    x[iprow] = tmp;
  }

  return;
}


#else

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
__global__ void gradient_3d_7pt_fluid_operator_gpu_d(const double* __restrict__ field_d,
						     double* __restrict__ grad_d,
						     double* __restrict__ del2_d,
						     const int* __restrict__ le_index_real_to_buffer_d) {

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


#define NITERATION 1


__global__ void gradient_3d_7pt_solid_gpu_d(int nop, int nhalo, 
						     int N_d[3], 
						     const double* __restrict__ field_d,
						     double* __restrict__ grad_d,
						     double* __restrict__ del2_d,
						     char* __restrict__ site_map_status_d,
					    const char* __restrict__ colloid_map_d,
					    const double* __restrict__ colloid_r_d,
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
						 double dn[3], int Nall[3], int nhalo, int nextra, int ii, int jj, int kk, const char* __restrict__ site_map_status_d,					    const char* __restrict__ colloid_map_d,
						 const double* __restrict__ colloid_r_d) {
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
	dn[ia] -= colloid_r_d[3*colloid_map_d[index1]+ia];
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


#endif

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
  cudaMemcpyToSymbol(Ngradcalc_cd, Ngradcalc, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(Nall_cd, Nall, 3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nhalo_cd, &nhalo, sizeof(int), 0, cudaMemcpyHostToDevice); 
cudaMemcpyToSymbol(nop_cd, &nop, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nsites_cd, &nsites, sizeof(int), 0, cudaMemcpyHostToDevice); 
  cudaMemcpyToSymbol(nextra_cd, &nextra, sizeof(int), 0, cudaMemcpyHostToDevice); 



  q0_=blue_phase_q0();
  kappa0_=blue_phase_kappa0();
  kappa1_=blue_phase_kappa1();
  kappa2_=kappa0_+kappa1_;
  w = colloids_q_tensor_w1();
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

#ifndef KEVIN_GPU
cudaFuncSetCacheConfig(gradient_3d_7pt_solid_gpu_d,cudaFuncCachePreferL1);
#endif

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
