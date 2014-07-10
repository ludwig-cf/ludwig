/*****************************************************************************
 *
 * utilities_gpu.cu
 *  
 * Alan Gray
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "common_gpu.h"

#include "pe.h"
#include "coords.h"

#include "utilities_gpu.h"
#include "utilities_internal_gpu.h"
#include "field_datamgmt_gpu.h"
#include "comms_gpu.h"
#include "util.h"
#include "model.h"
#include "timer.h"


extern "C" char site_map_get_status(int,int,int);
extern "C" char site_map_get_status_index(int);


/* external pointers to data on host*/
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];
extern const double wv[NVEL];
extern const int cv[NVEL][3];
extern const double q_[NVEL][3][3];

extern double * fluxe;
extern double * fluxw;
extern double * fluxy;
extern double * fluxz;

double * ma_d;
double * mi_d;
int * cv_d;
double * q_d;
double * wv_d;
char * site_map_status_d;
char * colloid_map_d;
double * colloid_r_d;
int * N_d;
double * force_global_d;
double * tmpscal1_d;
double * tmpscal2_d;

double * r3_d;
double * d_d;
double * e_d;

double * electric_d;

double * fluxe_d;
double * fluxw_d;
double * fluxy_d;
double * fluxz_d;


/* host memory address pointers for temporary staging of data */

char * site_map_status_temp;
char * colloid_map_temp;

/* data size variables */
static int nhalo;
static int nsites;
static int nop;
static  int N[3];
static  int Nall[3];

extern double * colloid_force_d;

#ifdef KEVIN_GPU
int utilities_init_extra(void);
#endif


/* Perform tasks necessary to initialise accelerator */
void initialise_gpu()
{

  double force_global[3];


  int devicenum=cart_rank()%GPUS_PER_NODE;

  cudaSetDevice(devicenum);

  if (cart_rank()==0){
    cudaGetDevice(&devicenum);
    printf("master rank running on device %d\n",devicenum);
  }

  calculate_data_sizes();
  allocate_memory_on_gpu();

  /* get global force from physics module */
  fluid_body_force(force_global);

  put_site_map_on_gpu();

  /* copy data from host to accelerator */
  cudaMemcpy(N_d, N, 3*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpy(ma_d, ma_, NVEL*NVEL*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mi_d, mi_, NVEL*NVEL*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cv_d, cv, NVEL*3*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpy(wv_d, wv, NVEL*sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpy(q_d, q_, NVEL*3*3*sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpy(force_global_d, force_global, 3*sizeof(double), \
	     cudaMemcpyHostToDevice);

  cudaMemcpy(r3_d, &r3_, sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_d, d_, 3*3*sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpy(e_d, e_, 3*3*3*sizeof(double), cudaMemcpyHostToDevice); 

  

  init_comms_gpu();
  init_field_gpu();


  checkCUDAError("Init GPU");  

#ifdef KEVIN_GPU
  utilities_init_extra();
  colloids_to_gpu(); /* to match call to put site_map on GPU above */
#endif

}

/* Perform tasks necessary to finalise accelerator */
void finalise_gpu()
{


  free_memory_on_gpu();
  finalise_field_gpu();
  //finalise_phi_gpu();
 

  checkCUDAError("Finalise GPU");


}




/* calculate sizes of data - needed for memory copies to accelerator */
static void calculate_data_sizes()
{
  coords_nlocal(N);  
  nhalo = coords_nhalo();  

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsites = Nall[X]*Nall[Y]*Nall[Z];
  nop = phi_nop();


}





/* Allocate memory on accelerator */
static void allocate_memory_on_gpu()
{

  /* temp arrays for staging data on  host */
  site_map_status_temp = (char *) malloc(nsites*sizeof(char));
  colloid_map_temp = (char *) calloc(nsites,sizeof(char));
  
  cudaMalloc((void **) &site_map_status_d, nsites*sizeof(char));
  cudaMalloc((void **) &colloid_map_d, nsites*sizeof(char));
  cudaMalloc((void **) &colloid_r_d, MAX_COLLOIDS*3*sizeof(double));
  cudaMalloc((void **) &ma_d, NVEL*NVEL*sizeof(double));
  cudaMalloc((void **) &mi_d, NVEL*NVEL*sizeof(double));
  cudaMalloc((void **) &cv_d, NVEL*3*sizeof(int));
  cudaMalloc((void **) &wv_d, NVEL*sizeof(double));
  cudaMalloc((void **) &q_d, NVEL*3*3*sizeof(double));
  cudaMalloc((void **) &tmpscal1_d, nsites*sizeof(double));
  cudaMalloc((void **) &tmpscal2_d, nsites*sizeof(double));

  cudaMalloc((void **) &fluxe_d, nop*nsites*sizeof(double));
  cudaMalloc((void **) &fluxw_d, nop*nsites*sizeof(double));
  cudaMalloc((void **) &fluxy_d, nop*nsites*sizeof(double));
  cudaMalloc((void **) &fluxz_d, nop*nsites*sizeof(double));
  
  cudaMalloc((void **) &N_d, sizeof(int)*3);
  cudaMalloc((void **) &force_global_d, sizeof(double)*3);



  cudaMalloc((void **) &r3_d, sizeof(double));
  cudaMalloc((void **) &d_d, sizeof(double)*3*3);
  cudaMalloc((void **) &e_d, sizeof(double)*3*3*3);

  cudaMalloc((void **) &electric_d, sizeof(double)*3);

  checkCUDAError("allocate_memory_on_gpu");

}


/* Free memory on accelerator */
static void free_memory_on_gpu()
{

  /* free temp memory on host */
  free(site_map_status_temp);
  free(colloid_map_temp);

  cudaFree(ma_d);
  cudaFree(mi_d);
  cudaFree(cv_d);
  cudaFree(wv_d);
  cudaFree(q_d);
  cudaFree(site_map_status_d);
  cudaFree(colloid_map_d);
  cudaFree(colloid_r_d);
  cudaFree(N_d);
  cudaFree(force_global_d);

  cudaFree(tmpscal1_d);
  cudaFree(tmpscal2_d);

  cudaFree(fluxe_d);
  cudaFree(fluxw_d);
  cudaFree(fluxy_d);
  cudaFree(fluxz_d);
 
  cudaFree(r3_d);
  cudaFree(d_d);
  cudaFree(e_d);

  cudaFree(electric_d);

  checkCUDAError("free_memory_on_gpu");
}

__global__ void printsitemap4421(char * site_map_status_d){

  printf("PPP %d\n",site_map_status_d[4421]);

}

/* copy site map from host to accelerator */
void put_site_map_on_gpu()
{

  int index, ic, jc, kc;
	      

  for (ic=0; ic<Nall[X]; ic++)
    {
      for (jc=0; jc<Nall[Y]; jc++)
	{
	  for (kc=0; kc<Nall[Z]; kc++)
	    {
	      

	      index = get_linear_index(ic, jc, kc, Nall); 
	      site_map_status_temp[index] = site_map_get_status_index(index);

	    }
	}
    }


  /* copy data from CPU to accelerator */
  cudaMemcpy(site_map_status_d, site_map_status_temp, nsites*sizeof(char), \
	     cudaMemcpyHostToDevice);


  checkCUDAError("put_site_map_on_gpu");

}


colloid_t* colloid_list[MAX_COLLOIDS];
double colloid_r[MAX_COLLOIDS*3];

int build_colloid_list()
{

  int index, icolloid;
  colloid_t *p_c;
  int ncolloids=0;

  // build list of colloids, one entry for each, stored as memory addresses
  for (index=0;index<nsites;index++){
    
    p_c=colloid_at_site_index(index);  
    if(p_c){

      //printf("HHH %f\n", p_c->s.r[0]);
      int match=0;
      for (icolloid=0;icolloid<ncolloids;icolloid++){
	
	if(p_c==colloid_list[icolloid]){
	  match=1;
	  continue;
	}
	
      }
      if (match==0)
	{
	  colloid_list[ncolloids]=p_c;
	  ncolloids++;
	}
      
    }
    
  }

  return ncolloids;

}


/* copy colloid map from host to accelerator */
void put_colloid_map_on_gpu()
{
  
  int index;
  
  colloid_t *p_c;
  int icolloid;
  int ncolloids=build_colloid_list();

  for (index=0;index<nsites;index++){
    
    p_c=colloid_at_site_index(index);  
    if(p_c){
      
      //find out which colloid
      for (icolloid=0;icolloid<ncolloids;icolloid++){
	if(p_c==colloid_list[icolloid])	  break;
      }
      colloid_map_temp[index]=icolloid;
      
      //printf("%d %d\n", index,colloid_map_temp[index]);
    }

  }
  
  //  for (icolloid=0;icolloid<ncolloids;icolloid++)printf("colloid %d %d\n",icolloid,colloid_list[icolloid]);

  /* copy data from CPU to accelerator */
    cudaMemcpy(colloid_map_d, colloid_map_temp, nsites*sizeof(char),	\
  	     cudaMemcpyHostToDevice);


  checkCUDAError("put_colloid_map_on_gpu");

}

/* copy colloid map from host to accelerator */
void put_colloid_properties_on_gpu()
{
  
  int ia;
  colloid_t *p_c;
  int icolloid;
  int ncolloids=build_colloid_list();
      
   for (icolloid=0;icolloid<ncolloids;icolloid++){
    
     p_c=(colloid_t*) colloid_list[icolloid]; 
     
     //printf("NNN %f\n", p_c->s.r[0]);
     for (ia=0; ia<3; ia++)
       colloid_r[3*icolloid+ia]=p_c->s.r[ia]; 

   } 
  
  /* copy data from CPU to accelerator */
  cudaMemcpy(colloid_r_d, colloid_r, ncolloids*3*sizeof(double), \
  	     cudaMemcpyHostToDevice);


  checkCUDAError("put_colloid_map_on_gpu");

}





void zero_colloid_force_on_gpu()
{

  int zero=0;
  cudaMemset(colloid_force_d,zero,nsites*6*3*sizeof(double));
  checkCUDAError("zero_colloid_force_on_gpu");
}




extern double * ftmp;
void put_fluxes_on_gpu(){

  int nop=phi_nop();
  int index,n;

  //transpose
  for (index=0;index<nsites;index++){
    for (n=0;n<nop;n++){
      ftmp[n*nsites+index]=fluxe[nop*index+n];
	}
  }
  cudaMemcpy(fluxe_d, ftmp, nsites*nop*sizeof(double),
	    cudaMemcpyHostToDevice);



  for (index=0;index<nsites;index++){
    for (n=0;n<nop;n++){
      ftmp[n*nsites+index]=fluxw[nop*index+n];
	}
  }
  cudaMemcpy(fluxw_d, ftmp, nsites*nop*sizeof(double),
	    cudaMemcpyHostToDevice);


  for (index=0;index<nsites;index++){
    for (n=0;n<nop;n++){
      ftmp[n*nsites+index]=fluxy[nop*index+n];
	}
  }
  cudaMemcpy(fluxy_d, ftmp, nsites*nop*sizeof(double),
	    cudaMemcpyHostToDevice);



  for (index=0;index<nsites;index++){
    for (n=0;n<nop;n++){
      ftmp[n*nsites+index]=fluxz[nop*index+n];
	}
  }
  cudaMemcpy(fluxz_d, ftmp, nsites*nop*sizeof(double),
	    cudaMemcpyHostToDevice);


  /* cudaMemcpy(fluxw_d, fluxw, nsites*nop*sizeof(double), */
  /* 	    cudaMemcpyHostToDevice); */
  /* cudaMemcpy(fluxy_d, fluxy, nsites*nop*sizeof(double), */
  /* 	    cudaMemcpyHostToDevice); */
  /* cudaMemcpy(fluxz_d, fluxz, nsites*nop*sizeof(double), */
  /* 	    cudaMemcpyHostToDevice); */


}

/* void get_fluxes_from_gpu(){ */

/*   cudaMemcpy(fluxe, fluxe_d, nsites*nop*sizeof(double), */
/* 	    cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(fluxw, fluxw_d, nsites*nop*sizeof(double), */
/* 	    cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(fluxy, fluxy_d, nsites*nop*sizeof(double), */
/* 	    cudaMemcpyDeviceToHost); */
/*   cudaMemcpy(fluxz, fluxz_d, nsites*nop*sizeof(double), */
/* 	    cudaMemcpyDeviceToHost); */


/* } */



__global__ void printgpuint(int *array_d, int index){

  printf("GPU array [%d] = %d \n",index,array_d[index]);

}

__global__ void printgpudouble(double *array_d, int index){

  printf("GPU array [%d] = %e \n",index,array_d[index]);

}


/* get linear index from 3d coordinates (host) */
int get_linear_index(int ii,int jj,int kk,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}


/* check for CUDA errors */
void checkCUDAError(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) 
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
				cudaGetErrorString( err) );
		fflush(stdout);
		fflush(stderr);
		exit(EXIT_FAILURE);
	}                         
}

#ifdef KEVIN_GPU

/*****************************************************************************
 *
 *  There a whole load of stuff here related to colloids.
 *
 *  The aim would be to re-encapsulate the coords stuff in coords.c
 *  and the colloid stuff to e.g., colloids.c
 *
 *****************************************************************************/

typedef struct coords_s coords_t;

struct coords_s {
  int nhalo;
  int nlocal[3];
  int noffset[3];
};

__constant__ coords_t coord;

__host__   int coords_kernel_blocks_1d(int nextra, int * nblocks);
__device__ int coords_nkthreads_gpu(int nextra, int * nkthreads);
__device__ int coords_from_threadindex_gpu(int nextra, int threadindex,
                                           int * ic, int * jc, int * kc);
__device__ int coords_index_gpu(int ic, int jc, int kc, int * index);

static coll_array_t * carry;
coll_array_t * carry_d;

__global__ void colloid_update_map_gpu(int nextra, coll_array_t * carry_d,
                                       char * __restrict__ map);

/*****************************************************************************
 *
 *  utilities_init_extra
 *
 *****************************************************************************/

int utilities_init_extra(void) {

  coords_t host;

  /* coords */

  host.nhalo = coords_nhalo();
  coords_nlocal(host.nlocal);
  coords_nlocal_offset(host.noffset);

  cudaMemcpyToSymbol(coord, &host, sizeof(coords_t), 0,
                     cudaMemcpyHostToDevice);

  /* colloid */

  carry = (coll_array_t *) calloc(1, sizeof(coll_array_t));
  cudaMalloc((void **) &carry_d, sizeof(coll_array_t));

  return 0;
}

/*****************************************************************************
 *
 *  colloids_to_gpu
 *
 *****************************************************************************/

int colloids_to_gpu(void) {

  int ic, jc, kc;
  int idc, jdc, kdc;
  int ncnew = 0;
  int n = 0;
  int nblocks;
  int nextra;
  colloid_t * pc;
  colloid_state_t * tmp = NULL;

  /* Count up all local colloids; only then, add any 'true' halo
   * colloids (not periodic images) */

  idc = 1 - (cart_size(X) == 1);
  jdc = 1 - (cart_size(Y) == 1);
  kdc = 1 - (cart_size(Z) == 1);

  for (ic = 1 - idc; ic <= Ncell(X) + idc; ic++) {
    for (jc = 1 - jdc; jc <= Ncell(Y) + jdc; jc++) {
      for (kc = 1 - kdc; kc <= Ncell(Z) + kdc; kc++) {

         pc = colloids_cell_list(ic, jc, kc);
         for ( ; pc; pc = pc->next) ncnew += 1;
      }
    }
  }

  /* Push colloid states into the temporary array on host  */

  tmp = (colloid_state_t *) calloc(ncnew, sizeof(colloid_state_t));

  for (ic = 1 - idc; ic <= Ncell(X) + idc; ic++) {
    for (jc = 1 - jdc; jc <= Ncell(Y) + jdc; jc++) {
      for (kc = 1 - kdc; kc <= Ncell(Z) + kdc; kc++) {

        pc = colloids_cell_list(ic, jc, kc);
        for ( ; pc; pc = pc->next) tmp[n++] = pc->s;
      }
    }
  }

  /* Increase device memory as required */

  if (ncnew > carry->nc) {
    if (carry->nc > 0) cudaFree((void *) carry->s);
    cudaMalloc((void **) &carry->s, ncnew*sizeof(colloid_state_t));
  }

  /* Copy */

  carry->nc = ncnew;
  cudaMemcpy(carry->s, tmp, ncnew*sizeof(colloid_state_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(carry_d, carry, sizeof(coll_array_t), cudaMemcpyHostToDevice);

  free(tmp);

  /* Update map */

  nextra = coords_nhalo();
  coords_kernel_blocks_1d(nextra, &nblocks);
  colloid_update_map_gpu<<<nblocks, DEFAULT_TPB>>>
	(nextra, carry_d, colloid_map_d);

  cudaThreadSynchronize();
  checkCUDAError("COLLOIDS TO GPU");

  return 0;
}

/*****************************************************************************
 *
 *  coords_kernel_blocks_1d
 *
 *****************************************************************************/

__host__
int coords_kernel_blocks_1d(int nextra, int * nblocks) {

  int nlocal[3];
  int nh = 2*nextra;
  int npoints;

  coords_nlocal(nlocal);
  npoints = (nlocal[X] + nh)*(nlocal[Y] + nh)*(nlocal[Z] + nh);

  *nblocks = (npoints + DEFAULT_TPB - 1) / DEFAULT_TPB;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_update_map_gpu
 *
 *****************************************************************************/

__global__
void colloid_update_map_gpu(int nextra, coll_array_t * cary,
                            char * __restrict__ map) {

  int threadIndex;
  int ic, jc, kc, index;   /* Lattice position */
  int nkthreads;           /* Threads required for this kernel */
  int n;

  double a2;
  double rsq;
  double x, y, z;

  threadIndex = blockIdx.x*blockDim.x + threadIdx.x;
  coords_nkthreads_gpu(nextra, &nkthreads);

  if (threadIndex >= nkthreads) return;

  coords_from_threadindex_gpu(nextra, threadIndex, &ic, &jc, &kc);
  coords_index_gpu(ic, jc, kc, &index);

  /* The index is the position in the array; -1 is no colloid */

  map[index] = -1;

  for (n = 0; n < cary->nc; n++) {
    a2 = cary->s[n].a0*cary->s[n].a0;
    x = 1.0*(coord.noffset[X] + ic) - cary->s[n].r[X];
    y = 1.0*(coord.noffset[Y] + jc) - cary->s[n].r[Y];
    z = 1.0*(coord.noffset[Z] + kc) - cary->s[n].r[Z];
	/*
    if (threadIndex == 0) printf("Kernel: %d %f %f %f %f\n", n, cary->s[n].r[X],
	cary->s[n].r[Y], cary->s[n].r[Z], a2);
*/
    /* Minimum distance */
    if (x > 64.0) x -= 128.0; if (x < -64.0) x += 128.0;
    if (y > 64.0) y -= 128.0; if (y < -64.0) y += 128.0;
    if (z > 64.0) z -= 128.0; if (z < -64.0) z += 128.0;
    rsq = x*x + y*y + z*z;
    if (rsq < a2) map[index] = n;
  }

  return;
}

/*****************************************************************************
 *
 *  coords_nkthreads_gpu
 *
 *  Number of threads required to execute kernel data parallel across
 *  lattice sites a function of additional halo points required:
 *
 *****************************************************************************/

__device__
int coords_nkthreads_gpu(int nextra, int * nkthreads) {

  *nkthreads = (coord.nlocal[X] + 2*nextra)
             * (coord.nlocal[Y] + 2*nextra)
             * (coord.nlocal[Z] + 2*nextra);

  return 0;
}

/*****************************************************************************
 *
 *  coords_from_threadindex_gpu
 *
 *  Return (ic, jc, kc) as function of threadIndex 0...nkthreads
 *
 *  The values returned are offset so that the (ic, jc, kc) are
 *  the 'true' coordinates in the local domain, i.e., those
 *  required by a call to coords_index().
 *
 *  This depends on the width of the additional halo region at
 *  which computation takes place in the kernel 'nextra'.
 *
 *****************************************************************************/

__device__
int coords_from_threadindex_gpu(int nextra, int threadindex,
                                int * ic, int * jc, int * kc) {

  int zs = 1;
  int ys = zs*(coord.nlocal[Z] + 2*nextra);  /* y stride in kernel extent */
  int xs = ys*(coord.nlocal[Y] + 2*nextra);  /* x stride in kernel extent */

  *ic = threadindex / xs;
  *jc = (threadindex - xs*(*ic)) / ys;
  *kc = threadindex - xs*(*ic) - ys*(*jc);

  /* This leaves us with ic = 0 ... etc; 0 must map to 1 - nhalo */

  *ic += (1 - coord.nhalo);
  *jc += (1 - coord.nhalo);
  *kc += (1 - coord.nhalo);

  return 0;
}

/*****************************************************************************
 *
 *  coords_index_gpu
 *
 *  A direct analogy of coords_index(ic, jc, kc) with the ordinates
 *  in the 'true' local coordinate system.
 *
 *****************************************************************************/

__device__
int coords_index_gpu(int ic, int jc, int kc, int * index) {

  int zs = 1;
  int ys = zs*(coord.nlocal[Z] + 2*coord.nhalo);
  int xs = ys*(coord.nlocal[Y] + 2*coord.nhalo);

  *index = xs*(coord.nhalo + ic - 1)
         + ys*(coord.nhalo + jc - 1)
         + zs*(coord.nhalo + kc - 1);

  return 0;
}

#endif