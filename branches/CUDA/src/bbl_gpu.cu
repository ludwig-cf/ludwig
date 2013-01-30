/*****************************************************************************
 *
 * bbl_gpu.cu
 * 
 * Alan Gray
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "bbl_gpu.h"
#include "utilities_gpu.h"
#include "util.h"
#include "model.h"
#include "timer.h"
#include "colloid_link.h"


/* external pointers to data on host*/
extern double * f_;
extern double * ftmp;
extern double * velocity_d;
extern double * phi_site_d;

/* external pointers to data on accelerator*/
extern int * cv_d;
extern int * N_d;
extern double * f_d;
extern double * ftmp_d;


/* buffers for bounce back on links */
int * findexall_d;
int * linktype_d;
double * dfall_d;
double * dgall1_d;
double * dgall2_d;
double * dmall_d;


/* data size variables */
static int ndata;
static int nhalo;
static int nsites;
static int ndist;
static int nop;
static  int N[3];
static  int Nall[3];
static int npvel; /* number of velocity components when packed */
static int nhalodataX;
static int nhalodataY;
static int nhalodataZ;
static int nlinkmax;


void bbl_init_temp_link_arrays_gpu(int nlink){

  nlinkmax=nlink;
  cudaMalloc((void **) &findexall_d, nlinkmax*sizeof(int));  
  cudaMalloc((void **) &linktype_d, nlinkmax*sizeof(int));  
  cudaMalloc((void **) &dfall_d, nlinkmax*sizeof(double));  
  cudaMalloc((void **) &dgall1_d, nlinkmax*sizeof(double));  
  cudaMalloc((void **) &dgall2_d, nlinkmax*sizeof(double));  
  cudaMalloc((void **) &dmall_d, nlinkmax*sizeof(double));  

}



void bbl_finalise_temp_link_arrays_gpu(){

  cudaFree(findexall_d);
  cudaFree(linktype_d);
  cudaFree(dfall_d);
  cudaFree(dgall1_d);
  cudaFree(dgall2_d);
  cudaFree(dmall_d);

}

void bbl_enlarge_temp_link_arrays_gpu(int nlink){

  bbl_finalise_temp_link_arrays_gpu();
  bbl_init_temp_link_arrays_gpu(nlink);

}





__global__ static void bounce_back_gpu_d(int *findexall_d, int *linktype_d,
					 double *dfall_d,
					 double *dgall1_d,
					 double *dgall2_d,
					 double *dmall_d,
					 double* f_d, double* phi_site_d,
					 int *N_d, 
					 int nhalo, int ndist,
					 int* cv_ptr, int nlink, int pass);


/* update distribution on accelerator for bounce back on links  */
/* host wrapper */
void bounce_back_gpu(int *findexall, int *linktype, double *dfall, 
		     double *dgall1,
		     double *dgall2,
		     double *dmall, int nlink, int pass){


  int nhalo = coords_nhalo();
  int ndist = distribution_ndist();
  coords_nlocal(N);

  /* allocate data on accelerator */
  checkCUDAError("bounce_back_gpu: malloc");
  
  /* copy data fom host to accelerator */
  cudaMemcpy(findexall_d, findexall, nlink*sizeof(int),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(linktype_d, linktype, nlink*sizeof(int),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dfall_d, dfall, nlink*sizeof(double), cudaMemcpyHostToDevice);
  if (ndist > 1 &&  pass==2){
    cudaMemcpy(dgall1_d, dgall1, nlink*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dgall2_d, dgall2, nlink*sizeof(double), cudaMemcpyHostToDevice);
  }

  checkCUDAError("bounce_back_gpu: memcpy to GPU");
  
  
  /* run the GPU kernel */
  int nblocks=(N[X]*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
 bounce_back_gpu_d<<<nblocks,DEFAULT_TPB>>>(findexall_d, linktype_d,
					       dfall_d, dgall1_d,dgall2_d,
					    dmall_d, f_d, phi_site_d, N_d,
					       nhalo, ndist, cv_d, nlink, pass);
 
 cudaThreadSynchronize();
 checkCUDAError("bounce_back_gpu");
 
 
 /* copy data fom accelerator to host */
 cudaMemcpy(dmall, dmall_d, nlink*sizeof(double),
	    cudaMemcpyDeviceToHost);
  if (ndist > 1 &&  pass==2){
    //dgall1 has been updated with multiplication by phi
    cudaMemcpy(dgall1, dgall1_d, nlink*sizeof(double),
	       cudaMemcpyDeviceToHost);
  }
 checkCUDAError("bounce_back_gpu: memcpy from GPU");
 
 /* free memory on accelerator */
 
}

/* update distribution on accelerator for bounce back on links */
/* GPU kernel */
__global__ static void bounce_back_gpu_d(int *findexall_d, int *linktype_d, 
					 double *dfall_d, double *dgall1_d,
					 double *dgall2_d,
					 double *dmall_d, double* f_d,
					 double *phi_site_d,
					 int *N_d, int nhalo, int ndist,
					 int *cv_ptr, int nlink, int pass){


  int threadIndex, nsite, Nall[3], ij, ji, ii, jj, kk, siteIndex;
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;
 
  Nall[X]=N_d[X]+2*nhalo;
  Nall[Y]=N_d[Y]+2*nhalo;
  Nall[Z]=N_d[Z]+2*nhalo;

  nsite = Nall[X]*Nall[Y]*Nall[Z];
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  if (threadIndex < nlink){
    
    int findex=findexall_d[threadIndex];

    /* velocity index outside->inside */
    ij = findex/nsite;
    /* velocity index inside->outside */
    ji = NVEL - ij;

    /* index for site (outside colloid) */
    siteIndex=findex-ij*nsite;

    /* 3D coords for site (outside colloid) */
    get_coords_from_index_gpu_d(&ii,&jj,&kk,siteIndex,Nall);

    /* 3D coords for neighbouring site (inside colloid) */
    int ii_=ii+cv_d[ij][0];
    int jj_=jj+cv_d[ij][1];
    int kk_=kk+cv_d[ij][2];

    /* index for neighbouring site (inside colloid) */
    int siteIndex_ = get_linear_index_gpu_d(ii_,jj_,kk_,Nall);

    /* get distribution for outside->inside link */
    double fdist = f_d[nsite*ndist*ij + siteIndex];

    if (linktype_d[threadIndex]==LINK_FLUID)
      {

	/* save a copy for updating colloid momentum */
	dmall_d[threadIndex] = fdist;
	
	/* update distribution */
	if (pass==1){
	  fdist = fdist + dfall_d[threadIndex];
	  f_d[nsite*ndist*ij+siteIndex]=fdist;
	}
	else if (pass==2){
	  fdist = fdist - dfall_d[threadIndex];
	  f_d[nsite*ndist*ji+siteIndex_]=fdist;

	  if (ndist>1){
	    
	    dgall1_d[threadIndex]=phi_site_d[siteIndex]*dgall1_d[threadIndex];
	    
	    fdist = f_d[nsite*ndist*ij + nsite + siteIndex];
	    fdist = fdist - dgall1_d[threadIndex] + dgall2_d[threadIndex];
	    f_d[nsite*ndist*ji + nsite + siteIndex_]=fdist;
	  }

	}
      }
    else
      {
	dmall_d[threadIndex] = fdist + f_d[nsite*ji+siteIndex_];
      }
      
    
  }
}


/* get 3d coordinates from the index on the accelerator */
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}

/* get linear index from 3d coordinates (device) */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}
