/*****************************************************************************
 *
 * dist_datamgmt_gpu.cu
 *  
 * Distribution data management for GPU adaptation of Ludwig
 * 
 * Alan Gray/ Alan Richardson 
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include "pe.h"
#include "utilities_gpu.h"
#include "util.h"
#include "model.h"
#include "timer.h"
#include "colloid_link.h"

/* external pointers to data on host*/
extern double * f_;

/* external pointers to data on accelerator*/
extern int * cv_d;
extern int * N_d;

/* accelerator memory address pointers for required data structures */
double * f_d;
double * ftmp_d;

/* edge and halo buffers on accelerator */
double * fedgeXLOW_d;
double * fedgeXHIGH_d;
double * fedgeYLOW_d;
double * fedgeYHIGH_d;
double * fedgeZLOW_d;
double * fedgeZHIGH_d;
double * fhaloXLOW_d;
double * fhaloXHIGH_d;
double * fhaloYLOW_d;
double * fhaloYHIGH_d;
double * fhaloZLOW_d;
double * fhaloZHIGH_d;


/* edge and halo buffers on host */
double * fedgeXLOW;
double * fedgeXHIGH;
double * fedgeYLOW;
double * fedgeYHIGH;
double * fedgeZLOW;
double * fedgeZHIGH;
double * fhaloXLOW;
double * fhaloXHIGH;
double * fhaloYLOW;
double * fhaloYHIGH;
double * fhaloZLOW;
double * fhaloZHIGH;

/* buffers for bounce back on links */
int * findexall_d;
int * linktype_d;
double * dfall_d;
double * dmall_d;


/* data size variables */
static int ndata;
static int nhalo;
static int nsites;
static int ndist;
static  int N[3];
static  int Nall[3];
static int npvel; /* number of velocity components when packed */
static int nhalodataX;
static int nhalodataY;
static int nhalodataZ;

int nlinkmax=11000;

/* handles for CUDA streams (for ovelapping)*/
static cudaStream_t streamX,streamY, streamZ;


/* Perform tasks necessary to initialise accelerator */
void init_dist_gpu()
{


  calculate_dist_data_sizes();
  allocate_dist_memory_on_gpu();
  
  /* create CUDA streams (for ovelapping)*/
  cudaStreamCreate(&streamX);
  cudaStreamCreate(&streamY);
  cudaStreamCreate(&streamZ);


  //checkCUDAError("Init GPU");  


}

void finalise_dist_gpu()
{
  free_dist_memory_on_gpu();

  /* destroy CUDA streams*/
  cudaStreamDestroy(streamX);
  cudaStreamDestroy(streamY);
  cudaStreamDestroy(streamZ);

}


/* calculate sizes of data - needed for memory copies to accelerator */
static void calculate_dist_data_sizes()
{
  coords_nlocal(N);  
  nhalo = coords_nhalo();  
  ndist = distribution_ndist();

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsites = Nall[X]*Nall[Y]*Nall[Z];
  ndata = nsites * ndist * NVEL;



  /* calculate number of velocity components when packed */
  int p;
  npvel=0;
  for (p=0; p<NVEL; p++)
    {
      if (cv[p][0] == 1) npvel++; 
    }

  nhalodataX = N[Y] * N[Z] * nhalo * ndist * npvel;
  nhalodataY = Nall[X] * N[Z] * nhalo * ndist * npvel;
  nhalodataZ = Nall[X] * Nall[Y] * nhalo * ndist * npvel;



}



/* Allocate memory on accelerator */
static void allocate_dist_memory_on_gpu()
{

  
  //fedgeXLOW = (double *) malloc(nhalodataX*sizeof(double));
  //fedgeXHIGH = (double *) malloc(nhalodataX*sizeof(double));
  //cudaMallocHost(&fedgeXLOW,nhalodataX*sizeof(double));
  //cudaMallocHost(&fedgeXHIGH,nhalodataX*sizeof(double));

  cudaHostAlloc( (void **)&fedgeXLOW, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fedgeXHIGH, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fedgeYLOW, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fedgeYHIGH, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fedgeZLOW, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fedgeZHIGH, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);


  cudaHostAlloc( (void **)&fhaloXLOW, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fhaloXHIGH, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fhaloYLOW, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fhaloYHIGH, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fhaloZLOW, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&fhaloZHIGH, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);

  //fedgeXLOW = (double *) malloc(nhalodataX*sizeof(double));
  //fedgeXHIGH = (double *) malloc(nhalodataX*sizeof(double));
  //fedgeYLOW = (double *) malloc(nhalodataY*sizeof(double));
  //fedgeYHIGH = (double *) malloc(nhalodataY*sizeof(double));
  //fedgeZLOW = (double *) malloc(nhalodataZ*sizeof(double));
  //fedgeZHIGH = (double *) malloc(nhalodataZ*sizeof(double));
  
  //fhaloXLOW = (double *) malloc(nhalodataX*sizeof(double));
  //fhaloXHIGH = (double *) malloc(nhalodataX*sizeof(double));
  //fhaloYLOW = (double *) malloc(nhalodataY*sizeof(double));
  //fhaloYHIGH = (double *) malloc(nhalodataY*sizeof(double));
  //fhaloZLOW = (double *) malloc(nhalodataZ*sizeof(double));
  //fhaloZHIGH = (double *) malloc(nhalodataZ*sizeof(double));
  
  /* arrays on accelerator */
  cudaMalloc((void **) &f_d, ndata*sizeof(double));
  cudaMalloc((void **) &ftmp_d, ndata*sizeof(double));
  
  cudaMalloc((void **) &fedgeXLOW_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &fedgeXHIGH_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &fedgeYLOW_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &fedgeYHIGH_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &fedgeZLOW_d, nhalodataZ*sizeof(double));
  cudaMalloc((void **) &fedgeZHIGH_d, nhalodataZ*sizeof(double));
  
  cudaMalloc((void **) &fhaloXLOW_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &fhaloXHIGH_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &fhaloYLOW_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &fhaloYHIGH_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &fhaloZLOW_d, nhalodataZ*sizeof(double));
  cudaMalloc((void **) &fhaloZHIGH_d, nhalodataZ*sizeof(double));

  cudaMalloc((void **) &findexall_d, nlinkmax*sizeof(int));  
  cudaMalloc((void **) &linktype_d, nlinkmax*sizeof(int));  
  cudaMalloc((void **) &dfall_d, nlinkmax*sizeof(double));  
  cudaMalloc((void **) &dmall_d, nlinkmax*sizeof(double));  

  


  //   checkCUDAError("allocate_memory_on_gpu");

}


/* Free memory on accelerator */
static void free_dist_memory_on_gpu()
{


  //free(fedgeXLOW);
  //free(fedgeXHIGH);
  cudaFreeHost(fedgeXLOW);
  cudaFreeHost(fedgeXHIGH);
  cudaFreeHost(fedgeYLOW);
  cudaFreeHost(fedgeYHIGH);
  cudaFreeHost(fedgeZLOW);
  cudaFreeHost(fedgeZHIGH);

  cudaFreeHost(fhaloXLOW);
  cudaFreeHost(fhaloXHIGH);
  cudaFreeHost(fhaloYLOW);
  cudaFreeHost(fhaloYHIGH);
  cudaFreeHost(fhaloZLOW);
  cudaFreeHost(fhaloZHIGH);

/*   free(fedgeYLOW); */
/*   free(fedgeYHIGH); */
/*   free(fedgeZLOW); */
/*   free(fedgeZHIGH); */

/*   free(fhaloXLOW); */
/*   free(fhaloXHIGH); */
/*   free(fhaloYLOW); */
/*   free(fhaloYHIGH); */
/*   free(fhaloZLOW); */
/*   free(fhaloZHIGH); */

  /* free memory on accelerator */
  cudaFree(f_d);
  cudaFree(ftmp_d);

  cudaFree(fedgeXLOW_d);
  cudaFree(fedgeXHIGH_d);
  cudaFree(fedgeYLOW_d);
  cudaFree(fedgeYHIGH_d);
  cudaFree(fedgeZLOW_d);
  cudaFree(fedgeZHIGH_d);

  cudaFree(fhaloXLOW_d);
  cudaFree(fhaloXHIGH_d);
  cudaFree(fhaloYLOW_d);
  cudaFree(fhaloYHIGH_d);
  cudaFree(fhaloZLOW_d);
  cudaFree(fhaloZHIGH_d);

 cudaFree(findexall_d);
 cudaFree(linktype_d);
 cudaFree(dfall_d);
 cudaFree(dmall_d);

}


/* copy f_ from host to accelerator */
void put_f_on_gpu()
{
  /* copy data from CPU to accelerator */
  cudaMemcpy(f_d, f_, ndata*sizeof(double), cudaMemcpyHostToDevice);

  //checkCUDAError("put_f_on_gpu");

}


/* copy f_ from accelerator back to host */
void get_f_from_gpu()
{

  /* copy data from accelerator to host */
  cudaMemcpy(f_, f_d, ndata*sizeof(double), cudaMemcpyDeviceToHost);
  //checkCUDAError("get_f_from_gpu");


}

/* copy f to ftmp on accelerator */
void copy_f_to_ftmp_on_gpu()
{
  /* copy data on accelerator */
  cudaMemcpy(ftmp_d, f_d, ndata*sizeof(double), cudaMemcpyDeviceToDevice);

  //checkCUDAError("cp_f_to_ftmp_on_gpu");

}

__global__ static void bounce_back_gpu_d(int *findexall_d, int *linktype_d,
					 double *dfall_d,
					 double *dmall_d,
					 double* f_d,
					 int *N_d, 
					 int nhalo, 
					 int* cv_ptr, int nlink, int pass);


/* update distribution on accelerator for bounce back on links  */
/* host wrapper */
void bounce_back_gpu(int *findexall, int *linktype, double *dfall,
		     double *dmall, int nlink, int pass){


  dim3 BlockDims;
  dim3 GridDims;
  int nhalo = coords_nhalo();
  
  /* allocate data on accelerator */
  //cudaMalloc((void **) &findexall_d, nlink*sizeof(int));  
  //cudaMalloc((void **) &linktype_d, nlink*sizeof(int));  
  //cudaMalloc((void **) &dfall_d, nlink*sizeof(double));  
  //cudaMalloc((void **) &dmall_d, nlink*sizeof(double));  
  checkCUDAError("bounce_back_gpu: malloc");
  
  /* copy data fom host to accelerator */
  cudaMemcpy(findexall_d, findexall, nlink*sizeof(int),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(linktype_d, linktype, nlink*sizeof(int),
	     cudaMemcpyHostToDevice);
  cudaMemcpy(dfall_d, dfall, nlink*sizeof(double), cudaMemcpyHostToDevice);
  checkCUDAError("bounce_back_gpu: memcpy to GPU");
  
  
  /* run the GPU kernel */
#define BLOCKSIZE 256
  BlockDims.x=BLOCKSIZE;
GridDims.x=(nlink+BlockDims.x-1)/BlockDims.x;
 bounce_back_gpu_d<<<GridDims.x,BlockDims.x>>>(findexall_d, linktype_d,
					       dfall_d,
					       dmall_d, f_d, N_d,
					       nhalo, cv_d, nlink, pass);
 
 cudaThreadSynchronize();
 checkCUDAError("bounce_back_gpu");
 
 
 /* copy data fom accelerator to host */
 cudaMemcpy(dmall, dmall_d, nlink*sizeof(double),
	    cudaMemcpyDeviceToHost);
 checkCUDAError("bounce_back_gpu: memcpy from GPU");
 
 /* free memory on accelerator */
 //cudaFree(findexall_d);
 //cudaFree(linktype_d);
 //cudaFree(dfall_d);
 //cudaFree(dmall_d);
 
}

/* update distribution on accelerator for bounce back on links */
/* GPU kernel */
__global__ static void bounce_back_gpu_d(int *findexall_d, int *linktype_d, 
					 double *dfall_d,
					 double *dmall_d, double* f_d, 
					 int *N_d, int nhalo,
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
    double fdist = f_d[nsite*ij + siteIndex];

    if (linktype_d[threadIndex]==LINK_FLUID)
      {

	/* save a copy for updating colloid momentum */
	dmall_d[threadIndex] = fdist;
	
	/* update distribution */
	if (pass==1){
	  fdist = fdist + dfall_d[threadIndex];
	  f_d[nsite*ij+siteIndex]=fdist;
	}
	else if (pass==2){
	  fdist = fdist - dfall_d[threadIndex];
	  f_d[nsite*ji+siteIndex_]=fdist;
	}
      }
    else
      {
	dmall_d[threadIndex] = fdist + f_d[nsite*ji+siteIndex_];
      }
      
    
  }
}


void halo_swap_gpu()
{
  int NedgeX[3], NedgeY[3], NedgeZ[3];

  int ii,jj,kk,m,p,index_source,index_target;

#define OVERLAP

  const int tagf = 900;
  const int tagb = 901;
  
  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm = cart_comm();




 static dim3 BlockDims;
 static dim3 GridDims;


  #define BLOCKSIZE 256
  /* 1D decomposition - use x grid and block dimension only */
  BlockDims.x=BLOCKSIZE;


  /* the sizes of the packed structures */
  NedgeX[X]=nhalo;
  NedgeX[Y]=N[Y];
  NedgeX[Z]=N[Z];

  NedgeY[X]=Nall[X];
  NedgeY[Y]=nhalo;
  NedgeY[Z]=N[Z];

  NedgeZ[X]=Nall[X];
  NedgeZ[Y]=Nall[Y];
  NedgeZ[Z]=nhalo;

  int npackedsiteX=NedgeX[X]*NedgeX[Y]*NedgeX[Z];
  int npackedsiteY=NedgeY[X]*NedgeY[Y]*NedgeY[Z];
  int npackedsiteZ=NedgeZ[X]*NedgeZ[Y]*NedgeZ[Z];

  /* the below code is structured to overlap packing, CPU-GPU comms and MPI 
   as and where possible */


 /* pack X edges on accelerator */
 GridDims.x=(nhalo*N[Y]*N[Z]+BlockDims.x-1)/BlockDims.x;
 pack_edgesX_gpu_d<<<GridDims.x,BlockDims.x,0,streamX>>>(ndist,nhalo,cv_d,
						N_d,fedgeXLOW_d,
						fedgeXHIGH_d,f_d);
 /* pack Y edges on accelerator */ 
 GridDims.x=(Nall[X]*nhalo*N[Z]+BlockDims.x-1)/BlockDims.x;
 pack_edgesY_gpu_d<<<GridDims.x,BlockDims.x,0,streamY>>>(ndist,nhalo,cv_d,
					       N_d,fedgeYLOW_d,
					       fedgeYHIGH_d,f_d);
 /* pack Z edges on accelerator */ 
 GridDims.x=(Nall[X]*Nall[Y]*nhalo+BlockDims.x-1)/BlockDims.x;
 pack_edgesZ_gpu_d<<<GridDims.x,BlockDims.x,0,streamZ>>>(ndist,nhalo,cv_d,
							 N_d,fedgeZLOW_d,
							 fedgeZHIGH_d,f_d); 

  /* get X low edges */
 cudaMemcpyAsync(fedgeXLOW, fedgeXLOW_d, nhalodataX*sizeof(double),
		 cudaMemcpyDeviceToHost,streamX);

 /* get X high edges */
 cudaMemcpyAsync(fedgeXHIGH, fedgeXHIGH_d, nhalodataX*sizeof(double),
		 cudaMemcpyDeviceToHost,streamX);

#ifndef OVERLAP
  TIMER_start(HALOGETX); 
  cudaStreamSynchronize(streamX);
  TIMER_stop(HALOGETX); 
#endif

 /* get Y low edges */
 cudaMemcpyAsync(fedgeYLOW, fedgeYLOW_d, nhalodataY*sizeof(double), 
		 cudaMemcpyDeviceToHost,streamY);
 /* get Y high edges */
 cudaMemcpyAsync(fedgeYHIGH, fedgeYHIGH_d, nhalodataY*sizeof(double), 
		 cudaMemcpyDeviceToHost,streamY);

#ifndef OVERLAP
  TIMER_start(HALOGETY); 
  cudaStreamSynchronize(streamY);
  TIMER_stop(HALOGETY); 
#endif

  /* get Z low edges */
  cudaMemcpyAsync(fedgeZLOW, fedgeZLOW_d, nhalodataZ*sizeof(double), 
	     cudaMemcpyDeviceToHost,streamZ);
  /* get Z high edges */
  cudaMemcpyAsync(fedgeZHIGH, fedgeZHIGH_d, nhalodataZ*sizeof(double), 
	     cudaMemcpyDeviceToHost,streamZ);

#ifndef OVERLAP
  TIMER_start(HALOGETZ); 
  cudaStreamSynchronize(streamZ);
  TIMER_stop(HALOGETZ); 
#endif

  TIMER_start(HALOGETX);
 /* wait for X data from accelerator*/ 
  cudaStreamSynchronize(streamX); 
  TIMER_stop(HALOGETX);


  /* The x-direction (YZ plane) */
   if (cart_size(X) == 1) {
    /* x up */
    memcpy(fhaloXLOW,fedgeXHIGH,nhalodataX*sizeof(double));
    
    /* x down */
    memcpy(fhaloXHIGH,fedgeXLOW,nhalodataX*sizeof(double));
      }
  else
    {
      /* initiate transfers */
      MPI_Irecv(fhaloXLOW, nhalodataX, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), tagf, comm, &request[0]);
      MPI_Irecv(fhaloXHIGH, nhalodataX, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), tagb, comm, &request[1]);
      MPI_Isend(fedgeXHIGH, nhalodataX, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), tagf, comm, &request[2]);
      MPI_Isend(fedgeXLOW, nhalodataX, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), tagb, comm, &request[3]);

    }


  TIMER_start(HALOMPIX);
 /* wait for X halo swaps to finish */ 
   if (cart_size(X) > 1)       MPI_Waitall(4, request, status);
  TIMER_stop(HALOMPIX);


 /* put X halos back on device, and unpack */
  cudaMemcpyAsync(fhaloXLOW_d, fhaloXLOW, nhalodataX*sizeof(double), 
	     cudaMemcpyHostToDevice,streamX);
  cudaMemcpyAsync(fhaloXHIGH_d, fhaloXHIGH, nhalodataX*sizeof(double), 
	     cudaMemcpyHostToDevice,streamX);
  GridDims.x=(nhalo*N[Y]*N[Z]+BlockDims.x-1)/BlockDims.x;
  unpack_halosX_gpu_d<<<GridDims.x,BlockDims.x,0,streamX>>>(ndist,nhalo,cv_d,
						  N_d,f_d,fhaloXLOW_d,
						  fhaloXHIGH_d);

#ifndef OVERLAP
  TIMER_start(HALOPUTX); 
  cudaStreamSynchronize(streamX);
  TIMER_stop(HALOPUTX); 
#endif

  TIMER_start(HALOGETY);
  /* wait for Y data from accelerator*/ 
  cudaStreamSynchronize(streamY); 
  TIMER_stop(HALOGETY);

  TIMER_start(HALOYCORNER);
  /* fill in corners of Y edge data  */

  for (p=0;p<npvel;p++)
    {
      for (m=0;m<ndist;m++)
	{
	  
	  
	  for (ii = 0; ii < nhalo; ii++) {
	    for (jj = 0; jj < nhalo; jj++) {
	      for (kk = 0; kk < N[Z]; kk++) {
		

		
		/* xlow part of ylow */
		index_source = get_linear_index(ii,jj,kk,NedgeX);
		index_target = get_linear_index(ii,jj,kk,NedgeY);
		
		fedgeYLOW[npackedsiteY*ndist*p+npackedsiteY*m+index_target] =
		fhaloXLOW[npackedsiteX*ndist*p+npackedsiteX*m+index_source];

		/* xlow part of yhigh */
		index_source = get_linear_index(ii,NedgeX[Y]-1-jj,kk,NedgeX);
		index_target = get_linear_index(ii,jj,kk,NedgeY);

		fedgeYHIGH[npackedsiteY*ndist*p+npackedsiteY*m+index_target] =
		fhaloXLOW[npackedsiteX*ndist*p+npackedsiteX*m+index_source];


		/* get high X data */

		/* xhigh part of ylow */
		index_source = get_linear_index(ii,jj,kk,NedgeX);
		index_target = get_linear_index(NedgeY[X]-1-ii,jj,kk,NedgeY);

		fedgeYLOW[npackedsiteY*ndist*p+npackedsiteY*m+index_target] =
		fhaloXHIGH[npackedsiteX*ndist*p+npackedsiteX*m+index_source];

		/* xhigh part of yhigh */
		index_source = get_linear_index(ii,NedgeX[Y]-1-jj,kk,NedgeX);			index_target = get_linear_index(NedgeY[X]-1-ii,jj,kk,NedgeY);

		fedgeYHIGH[npackedsiteY*ndist*p+npackedsiteY*m+index_target] =
		fhaloXHIGH[npackedsiteX*ndist*p+npackedsiteX*m+index_source];



	      }
	    }
	    
	  }
	}
    }
  TIMER_stop(HALOYCORNER);  

  /* The y-direction (XZ plane) */

   if (cart_size(Y) == 1) {
  /* y up */
     memcpy(fhaloYLOW,fedgeYHIGH,nhalodataY*sizeof(double));

  /* y down */
     memcpy(fhaloYHIGH,fedgeYLOW,nhalodataY*sizeof(double));
      }
  else
    {

      /* initiate transfers */
      MPI_Irecv(fhaloYLOW, nhalodataY, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), tagf, comm, &request[0]);
      MPI_Irecv(fhaloYHIGH, nhalodataY, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), tagb, comm, &request[1]);
      MPI_Isend(fedgeYHIGH, nhalodataY, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), tagf, comm, &request[2]);
      MPI_Isend(fedgeYLOW, nhalodataY, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), tagb, comm, &request[3]);

    }

  TIMER_start(HALOMPIY);
 /* wait for Y halo swaps to finish */ 
    if (cart_size(Y) > 1)       MPI_Waitall(4, request, status); 
  TIMER_stop(HALOMPIY);

 /* put Y halos back on device, and unpack */
  cudaMemcpyAsync(fhaloYLOW_d, fhaloYLOW, nhalodataY*sizeof(double), 
		  cudaMemcpyHostToDevice,streamY);
  cudaMemcpyAsync(fhaloYHIGH_d, fhaloYHIGH, nhalodataY*sizeof(double), 
	     cudaMemcpyHostToDevice,streamY);

  GridDims.x=(Nall[X]*nhalo*N[Z]+BlockDims.x-1)/BlockDims.x;
  unpack_halosY_gpu_d<<<GridDims.x,BlockDims.x,0,streamY>>>(ndist,nhalo,cv_d,
						  N_d,f_d,fhaloYLOW_d,
						  fhaloYHIGH_d);

#ifndef OVERLAP
  TIMER_start(HALOPUTY); 
  cudaStreamSynchronize(streamY);
  TIMER_stop(HALOPUTY); 
#endif
 
  TIMER_start(HALOGETZ);
  /* wait for Z data from accelerator*/ 
  cudaStreamSynchronize(streamZ); 
  TIMER_stop(HALOGETZ);

  TIMER_start(HALOZCORNER);
  /* fill in corners of Z edge data: from Xhalo  */
  for (p=0;p<npvel;p++)
    {
      for (m=0;m<ndist;m++)
	{
	  
	  for (ii = 0; ii < nhalo; ii++) {
	    for (jj = 0; jj < N[Y]; jj++) {
	      for (kk = 0; kk < nhalo; kk++) {
		


		/* xlow part of zlow */
		index_source = get_linear_index(ii,jj,kk,NedgeX);
		index_target = get_linear_index(ii,jj+nhalo,kk,NedgeZ);

		fedgeZLOW[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloXLOW[npackedsiteX*ndist*p+npackedsiteX*m+index_source];


		/* xlow part of zhigh */
		index_source = get_linear_index(ii,jj,NedgeX[Z]-1-kk,NedgeX);
		index_target = get_linear_index(ii,jj+nhalo,kk,NedgeZ);

		fedgeZHIGH[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloXLOW[npackedsiteX*ndist*p+npackedsiteX*m+index_source];



		/* xhigh part of zlow */
		index_source = get_linear_index(ii,jj,kk,NedgeX);
		index_target = get_linear_index(NedgeZ[X]-1-ii,jj+nhalo,kk,
						NedgeZ);

		fedgeZLOW[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloXHIGH[npackedsiteX*ndist*p+npackedsiteX*m+index_source];


		/* xhigh part of zhigh */

		index_source = get_linear_index(ii,jj,NedgeX[Z]-1-kk,NedgeX);
		index_target = get_linear_index(NedgeZ[X]-1-ii,jj+nhalo,kk,
						NedgeZ);

		fedgeZHIGH[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloXHIGH[npackedsiteX*ndist*p+npackedsiteX*m+index_source];
		
		
	      }
	    }
	    
	    
	  }
	}
    }


  /* fill in corners of Z edge data: from Yhalo  */
  for (p=0;p<npvel;p++)
    {
      for (m=0;m<ndist;m++)
	{
	  
	  
	  
	  for (ii = 0; ii < Nall[X]; ii++) {
	    for (jj = 0; jj < nhalo; jj++) {
	      for (kk = 0; kk < nhalo; kk++) {
		
		
		
		/* ylow part of zlow */
		index_source = get_linear_index(ii,jj,kk,NedgeY);
		index_target = get_linear_index(ii,jj,kk,NedgeZ);

		fedgeZLOW[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloYLOW[npackedsiteY*ndist*p+npackedsiteY*m+index_source];


		/* ylow part of zhigh */
		index_source = get_linear_index(ii,jj,NedgeY[Z]-1-kk,NedgeY);
		index_target = get_linear_index(ii,jj,kk,NedgeZ);

		fedgeZHIGH[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloYLOW[npackedsiteY*ndist*p+npackedsiteY*m+index_source];



		/* yhigh part of zlow */
		index_source = get_linear_index(ii,jj,kk,NedgeY);
		index_target = get_linear_index(ii,NedgeZ[Y]-1-jj,kk,NedgeZ);

		fedgeZLOW[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloYHIGH[npackedsiteY*ndist*p+npackedsiteY*m+index_source];


		/* yhigh part of zhigh */

		index_source = get_linear_index(ii,jj,NedgeY[Z]-1-kk,NedgeY);
		index_target = get_linear_index(ii,NedgeZ[Y]-1-jj,kk,NedgeZ);

		fedgeZHIGH[npackedsiteZ*ndist*p+npackedsiteZ*m+index_target] =
		  fhaloYHIGH[npackedsiteY*ndist*p+npackedsiteY*m+index_source];


	      }
	    }
	    
	  }
	}
    }
  TIMER_stop(HALOZCORNER);

  TIMER_start(HALOMPIZ); 
  if (cart_size(Z) == 1) {
  /* The z-direction (xy plane) */
  /* z up */
  memcpy(fhaloZLOW,fedgeZHIGH,nhalodataZ*sizeof(double));

  /* z down */
  memcpy(fhaloZHIGH,fedgeZLOW,nhalodataZ*sizeof(double));
      }
  else
    {
      MPI_Irecv(fhaloZLOW, nhalodataZ, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), tagf, comm, &request[0]);
      MPI_Irecv(fhaloZHIGH, nhalodataZ, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), tagb, comm, &request[1]);
      MPI_Isend(fedgeZHIGH, nhalodataZ, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), tagf, comm, &request[2]);
      MPI_Isend(fedgeZLOW, nhalodataZ, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), tagb, comm, &request[3]);

    }


  /* wait for Z halo swaps to finish */ 
  if (cart_size(Z) > 1)       MPI_Waitall(4, request, status); 
  TIMER_stop(HALOMPIZ); 

 /* put Z halos back on device, and unpack */
  cudaMemcpyAsync(fhaloZLOW_d, fhaloZLOW, nhalodataZ*sizeof(double), 
	     cudaMemcpyHostToDevice,streamZ);
  cudaMemcpyAsync(fhaloZHIGH_d, fhaloZHIGH, nhalodataZ*sizeof(double), 
	     cudaMemcpyHostToDevice,streamZ);
  GridDims.x=(Nall[X]*Nall[Y]*nhalo+BlockDims.x-1)/BlockDims.x;
  unpack_halosZ_gpu_d<<<GridDims.x,BlockDims.x,0,streamZ>>>(ndist,nhalo,cv_d,
  						  N_d,f_d,fhaloZLOW_d,
  					  fhaloZHIGH_d);


  /* wait for all streams to complete */
  TIMER_start(HALOPUTX); 
  cudaStreamSynchronize(streamX);
  TIMER_stop(HALOPUTX); 

  TIMER_start(HALOPUTY); 
  cudaStreamSynchronize(streamY);
  TIMER_stop(HALOPUTY); 

  TIMER_start(HALOPUTZ); 
  cudaStreamSynchronize(streamZ);
  TIMER_stop(HALOPUTZ); 

  //cudaThreadSynchronize();

}



/* pack X edges on the accelerator */
__global__ static void pack_edgesX_gpu_d(int ndist, int nhalo,
					 int* cv_ptr, int N[3],
					 double* fedgeXLOW_d,
					 double* fedgeXHIGH_d, double* f_d)
{

  int p,m, index,ii,jj,kk;
  int packed_index,packedp;
 /* cast dummy cv pointer to correct 2d type */
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;


  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];
 
  int Nedge[3];
  Nedge[X]=nhalo;
  Nedge[Y]=N[Y];
  Nedge[Z]=N[Z];

  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);
      
      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii+nhalo,jj+nhalo,kk+nhalo,Nall);

      /* variables to determine how vel packing is done from cv array */
      int dirn=X; /* 3d direction */ 
      int ud=-1; /* up or down */
      int pn=-1; /* positive 1 or negative 1 factor */

      
      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud )
	  {
	    for (m = 0; m < ndist; m++) {
	      fedgeXLOW_d[ndist*npackedsite*packedp+m*npackedsite
			  +packed_index]
		= f_d[ndist*nsite*p+nsite*m+index];
	    }
	    packedp++;
	  }
      }
      
  
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(Nall[X]-nhalo-1-ii,jj+nhalo,kk+nhalo,Nall);
      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud*pn )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      fedgeXHIGH_d[ndist*npackedsite*packedp+m*npackedsite
			   +packed_index]
		= f_d[ndist*nsite*p+nsite*m+index];
	      
	    }
	    packedp++;
	  }
      }
    }
  
  
}

/* unpack X halos on the accelerator */
__global__ static void unpack_halosX_gpu_d(int ndist, int nhalo,
					   int* cv_ptr,int N[3],
					   double* f_d, double* fhaloXLOW_d,
					   double* fhaloXHIGH_d)
{



  int p,m, index,ii,jj,kk;
  int packed_index, packedp;
 /* cast dummy cv pointer to correct 2d type */
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];

  int Nedge[3];
  Nedge[X]=nhalo;
  Nedge[Y]=N[Y];
  Nedge[Z]=N[Z];

  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);

      /* variables to determine how vel packing is done from cv array */
      int dirn=X; /* 3d direction */ 
      int ud=1; /* up or down */
      int pn=-1; /* positive 1 or negative 1 factor */
      
      /* LOW HALO */
      index = get_linear_index_gpu_d(ii,jj+nhalo,kk+nhalo,Nall);
      
      /* copy packed structure data to original array */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud)
	  {
	    for (m = 0; m < ndist; m++) {
	  
	      f_d[ndist*nsite*p+nsite*m+index] =
		fhaloXLOW_d[ndist*npackedsite*packedp+m*npackedsite
			    +packed_index];

	    }
	    packedp++;
	  }
      }
           
  
      /* HIGH HALO */
      index = get_linear_index_gpu_d(Nall[X]-1-ii,jj+nhalo,kk+nhalo,Nall);
      /* copy packed structure data to original array */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud*pn )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      f_d[ndist*nsite*p+nsite*m+index] =
		fhaloXHIGH_d[ndist*npackedsite*packedp+m*npackedsite
			     +packed_index];
	      
	    }
	    packedp++;
	  }
      }
    }
  
  
}


/* pack Y edges on the accelerator */
__global__ static void pack_edgesY_gpu_d(int ndist, int nhalo,
					 int* cv_ptr, int N[3], 					 double* fedgeYLOW_d,
					 double* fedgeYHIGH_d, double* f_d) {

  int p,m, index,ii,jj,kk;
  int packed_index, packedp;

 /* cast dummy cv pointer to correct 2d type */
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];

  int Nedge[3];
  Nedge[X]=Nall[X];
  Nedge[Y]=nhalo;
  Nedge[Z]=N[Z];

  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);

      /* variables to determine how vel packing is done from cv array */
      int dirn=Y; /* 3d direction */ 
      int ud=-1; /* up or down */
      int pn=-1; /* positive 1 or negative 1 factor */
  
    
      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj+nhalo,kk+nhalo,Nall);
      
      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      fedgeYLOW_d[ndist*npackedsite*packedp+m*npackedsite+packed_index]
		= f_d[ndist*nsite*p+nsite*m+index];
	      
	    }
	    packedp++;
	  }
      }
      
      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,Nall[Y]-nhalo-1-jj,kk+nhalo,Nall);
      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud*pn )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      fedgeYHIGH_d[ndist*npackedsite*packedp+m*npackedsite+packed_index]
		= f_d[ndist*nsite*p+nsite*m+index];
	      
	    }
	    packedp++;
	  }
      }
    }
  
  
}




/* unpack Y halos on the accelerator */
__global__ static void unpack_halosY_gpu_d(int ndist, int nhalo,
					 int* cv_ptr, int N[3],
					   double* f_d, double* fhaloYLOW_d,
					   double* fhaloYHIGH_d)
{

  int p,m, index,ii,jj,kk;
  int packed_index, packedp;
 /* cast dummy cv pointer to correct 2d type */
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;
  

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];

  int Nedge[3];
  Nedge[X]=Nall[X];
  Nedge[Y]=nhalo;
  Nedge[Z]=N[Z];

  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);

      /* variables to determine how vel packing is done from cv array */
      int dirn=Y; /* 3d direction */ 
      int ud=1; /* up or down */
      int pn=-1; /* positive 1 or negative 1 factor */

      /* correct for diagonal data that was packed by X packing subroutine */
      if (ii < nhalo) 
	{ 
	  dirn = X;
	  ud=1;
	  pn=1;
	}
      if (ii >= Nall[X]-nhalo)
	{ 
	  dirn = X;
	  ud=-1;
	  pn=1;
	}



      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj,kk+nhalo,Nall);
      
       /* copy packed structure data to original array */
      packedp=0;

      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      f_d[ndist*nsite*p+nsite*m+index] =
	      fhaloYLOW_d[ndist*npackedsite*packedp+m*npackedsite+packed_index];
	      
	    }
	    packedp++;
	  }
  
      
      }

      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,Nall[Y]-1-jj,kk+nhalo,Nall);
      /* copy packed structure data to original array */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud*pn )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      f_d[ndist*nsite*p+nsite*m+index] =
		fhaloYHIGH_d[ndist*npackedsite*packedp+m*npackedsite+packed_index];
	      
	    }
	    packedp++;
	  }
	
      }
      
    }
  
  
  
}



/* pack Z edges on the accelerator */
__global__ static void pack_edgesZ_gpu_d(int ndist, int nhalo,
					 int* cv_ptr, int N[3],
					 double* fedgeZLOW_d,
					 double* fedgeZHIGH_d, double* f_d)
{

  int p,m, index,ii,jj,kk;
  int packed_index, packedp;
 /* cast dummy cv pointer to correct 2d type */
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];

  int Nedge[3];
  Nedge[X]=Nall[X];
  Nedge[Y]=Nall[Y];
  Nedge[Z]=nhalo;

  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);

      /* variables to determine how vel packing is done from cv array */
      int dirn=Z; /* 3d direction */ 
      int ud=-1; /* up or down */
      int pn=-1; /* positive 1 or negative 1 factor */

      
      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj,kk+nhalo,Nall);
      
      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      fedgeZLOW_d[ndist*npackedsite*packedp+m*npackedsite+packed_index]
		= f_d[ndist*nsite*p+nsite*m+index];
	      
	    }
	    packedp++;
	  }
      }
      
      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,jj,Nall[Z]-nhalo-1-kk,Nall);
      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud*pn )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      fedgeZHIGH_d[ndist*npackedsite*packedp+m*npackedsite+packed_index]
		= f_d[ndist*nsite*p+nsite*m+index];
	      
	    }
	    packedp++;
	  }
      }
    }
  
  
}




/* unpack Z halos on the accelerator */
__global__ static void unpack_halosZ_gpu_d(int ndist, int nhalo,
					   int* cv_ptr, int N[3],
					   double* f_d, double* fhaloZLOW_d,
					   double* fhaloZHIGH_d)
{

  int p,m, index,ii,jj,kk;
  int packed_index, packedp;
 /* cast dummy cv pointer to correct 2d type */
  int (*cv_d)[3] = (int (*)[3]) cv_ptr;
  
  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];

  int Nedge[3];
  Nedge[X]=Nall[X];
  Nedge[Y]=Nall[Y];
  Nedge[Z]=nhalo;
  
  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);

      /* variables to determine how vel packing is done from cv array */
      int dirn=Z; /* 3d direction */ 
      int ud=1; /* up or down */
      int pn=-1; /* positive 1 or negative 1 factor */

      /* correct for diagonal data that was packed by X packing subroutine */
      if (ii < nhalo)
	{ 
	  dirn = X;
	  ud=1;
	  pn=1;
	}
      if (ii >= Nall[X]-nhalo)
	{ 
	  dirn = X;
	  ud=-1;
	  pn=1;
	}
      /* correct for diagonal data that was packed by Y packing subroutine */
      if (jj < nhalo)
	{ 
	  dirn = Y;
	  ud=1;
	  pn=1;
	}

      if (jj >= Nall[Y]-nhalo)
	{ 
	  dirn = Y;
	  ud=-1;
	  pn=1;
	}

      
      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj,kk,Nall);
      
      /* copy packed structure data to original array */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      f_d[ndist*nsite*p+nsite*m+index] =
		fhaloZLOW_d[ndist*npackedsite*packedp+m*npackedsite+packed_index];
	      
	    }
	    packedp++;
	  }
      }
      
      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,jj,Nall[Z]-1-kk,Nall);
      /* copy packed structure data to original array */
      packedp=0;
      for (p = 0; p < NVEL; p++) {
	if (cv_d[p][dirn] == ud*pn )
	  {
	    for (m = 0; m < ndist; m++) {
	      
	      f_d[ndist*nsite*p+nsite*m+index] =
		fhaloZHIGH_d[ndist*npackedsite*packedp+m*npackedsite+packed_index];
	      
	    }
	    packedp++;
	  }
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

