/*****************************************************************************
 *
 * utilities_gpu.cu
 *  
 * Data management and other utilities for GPU adaptation of Ludwig
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


/* edge and  halo buffers on host */
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
static int nlexbuf;

/* Perform tasks necessary to initialise accelerator */
void init_dist_gpu()
{


  calculate_dist_data_sizes();
  allocate_dist_memory_on_gpu();

  //checkCUDAError("Init GPU");  


}

void finalise_dist_gpu()
{
  free_dist_memory_on_gpu();
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
  cudaMallocHost(&fedgeXLOW,nhalodataX*sizeof(double));
  cudaMallocHost(&fedgeXHIGH,nhalodataX*sizeof(double));

  fedgeXLOW = (double *) malloc(nhalodataX*sizeof(double));
  fedgeXHIGH = (double *) malloc(nhalodataX*sizeof(double));
  fedgeYLOW = (double *) malloc(nhalodataY*sizeof(double));
  fedgeYHIGH = (double *) malloc(nhalodataY*sizeof(double));
  fedgeZLOW = (double *) malloc(nhalodataZ*sizeof(double));
  fedgeZHIGH = (double *) malloc(nhalodataZ*sizeof(double));
  
  fhaloXLOW = (double *) malloc(nhalodataX*sizeof(double));
  fhaloXHIGH = (double *) malloc(nhalodataX*sizeof(double));
  fhaloYLOW = (double *) malloc(nhalodataY*sizeof(double));
  fhaloYHIGH = (double *) malloc(nhalodataY*sizeof(double));
  fhaloZLOW = (double *) malloc(nhalodataZ*sizeof(double));
  fhaloZHIGH = (double *) malloc(nhalodataZ*sizeof(double));
  
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

  //   checkCUDAError("allocate_memory_on_gpu");

}


/* Free memory on accelerator */
static void free_dist_memory_on_gpu()
{


  //free(fedgeXLOW);
  //free(fedgeXHIGH);
  cudaFreeHost(fedgeXLOW);
  cudaFreeHost(fedgeXHIGH);
  free(fedgeYLOW);
  free(fedgeYHIGH);
  free(fedgeZLOW);
  free(fedgeZHIGH);

  free(fhaloXLOW);
  free(fhaloXHIGH);
  free(fhaloYLOW);
  free(fhaloYHIGH);
  free(fhaloZLOW);
  free(fhaloZHIGH);

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


/* copy f_ edges from accelerator to host */
void get_f_edges_from_gpu()
{
  
  
  //checkCUDAError("get_f_edges_from_gpu");

}




/* copy f_ halos from host to accelerator */
void put_f_halos_on_gpu()
{

  static dim3 BlockDims;
  static dim3 GridDims;



  //checkCUDAError("get_f_edges_from_gpu");

}

void halo_swap_gpu()
{
  int NedgeX[3], NedgeY[3], NedgeZ[3];

  int ii,jj,kk,m,p,index_source,index_target;

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

  

  /* pack X edges on device */
 GridDims.x=(nhalo*N[Y]*N[Z]+BlockDims.x-1)/BlockDims.x;
 pack_edgesX_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,cv_d,
						N_d,fedgeXLOW_d,
						fedgeXHIGH_d,f_d);

  /* get X low edges */
 cudaMemcpy(fedgeXLOW, fedgeXLOW_d, nhalodataX*sizeof(double),
		 cudaMemcpyDeviceToHost);
 /* get X high edges */
 cudaMemcpy(fedgeXHIGH, fedgeXHIGH_d, nhalodataX*sizeof(double),
		 cudaMemcpyDeviceToHost);


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


  /* pack Y edges on device */ 
 GridDims.x=(Nall[X]*nhalo*N[Z]+BlockDims.x-1)/BlockDims.x;
 pack_edgesY_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,cv_d,
					       N_d,fedgeYLOW_d,
					       fedgeYHIGH_d,f_d);
  /* get Y low edges */
  cudaMemcpy(fedgeYLOW, fedgeYLOW_d, nhalodataY*sizeof(double), 
	     cudaMemcpyDeviceToHost);
  /* get Y high edges */
  cudaMemcpy(fedgeYHIGH, fedgeYHIGH_d, nhalodataY*sizeof(double), 
	     cudaMemcpyDeviceToHost);


 /* wait for X halo swaps to finish */ 
 if (cart_size(X) > 1)       MPI_Waitall(4, request, status);


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


 /* put X halos back on device, and unpack */
  cudaMemcpy(fhaloXLOW_d, fhaloXLOW, nhalodataX*sizeof(double), 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(fhaloXHIGH_d, fhaloXHIGH, nhalodataX*sizeof(double), 
	     cudaMemcpyHostToDevice);
  GridDims.x=(nhalo*N[Y]*N[Z]+BlockDims.x-1)/BlockDims.x;
  unpack_halosX_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,cv_d,
						  N_d,f_d,fhaloXLOW_d,
						  fhaloXHIGH_d);



 
 /* get Z data */
 /* pack Z edges on accelerator */ 
    GridDims.x=(Nall[X]*Nall[Y]*nhalo+BlockDims.x-1)/BlockDims.x;
    pack_edgesZ_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,cv_d,
  						N_d,fedgeZLOW_d,
  						fedgeZHIGH_d,f_d); 

  /* get Z low edges */
  cudaMemcpy(fedgeZLOW, fedgeZLOW_d, nhalodataZ*sizeof(double), 
		  cudaMemcpyDeviceToHost);
  /* get Z high edges */
  cudaMemcpy(fedgeZHIGH, fedgeZHIGH_d, nhalodataZ*sizeof(double), 
		  cudaMemcpyDeviceToHost);


  /* wait for Y halo swaps to finish */ 
  if (cart_size(Y) > 1)       MPI_Waitall(4, request, status); 



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


 /* put Y halos back on device, and unpack */
  cudaMemcpy(fhaloYLOW_d, fhaloYLOW, nhalodataY*sizeof(double), 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(fhaloYHIGH_d, fhaloYHIGH, nhalodataY*sizeof(double), 
	     cudaMemcpyHostToDevice);

  GridDims.x=(Nall[X]*nhalo*N[Z]+BlockDims.x-1)/BlockDims.x;
  unpack_halosY_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,cv_d,
						  N_d,f_d,fhaloYLOW_d,
						  fhaloYHIGH_d);

  /* wait for Z halo swaps to finish */ 
  if (cart_size(Z) > 1)       MPI_Waitall(4, request, status); 


 /* put Z halos back on device, and unpack */
  cudaMemcpy(fhaloZLOW_d, fhaloZLOW, nhalodataZ*sizeof(double), 
	     cudaMemcpyHostToDevice);
  cudaMemcpy(fhaloZHIGH_d, fhaloZHIGH, nhalodataZ*sizeof(double), 
	     cudaMemcpyHostToDevice);
  GridDims.x=(Nall[X]*Nall[Y]*nhalo+BlockDims.x-1)/BlockDims.x;
   unpack_halosZ_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,cv_d,
  						  N_d,f_d,fhaloZLOW_d,
  					  fhaloZHIGH_d);


  cudaThreadSynchronize();
  //TIMER_stop(HALOUNPACK);




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

