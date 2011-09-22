/*****************************************************************************
 *
 * phi_datamgmt_gpu.cu
 *  
 * Phi data management for GPU adaptation of Ludwig
 * Alan Gray 
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


/* edge and halo buffers on accelerator */

double * phiedgeXLOW_d;
double * phiedgeXHIGH_d;
double * phiedgeYLOW_d;
double * phiedgeYHIGH_d;
double * phiedgeZLOW_d;
double * phiedgeZHIGH_d;
double * phihaloXLOW_d;
double * phihaloXHIGH_d;
double * phihaloYLOW_d;
double * phihaloYHIGH_d;
double * phihaloZLOW_d;
double * phihaloZHIGH_d;


/* host memory address pointers for temporary staging of data */
double * phi_site_temp;
double * grad_phi_site_temp;
double * delsq_phi_site_temp;


/* edge and  halo buffers on host */

double * phiedgeXLOW;
double * phiedgeXHIGH;
double * phiedgeYLOW;
double * phiedgeYHIGH;
double * phiedgeZLOW;
double * phiedgeZHIGH;
double * phihaloXLOW;
double * phihaloXHIGH;
double * phihaloYLOW;
double * phihaloYHIGH;
double * phihaloZLOW;
double * phihaloZHIGH;


/* pointers to data resident on accelerator */

extern int * N_d;
double * phi_site_d;
double * grad_phi_site_d;
double * delsq_phi_site_d;
double * le_index_real_to_buffer_d;

int * le_index_real_to_buffer_temp;

/* data size variables */
static int nhalo;
static int nsites;
static int nop;
static  int N[3];
static  int Nall[3];
static int nphihalodataX;
static int nphihalodataY;
static int nphihalodataZ;
static int nlexbuf;

/* handles for CUDA streams (for ovelapping)*/
static cudaStream_t streamX,streamY, streamZ;


void init_phi_gpu(){

  int ic;

  calculate_phi_data_sizes();
  allocate_phi_memory_on_gpu();
  

  /* get le_index_to_real_buffer values */
/*le_index_real_to_buffer holds -1 then +1 translation values */
  for (ic=0; ic<Nall[X]; ic++)
    {

      le_index_real_to_buffer_temp[ic]=
	le_index_real_to_buffer(ic,-1);

      le_index_real_to_buffer_temp[Nall[X]+ic]=
	le_index_real_to_buffer(ic,+1);

    }

  cudaMemcpy(le_index_real_to_buffer_d, le_index_real_to_buffer_temp, 
	     nlexbuf*sizeof(int),cudaMemcpyHostToDevice);

  /* create CUDA streams (for ovelapping)*/
  cudaStreamCreate(&streamX);
  cudaStreamCreate(&streamY);
  cudaStreamCreate(&streamZ);


}


void finalise_phi_gpu()
{
  free_phi_memory_on_gpu();

  /* destroy CUDA streams*/
  cudaStreamDestroy(streamX);
  cudaStreamDestroy(streamY);
  cudaStreamDestroy(streamZ);

}

/* calculate sizes of data - needed for memory copies to accelerator */
static void calculate_phi_data_sizes()
{
  coords_nlocal(N);  
  nhalo = coords_nhalo();  

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsites = Nall[X]*Nall[Y]*Nall[Z];

  nop = phi_nop();

  nphihalodataX = N[Y] * N[Z] * nhalo * nop ;
  nphihalodataY = Nall[X] * N[Z] * nhalo * nop;
  nphihalodataZ = Nall[X] * Nall[Y] * nhalo * nop;



  //nlexbuf = le_get_nxbuffer();
/*for holding le buffer index translation */
/* -1 then +1 values */
  nlexbuf = 2*Nall[X]; 



}





/* Allocate memory on accelerator */
static void allocate_phi_memory_on_gpu()
{

  /* temp arrays for staging data on  host */
  phi_site_temp = (double *) malloc(nsites*sizeof(double));
  grad_phi_site_temp = (double *) malloc(nsites*3*sizeof(double));
  delsq_phi_site_temp = (double *) malloc(nsites*sizeof(double));
  le_index_real_to_buffer_temp = (int *) malloc(nlexbuf*sizeof(int));

  cudaHostAlloc( (void **)&phiedgeXLOW, nphihalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phiedgeXHIGH, nphihalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phiedgeYLOW, nphihalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phiedgeYHIGH, nphihalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phiedgeZLOW, nphihalodataZ*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phiedgeZHIGH, nphihalodataZ*sizeof(double), 
		 cudaHostAllocDefault);


  cudaHostAlloc( (void **)&phihaloXLOW, nphihalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phihaloXHIGH, nphihalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phihaloYLOW, nphihalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phihaloYHIGH, nphihalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phihaloZLOW, nphihalodataZ*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&phihaloZHIGH, nphihalodataZ*sizeof(double), 
		 cudaHostAllocDefault);
  

//  phiedgeXLOW = (double *) malloc(nphihalodataX*sizeof(double));
//  phiedgeXHIGH = (double *) malloc(nphihalodataX*sizeof(double));
//  phiedgeYLOW = (double *) malloc(nphihalodataY*sizeof(double));
//  phiedgeYHIGH = (double *) malloc(nphihalodataY*sizeof(double));
//  phiedgeZLOW = (double *) malloc(nphihalodataZ*sizeof(double));
//  phiedgeZHIGH = (double *) malloc(nphihalodataZ*sizeof(double));
  
//  phihaloXLOW = (double *) malloc(nphihalodataX*sizeof(double));
//  phihaloXHIGH = (double *) malloc(nphihalodataX*sizeof(double));
//  phihaloYLOW = (double *) malloc(nphihalodataY*sizeof(double));
//  phihaloYHIGH = (double *) malloc(nphihalodataY*sizeof(double));
//  phihaloZLOW = (double *) malloc(nphihalodataZ*sizeof(double));
//  phihaloZHIGH = (double *) malloc(nphihalodataZ*sizeof(double));

  cudaMalloc((void **) &phiedgeXLOW_d, nphihalodataX*sizeof(double));
  cudaMalloc((void **) &phiedgeXHIGH_d, nphihalodataX*sizeof(double));
  cudaMalloc((void **) &phiedgeYLOW_d, nphihalodataY*sizeof(double));
  cudaMalloc((void **) &phiedgeYHIGH_d, nphihalodataY*sizeof(double));
  cudaMalloc((void **) &phiedgeZLOW_d, nphihalodataZ*sizeof(double));
  cudaMalloc((void **) &phiedgeZHIGH_d, nphihalodataZ*sizeof(double));
  
  cudaMalloc((void **) &phihaloXLOW_d, nphihalodataX*sizeof(double));
  cudaMalloc((void **) &phihaloXHIGH_d, nphihalodataX*sizeof(double));
  cudaMalloc((void **) &phihaloYLOW_d, nphihalodataY*sizeof(double));
  cudaMalloc((void **) &phihaloYHIGH_d, nphihalodataY*sizeof(double));
  cudaMalloc((void **) &phihaloZLOW_d, nphihalodataZ*sizeof(double));
  cudaMalloc((void **) &phihaloZHIGH_d, nphihalodataZ*sizeof(double));
  
  cudaMalloc((void **) &phi_site_d, nsites*sizeof(double));
  cudaMalloc((void **) &delsq_phi_site_d, nsites*sizeof(double));
  cudaMalloc((void **) &grad_phi_site_d, nsites*3*sizeof(double));
  cudaMalloc((void **) &le_index_real_to_buffer_d, nlexbuf*sizeof(int));


  //   checkCUDAError("allocate_memory_on_gpu");

}


/* Free memory on accelerator */
static void free_phi_memory_on_gpu()
{

  /* free temp memory on host */
  free(phi_site_temp);
  free(grad_phi_site_temp);
  free(delsq_phi_site_temp);
  free(le_index_real_to_buffer_temp);

  cudaFreeHost(phiedgeXLOW);
  cudaFreeHost(phiedgeXHIGH);
  cudaFreeHost(phiedgeYLOW);
  cudaFreeHost(phiedgeYHIGH);
  cudaFreeHost(phiedgeZLOW);
  cudaFreeHost(phiedgeZHIGH);

  cudaFreeHost(phihaloXLOW);
  cudaFreeHost(phihaloXHIGH);
  cudaFreeHost(phihaloYLOW);
  cudaFreeHost(phihaloYHIGH);
  cudaFreeHost(phihaloZLOW);
  cudaFreeHost(phihaloZHIGH);


 // free(phiedgeXLOW);
 // free(phiedgeXHIGH);
 // free(phiedgeYLOW);
 // free(phiedgeYHIGH);
 // free(phiedgeZLOW);
 // free(phiedgeZHIGH);

 // free(phihaloXLOW);
 // free(phihaloXHIGH);
 // free(phihaloYLOW);
 // free(phihaloYHIGH);
 // free(phihaloZLOW);
 // free(phihaloZHIGH);

  /* free memory on accelerator */

  cudaFree(phiedgeXLOW_d);
  cudaFree(phiedgeXHIGH_d);
  cudaFree(phiedgeYLOW_d);
  cudaFree(phiedgeYHIGH_d);
  cudaFree(phiedgeZLOW_d);
  cudaFree(phiedgeZHIGH_d);

  cudaFree(phihaloXLOW_d);
  cudaFree(phihaloXHIGH_d);
  cudaFree(phihaloYLOW_d);
  cudaFree(phihaloYHIGH_d);
  cudaFree(phihaloZLOW_d);
  cudaFree(phihaloZHIGH_d);

  cudaFree(phi_site_d);
  cudaFree(delsq_phi_site_d);
  cudaFree(grad_phi_site_d);
  cudaFree(le_index_real_to_buffer_d);

}

/* copy phi from host to accelerator */
void put_phi_on_gpu()
{

  int index, ic, jc, kc;
	      

  /* get temp host copies of arrays */
  for (ic=0; ic<=Nall[X]; ic++)
    {
      for (jc=0; jc<=Nall[Y]; jc++)
	{
	  for (kc=0; kc<=Nall[Z]; kc++)
	    {
	      index = get_linear_index(ic, jc, kc, Nall); 

	      phi_site_temp[index]=phi_get_phi_site(index);
	    }
	}
    }


  /* copy data from CPU to accelerator */
  cudaMemcpy(phi_site_d, phi_site_temp, nsites*sizeof(double), \
	     cudaMemcpyHostToDevice);

  //checkCUDAError("put_phi_on_gpu");

}

/* copy phi from host to accelerator */
void put_grad_phi_on_gpu()
{

  int index, i, ic, jc, kc;
  double grad_phi[3];
	      

  /* get temp host copies of arrays */
  for (ic=1; ic<=N[X]; ic++)
    {
      for (jc=1; jc<=N[Y]; jc++)
	{
	  for (kc=1; kc<=N[Z]; kc++)
	    {
	      index = coords_index(ic, jc, kc); 


	      phi_gradients_grad(index, grad_phi);

	      for (i=0;i<3;i++)
		{
		  grad_phi_site_temp[index*3+i]=grad_phi[i];
		}
	    }
	}
    }


  /* copy data from CPU to accelerator */
  cudaMemcpy(grad_phi_site_d, grad_phi_site_temp, nsites*3*sizeof(double), \
	     cudaMemcpyHostToDevice);


  //checkCUDAError("put_phi_on_gpu");

}

/* copy phi from host to accelerator */
void put_delsq_phi_on_gpu()
{

  int index, ic, jc, kc;

  /* get temp host copies of arrays */
  for (ic=1; ic<=N[X]; ic++)
    {
      for (jc=1; jc<=N[Y]; jc++)
	{
	  for (kc=1; kc<=N[Z]; kc++)
	    {
	      index = coords_index(ic, jc, kc); 

	      delsq_phi_site_temp[index] = phi_gradients_delsq(index);
	      	      
	    }
	}
    }


  /* copy data from CPU to accelerator */
  cudaMemcpy(delsq_phi_site_d, delsq_phi_site_temp, nsites*sizeof(double), \
	     cudaMemcpyHostToDevice);

  //checkCUDAError("put_phi_on_gpu");

}



/* copy phi from accelerator to host */
void get_phi_from_gpu()
{

  int index, ic, jc, kc;
	      

  /* copy data from accelerator to host */
  cudaMemcpy(phi_site_temp, phi_site_d, nsites*sizeof(double), \
	     cudaMemcpyDeviceToHost);

  /* set phi */
  for (ic=1; ic<=N[X]; ic++)
    {
      for (jc=1; jc<=N[Y]; jc++)
	{
	  for (kc=1; kc<=N[Z]; kc++)
	    {
	      index = coords_index(ic, jc, kc); 

	      phi_set_phi_site(index,phi_site_temp[index]);

	    }
	}
    }

  //checkCUDAError("get_phi_site_from_gpu");

}



void phi_halo_swap_gpu()
{
  int NedgeX[3], NedgeY[3], NedgeZ[3];

  int ii,jj,kk,m,index_source,index_target;

#define OVERLAP

  const int tagf = 903;
  const int tagb = 904;
  
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
 pack_phi_edgesX_gpu_d<<<GridDims.x,BlockDims.x,0,streamX>>>(nop,nhalo,
						N_d,phiedgeXLOW_d,
						phiedgeXHIGH_d,phi_site_d);


 /* pack Y edges on accelerator */ 
  GridDims.x=(Nall[X]*nhalo*N[Z]+BlockDims.x-1)/BlockDims.x;
  pack_phi_edgesY_gpu_d<<<GridDims.x,BlockDims.x,0,streamY>>>(nop,nhalo,
						N_d,phiedgeYLOW_d,
						phiedgeYHIGH_d,phi_site_d);

 /* pack Z edges on accelerator */ 
    GridDims.x=(Nall[X]*Nall[Y]*nhalo+BlockDims.x-1)/BlockDims.x;
  pack_phi_edgesZ_gpu_d<<<GridDims.x,BlockDims.x,0,streamZ>>>(nop,nhalo,
  						N_d,phiedgeZLOW_d,
  						phiedgeZHIGH_d,phi_site_d);


  /* get X low edges */
  cudaMemcpyAsync(phiedgeXLOW, phiedgeXLOW_d, nphihalodataX*sizeof(double),
		  cudaMemcpyDeviceToHost,streamX);
 /* get X high edges */
  cudaMemcpyAsync(phiedgeXHIGH, phiedgeXHIGH_d, nphihalodataX*sizeof(double),
		  cudaMemcpyDeviceToHost,streamX);


#ifndef OVERLAP
  cudaStreamSynchronize(streamX);
#endif

 /* get Y low edges */
  cudaMemcpyAsync(phiedgeYLOW, phiedgeYLOW_d, nphihalodataY*sizeof(double),
		  cudaMemcpyDeviceToHost,streamY);
 /* get Y high edges */
  cudaMemcpyAsync(phiedgeYHIGH, phiedgeYHIGH_d, nphihalodataY*sizeof(double),
		  cudaMemcpyDeviceToHost,streamY);


#ifndef OVERLAP
  cudaStreamSynchronize(streamY);
#endif

  /* get Z low edges */
  cudaMemcpyAsync(phiedgeZLOW, phiedgeZLOW_d, nphihalodataZ*sizeof(double),
		  cudaMemcpyDeviceToHost,streamZ);
  /* get Z high edges */
  cudaMemcpyAsync(phiedgeZHIGH, phiedgeZHIGH_d, nphihalodataZ*sizeof(double),
		  cudaMemcpyDeviceToHost,streamZ);


#ifndef OVERLAP
  cudaStreamSynchronize(streamZ);
#endif


 /* wait for X data from accelerator*/ 
  cudaStreamSynchronize(streamX); 



   if (cart_size(X) == 1) {
     /* x up */
     memcpy(phihaloXLOW,phiedgeXHIGH,nphihalodataX*sizeof(double));
     
     /* x down */
     memcpy(phihaloXHIGH,phiedgeXLOW,nphihalodataX*sizeof(double));
     
      }
  else
    {
      /* initiate transfers */
      MPI_Irecv(phihaloXLOW, nphihalodataX, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), tagf, comm, &request[0]);
      MPI_Irecv(phihaloXHIGH, nphihalodataX, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), tagb, comm, &request[1]);
      MPI_Isend(phiedgeXHIGH, nphihalodataX, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), tagf, comm, &request[2]);
      MPI_Isend(phiedgeXLOW,  nphihalodataX, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), tagb, comm, &request[3]);
     }


 /* wait for X halo swaps to finish */ 
   if (cart_size(X) > 1)       MPI_Waitall(4, request, status);


 /* put X halos back on device, and unpack */
  cudaMemcpyAsync(phihaloXLOW_d, phihaloXLOW, nphihalodataX*sizeof(double),
		  cudaMemcpyHostToDevice,streamX);
  cudaMemcpyAsync(phihaloXHIGH_d, phihaloXHIGH, nphihalodataX*sizeof(double),
		  cudaMemcpyHostToDevice,streamX);
  GridDims.x=(nhalo*N[Y]*N[Z]+BlockDims.x-1)/BlockDims.x;
  unpack_phi_halosX_gpu_d<<<GridDims.x,BlockDims.x,0,streamX>>>(nop,nhalo,
						  N_d,phi_site_d,phihaloXLOW_d,
						  phihaloXHIGH_d);



#ifndef OVERLAP
  cudaStreamSynchronize(streamX);
#endif

  /* wait for Y data from accelerator*/ 
  cudaStreamSynchronize(streamY); 


  /* fill in corners of Y edge data  */

  for (m=0;m<nop;m++)
    {
      
      
      for (ii = 0; ii < nhalo; ii++) {
	for (jj = 0; jj < nhalo; jj++) {
	  for (kk = 0; kk < N[Z]; kk++) {
	    
	    
	    
	    /* xlow part of ylow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(ii,jj,kk,NedgeY);
	    
	    phiedgeYLOW[npackedsiteY*m+index_target] =
	      phihaloXLOW[npackedsiteX*m+index_source];
	    
	    /* xlow part of yhigh */
	    index_source = get_linear_index(ii,NedgeX[Y]-1-jj,kk,NedgeX);
	    index_target = get_linear_index(ii,jj,kk,NedgeY);
	    
	    phiedgeYHIGH[npackedsiteY*m+index_target] =
	      phihaloXLOW[npackedsiteX*m+index_source];
	    
	    
	    /* get high X data */
	    
	    /* xhigh part of ylow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(NedgeY[X]-1-ii,jj,kk,NedgeY);
	    
	    phiedgeYLOW[npackedsiteY*m+index_target] =
	      phihaloXHIGH[npackedsiteX*m+index_source];
	    
	    /* xhigh part of yhigh */
	    index_source = get_linear_index(ii,NedgeX[Y]-1-jj,kk,NedgeX);			index_target = get_linear_index(NedgeY[X]-1-ii,jj,kk,NedgeY);
	    
	    phiedgeYHIGH[npackedsiteY*m+index_target] =
	      phihaloXHIGH[npackedsiteX*m+index_source];
	    
	    
	    
	  }
	}
	
      }
    }
  


  /* The y-direction (XZ plane) */
   if (cart_size(Y) == 1) {
  /* y up */
  memcpy(phihaloYLOW,phiedgeYHIGH,nphihalodataY*sizeof(double));
  
  /* y down */
  memcpy(phihaloYHIGH,phiedgeYLOW,nphihalodataY*sizeof(double));
  
      }
  else
    {
      /* initiate transfers */
      MPI_Irecv(phihaloYLOW, nphihalodataY, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), tagf, comm, &request[0]);
      MPI_Irecv(phihaloYHIGH, nphihalodataY, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), tagb, comm, &request[1]);
      MPI_Isend(phiedgeYHIGH, nphihalodataY, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), tagf, comm, &request[2]);
      MPI_Isend(phiedgeYLOW,  nphihalodataY, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), tagb, comm, &request[3]);
    }


 /* wait for Y halo swaps to finish */ 
    if (cart_size(Y) > 1)       MPI_Waitall(4, request, status); 

 /* put Y halos back on device, and unpack */
  cudaMemcpyAsync(phihaloYLOW_d, phihaloYLOW, nphihalodataY*sizeof(double),
		  cudaMemcpyHostToDevice,streamY);
  cudaMemcpyAsync(phihaloYHIGH_d, phihaloYHIGH, nphihalodataY*sizeof(double),
		  cudaMemcpyHostToDevice,streamY);
  GridDims.x=(Nall[X]*nhalo*N[Z]+BlockDims.x-1)/BlockDims.x;
  unpack_phi_halosY_gpu_d<<<GridDims.x,BlockDims.x,0,streamY>>>(nop,nhalo,
						  N_d,phi_site_d,phihaloYLOW_d,
						  phihaloYHIGH_d);



#ifndef OVERLAP
  cudaStreamSynchronize(streamY);
#endif

 
  /* wait for Z data from accelerator*/ 
  cudaStreamSynchronize(streamZ); 

  /* fill in corners of Z edge data: from Xhalo  */
    
  for (m=0;m<nop;m++)
    {
      
      for (ii = 0; ii < nhalo; ii++) {
	for (jj = 0; jj < N[Y]; jj++) {
	  for (kk = 0; kk < nhalo; kk++) {
	    
	    
	    
	    /* xlow part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(ii,jj+nhalo,kk,NedgeZ);
	    
	    phiedgeZLOW[npackedsiteZ*m+index_target] =
	      phihaloXLOW[npackedsiteX*m+index_source];
	    
	    
	    /* xlow part of zhigh */
	    index_source = get_linear_index(ii,jj,NedgeX[Z]-1-kk,NedgeX);
	    index_target = get_linear_index(ii,jj+nhalo,kk,NedgeZ);
	    
	    phiedgeZHIGH[npackedsiteZ*m+index_target] =
	      phihaloXLOW[npackedsiteX*m+index_source];
	    
	    
	    
	    /* xhigh part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(NedgeZ[X]-1-ii,jj+nhalo,kk,
					    NedgeZ);
	    
	    phiedgeZLOW[npackedsiteZ*m+index_target] =
	      phihaloXHIGH[npackedsiteX*m+index_source];
	    
	    
	    /* xhigh part of zhigh */
	    
	    index_source = get_linear_index(ii,jj,NedgeX[Z]-1-kk,NedgeX);
	    index_target = get_linear_index(NedgeZ[X]-1-ii,jj+nhalo,kk,
					    NedgeZ);
	    
	    phiedgeZHIGH[npackedsiteZ*m+index_target] =
	      phihaloXHIGH[npackedsiteX*m+index_source];
	    
	    
	  }
	}
	
	
      }
    }
  
  /* fill in corners of Z edge data: from Yhalo  */
  
  
  
  for (m=0;m<nop;m++)
    {
      
      
      
      for (ii = 0; ii < Nall[X]; ii++) {
	for (jj = 0; jj < nhalo; jj++) {
	  for (kk = 0; kk < nhalo; kk++) {
	    
	    
	    
	    /* ylow part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeY);
	    index_target = get_linear_index(ii,jj,kk,NedgeZ);
	    
	    phiedgeZLOW[npackedsiteZ*m+index_target] =
	      phihaloYLOW[npackedsiteY*m+index_source];
	    
	    
	    /* ylow part of zhigh */
	    index_source = get_linear_index(ii,jj,NedgeY[Z]-1-kk,NedgeY);
	    index_target = get_linear_index(ii,jj,kk,NedgeZ);
	    
	    phiedgeZHIGH[npackedsiteZ*m+index_target] =
	      phihaloYLOW[npackedsiteY*m+index_source];
	    
	    
	    
	    /* yhigh part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeY);
	    index_target = get_linear_index(ii,NedgeZ[Y]-1-jj,kk,NedgeZ);
	    
	    phiedgeZLOW[npackedsiteZ*m+index_target] =
	      phihaloYHIGH[npackedsiteY*m+index_source];
	    
	    
	    /* yhigh part of zhigh */
	    
	    index_source = get_linear_index(ii,jj,NedgeY[Z]-1-kk,NedgeY);
	    index_target = get_linear_index(ii,NedgeZ[Y]-1-jj,kk,NedgeZ);
	    
	    phiedgeZHIGH[npackedsiteZ*m+index_target] =
	      phihaloYHIGH[npackedsiteY*m+index_source];
	    
	    
	  }
	}
	
      }
    }
  


  /* The z-direction (xy plane) */
   if (cart_size(Z) == 1) {
  /* z up */
  memcpy(phihaloZLOW,phiedgeZHIGH,nphihalodataZ*sizeof(double));

  /* z down */
  memcpy(phihaloZHIGH,phiedgeZLOW,nphihalodataZ*sizeof(double));
      }
  else
    {
      MPI_Irecv(phihaloZLOW, nphihalodataZ, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), tagf, comm, &request[0]);
      MPI_Irecv(phihaloZHIGH, nphihalodataZ, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), tagb, comm, &request[1]);
      MPI_Isend(phiedgeZHIGH, nphihalodataZ, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), tagf, comm, &request[2]);
      MPI_Isend(phiedgeZLOW,  nphihalodataZ, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), tagb, comm, &request[3]);
      MPI_Waitall(4, request, status);

    }

 /* put Z halos back on device, and unpack */
  cudaMemcpyAsync(phihaloZLOW_d, phihaloZLOW, nphihalodataZ*sizeof(double),
		  cudaMemcpyHostToDevice,streamZ);
  cudaMemcpyAsync(phihaloZHIGH_d, phihaloZHIGH, nphihalodataZ*sizeof(double),
		  cudaMemcpyHostToDevice,streamZ);

  GridDims.x=(Nall[X]*Nall[Y]*nhalo+BlockDims.x-1)/BlockDims.x;
   unpack_phi_halosZ_gpu_d<<<GridDims.x,BlockDims.x,0,streamZ>>>(nop,nhalo,
  						  N_d,phi_site_d,phihaloZLOW_d,
  					  phihaloZHIGH_d);


  /* wait for all streams to complete */
  cudaStreamSynchronize(streamX);
  cudaStreamSynchronize(streamY);
  cudaStreamSynchronize(streamZ);
  

}


/* pack X edges on the accelerator */
__global__ static void pack_phi_edgesX_gpu_d(int nop, int nhalo,
					 int N[3],
					 double* phiedgeXLOW_d,
					 double* phiedgeXHIGH_d, double* phi_site_d)
{

  int m,index,ii,jj,kk,packed_index;

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

      
      /* copy data to packed structure */
	    for (m = 0; m < nop; m++) {
	      phiedgeXLOW_d[m*npackedsite+packed_index]
		= phi_site_d[nsite*m+index];
	    }
      
  
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(Nall[X]-nhalo-1-ii,jj+nhalo,kk+nhalo,Nall);
      /* copy data to packed structure */
	    for (m = 0; m < nop; m++) {
	      
	      phiedgeXHIGH_d[m*npackedsite+packed_index]
		= phi_site_d[nsite*m+index];
	      
	    }
    }
  
}

/* unpack X halos on the accelerator */
__global__ static void unpack_phi_halosX_gpu_d(int nop, int nhalo,
					   int N[3],
					   double* phi_site_d, double* phihaloXLOW_d,
					   double* phihaloXHIGH_d)
{


  int m,index,ii,jj,kk,packed_index;


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

      /* LOW HALO */
      index = get_linear_index_gpu_d(ii,jj+nhalo,kk+nhalo,Nall);
      
      /* copy packed structure data to original array */
	    for (m = 0; m < nop; m++) {
	  
	      phi_site_d[nsite*m+index] =
		phihaloXLOW_d[m*npackedsite+packed_index];

	    }
           
  
      /* HIGH HALO */
      index = get_linear_index_gpu_d(Nall[X]-1-ii,jj+nhalo,kk+nhalo,Nall);
      /* copy packed structure data to original array */
	    for (m = 0; m < nop; m++) {
	      
	      phi_site_d[nsite*m+index] =
		phihaloXHIGH_d[m*npackedsite+packed_index];
	      
	    }
    }
  
  
}


/* pack Y edges on the accelerator */
__global__ static void pack_phi_edgesY_gpu_d(int nop, int nhalo,
					 int N[3], 					 double* phiedgeYLOW_d,
					 double* phiedgeYHIGH_d, double* phi_site_d) {

  int m,index,ii,jj,kk,packed_index;


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

      
      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj+nhalo,kk+nhalo,Nall);
      
      /* copy data to packed structure */
	    for (m = 0; m < nop; m++) {
	      
	      phiedgeYLOW_d[m*npackedsite+packed_index]
		= phi_site_d[nsite*m+index];
	      
	    }
      
      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,Nall[Y]-nhalo-1-jj,kk+nhalo,Nall);
      /* copy data to packed structure */
	    for (m = 0; m < nop; m++) {
	      
	      phiedgeYHIGH_d[m*npackedsite+packed_index]
		= phi_site_d[nsite*m+index];
	      
	    }
    }
  
  
}




/* unpack Y halos on the accelerator */
__global__ static void unpack_phi_halosY_gpu_d(int nop, int nhalo,
					  int N[3],
					   double* phi_site_d, double* phihaloYLOW_d,
					   double* phihaloYHIGH_d)
{

  int m,index,ii,jj,kk,packed_index;

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



      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj,kk+nhalo,Nall);
      
       /* copy packed structure data to original array */

	    for (m = 0; m < nop; m++) {
	      
	      phi_site_d[nsite*m+index] =
	      phihaloYLOW_d[m*npackedsite+packed_index];
	      
	    }
      

      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,Nall[Y]-1-jj,kk+nhalo,Nall);
      /* copy packed structure data to original array */
	    for (m = 0; m < nop; m++) {
	      
	      phi_site_d[nsite*m+index] =
		phihaloYHIGH_d[m*npackedsite+packed_index];
	      
	    }
      
    }
  
  
  
}



/* pack Z edges on the accelerator */
__global__ static void pack_phi_edgesZ_gpu_d(int nop, int nhalo,
					 int N[3],
					 double* phiedgeZLOW_d,
					 double* phiedgeZHIGH_d, double* phi_site_d)
{

  int m,index,ii,jj,kk,packed_index;

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
      
      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj,kk+nhalo,Nall);
      
      /* copy data to packed structure */
	    for (m = 0; m < nop; m++) {
	      
	      phiedgeZLOW_d[m*npackedsite+packed_index]
		= phi_site_d[nsite*m+index];
	      
	    }
      
      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,jj,Nall[Z]-nhalo-1-kk,Nall);
      /* copy data to packed structure */
	    for (m = 0; m < nop; m++) {
	      
	      phiedgeZHIGH_d[m*npackedsite+packed_index]
		= phi_site_d[nsite*m+index];
	      
	    }
    }
  
  
}




/* unpack Z halos on the accelerator */
__global__ static void unpack_phi_halosZ_gpu_d(int nop, int nhalo,
					   int N[3],
					   double* phi_site_d, double* phihaloZLOW_d,
					   double* phihaloZHIGH_d)
{

  int m,index,ii,jj,kk,packed_index;
  
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

      
      /* LOW EDGE */
      index = get_linear_index_gpu_d(ii,jj,kk,Nall);
      
      /* copy packed structure data to original array */
	    for (m = 0; m < nop; m++) {
	      
	      phi_site_d[nsite*m+index] =
		phihaloZLOW_d[m*npackedsite+packed_index];
	      
	    }
      
      
      /* HIGH EDGE */
      index = get_linear_index_gpu_d(ii,jj,Nall[Z]-1-kk,Nall);
      /* copy packed structure data to original array */
	    for (m = 0; m < nop; m++) {
	      
	      phi_site_d[nsite*m+index] =
		phihaloZHIGH_d[m*npackedsite+packed_index];
	      
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

