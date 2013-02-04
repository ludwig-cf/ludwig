/*****************************************************************************
 *
 * comms_gpu.cu
 * 
 * Alan Gray
 *
 * 
 *****************************************************************************/

#include <stdio.h>

#include "comms_gpu.h"
#include "comms_internal_gpu.h"
#include "utilities_gpu.h"
#include "common_gpu.h"
#include "model.h" 
extern "C" int  RUN_get_string_parameter(const char *, char *, const int);

/* external pointers to data on host*/
extern double * f_;
extern double * ftmp;
extern double * colloid_force_tmp;
extern double * velocity_d;

/* external pointers to data on accelerator*/
extern int * cv_d;
extern int * N_d;
extern double * f_d;
extern double * ftmp_d;

int *packedindex_d;
char *mask_d;
char *mask_;
char *mask_with_neighbours;


/* edge and halo buffers on accelerator */
static double * edgeXLOW_d;
static double * edgeXHIGH_d;
static double * edgeYLOW_d;
static double * edgeYHIGH_d;
static double * edgeZLOW_d;
static double * edgeZHIGH_d;
static double * haloXLOW_d;
static double * haloXHIGH_d;
static double * haloYLOW_d;
static double * haloYHIGH_d;
static double * haloZLOW_d;
static double * haloZHIGH_d;


/* edge and halo buffers on host */
static double * edgeXLOW;
static double * edgeXHIGH;
static double * edgeYLOW;
static double * edgeYHIGH;
static double * edgeZLOW;
static double * edgeZHIGH;
static double * haloXLOW;
static double * haloXHIGH;
static double * haloYLOW;
static double * haloYHIGH;
static double * haloZLOW;
static double * haloZHIGH;




static int * packedindex;


/* data size variables */
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

/* handles for CUDA streams (for ovelapping)*/
static cudaStream_t streamX,streamY, streamZ;


static int reduced_halo=0;

/* constant memory symbols internal to this module */
__constant__ int cv_cd[NVEL][3];


/* Perform tasks necessary to initialise accelerator */
void init_comms_gpu()
{

  calculate_comms_data_sizes();
  allocate_comms_memory_on_gpu();


  char string[FILENAME_MAX];

  RUN_get_string_parameter("reduced_halo", string, FILENAME_MAX);
  if (strcmp(string, "yes") == 0) reduced_halo = 1;
  
  /* create CUDA streams (for ovelapping)*/
  cudaStreamCreate(&streamX);
  cudaStreamCreate(&streamY);
  cudaStreamCreate(&streamZ);


  cudaMemcpyToSymbol(cv_cd, cv, NVEL*3*sizeof(int), 0, cudaMemcpyHostToDevice); 
 
  //checkCUDAError("Init GPU");  


}

void finalise_comms_gpu()
{
  free_comms_memory_on_gpu();

  cudaStreamDestroy(streamX);
  cudaStreamDestroy(streamY);
  cudaStreamDestroy(streamZ);

}


/* calculate sizes of data - needed for memory copies to accelerator */
static void calculate_comms_data_sizes()
{
  coords_nlocal(N);  
  nhalo = coords_nhalo();  
  ndist = distribution_ndist();
  nop = phi_nop();

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsites = Nall[X]*Nall[Y]*Nall[Z];

  

  /* calculate number of velocity components when packed */
  int p;
  npvel=0;
  for (p=0; p<NVEL; p++)
    {
      if (cv[p][0] == 1 || !reduced_halo) npvel++; 
    }

  int n1=ndist*npvel;
  if (nop > n1) n1=nop;

  nhalodataX = N[Y] * N[Z] * nhalo * n1;
  nhalodataY = Nall[X] * N[Z] * nhalo * n1;
  nhalodataZ = Nall[X] * Nall[Y] * nhalo * n1;



}



/* Allocate memory on accelerator */
static void allocate_comms_memory_on_gpu()
{

  
  cudaHostAlloc( (void **)&packedindex, nsites*sizeof(int), 
		 cudaHostAllocDefault);

  cudaHostAlloc( (void **)&mask_, nsites*sizeof(char), 
		 cudaHostAllocDefault);

  cudaHostAlloc( (void **)&mask_with_neighbours, nsites*sizeof(char), 
		 cudaHostAllocDefault);

  cudaHostAlloc( (void **)&edgeXLOW, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&edgeXHIGH, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&edgeYLOW, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&edgeYHIGH, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&edgeZLOW, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&edgeZHIGH, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);


  cudaHostAlloc( (void **)&haloXLOW, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&haloXHIGH, nhalodataX*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&haloYLOW, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&haloYHIGH, nhalodataY*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&haloZLOW, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);
  cudaHostAlloc( (void **)&haloZHIGH, nhalodataZ*sizeof(double), 
		 cudaHostAllocDefault);




  
  
  cudaMalloc((void **) &edgeXLOW_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &edgeXHIGH_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &edgeYLOW_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &edgeYHIGH_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &edgeZLOW_d, nhalodataZ*sizeof(double));
  cudaMalloc((void **) &edgeZHIGH_d, nhalodataZ*sizeof(double));
  
  cudaMalloc((void **) &haloXLOW_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &haloXHIGH_d, nhalodataX*sizeof(double));
  cudaMalloc((void **) &haloYLOW_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &haloYHIGH_d, nhalodataY*sizeof(double));
  cudaMalloc((void **) &haloZLOW_d, nhalodataZ*sizeof(double));
  cudaMalloc((void **) &haloZHIGH_d, nhalodataZ*sizeof(double));

  cudaMalloc((void **) &mask_d, nsites*sizeof(char));
  cudaMalloc((void **) &packedindex_d, nsites*sizeof(int));




  //   checkCUDAError("allocate_memory_on_gpu");

}


/* Free memory on accelerator */
static void free_comms_memory_on_gpu()
{



  cudaFreeHost(packedindex);
  cudaFreeHost(mask_);
  cudaFreeHost(mask_with_neighbours);

  cudaFreeHost(edgeXLOW);
  cudaFreeHost(edgeXHIGH);
  cudaFreeHost(edgeYLOW);
  cudaFreeHost(edgeYHIGH);
  cudaFreeHost(edgeZLOW);
  cudaFreeHost(edgeZHIGH);

  cudaFreeHost(haloXLOW);
  cudaFreeHost(haloXHIGH);
  cudaFreeHost(haloYLOW);
  cudaFreeHost(haloYHIGH);
  cudaFreeHost(haloZLOW);
  cudaFreeHost(haloZHIGH);

  /* free memory on accelerator */

  cudaFree(mask_d);
  cudaFree(packedindex_d);

  cudaFree(edgeXLOW_d);
  cudaFree(edgeXHIGH_d);
  cudaFree(edgeYLOW_d);
  cudaFree(edgeYHIGH_d);
  cudaFree(edgeZLOW_d);
  cudaFree(edgeZHIGH_d);

  cudaFree(haloXLOW_d);
  cudaFree(haloXHIGH_d);
  cudaFree(haloYLOW_d);
  cudaFree(haloYHIGH_d);
  cudaFree(haloZLOW_d);
  cudaFree(haloZHIGH_d);

}







void fill_mask_with_neighbours(char *mask)
{

  int i, ib[3], p;

  for (i=0; i<nsites; i++)
    mask_with_neighbours[i]=0;


  for (i=0; i<nsites; i++){
    if(mask[i]){
      mask_with_neighbours[i]=1;
      coords_index_to_ijk(i, ib);
      /* if not a halo */
      int halo = (ib[X] < 1 || ib[Y] < 1 || ib[Z] < 1 ||
		  ib[X] > N[X] || ib[Y] > N[Y] || ib[Z] > N[Z]);
      
      if (!halo){
	
	for (p=1; p<NVEL; p++){
	  int indexn = coords_index(ib[X] + cv[p][X], ib[Y] + cv[p][Y],
				    ib[Z] + cv[p][Z]);
	  mask_with_neighbours[indexn]=1;
	}
      }
    }
    
  }
  
  

}




void put_field_partial_on_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *)){

  char *mask;
  int i;
  int index;
  double field_tmp[50];
  
  if(include_neighbours){
    fill_mask_with_neighbours(mask_);
    mask=mask_with_neighbours;
  }
  else{
    mask=mask_;
  }



  int packedsize=0;
  for (index=0; index<nsites; index++){
    if(mask[index]) packedsize++;
  }


  int j=0;
  for (index=0; index<nsites; index++){
    
    if(mask[index]){
 
      access_function(index,field_tmp);
      
      for (i=0;i<(nfields1*nfields2);i++)
	{
	  ftmp[i*packedsize+j]=field_tmp[i];
	}
      
      packedindex[index]=j;
      j++;

    }

  }

  cudaMemcpy(ftmp_d, ftmp, packedsize*nfields1*nfields2*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mask_d, mask, nsites*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(packedindex_d, packedindex, nsites*sizeof(int), cudaMemcpyHostToDevice);

  /* run the GPU kernel */

  int nblocks=(Nall[X]*Nall[Y]*Nall[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(nfields1*nfields2, nhalo, N_d,
  						data_d, ftmp_d, mask_d,
  						packedindex_d, packedsize, 1);
  cudaThreadSynchronize();
  checkCUDAError("put_partial_field_on_gpu");

}


/* copy part of velocity_ from accelerator to host, using mask structure */
void get_field_partial_from_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *))
{


  char *mask;
  int i;
  int index;
  double field_tmp[50];

  if(include_neighbours){
    fill_mask_with_neighbours(mask_);
    mask=mask_with_neighbours;
  }
  else{
    mask=mask_;
  }

  int j=0;
  for (i=0; i<nsites; i++){
    if(mask[i]){
      packedindex[i]=j;
      j++;
    }
    
  }

  int packedsize=j;

  cudaMemcpy(mask_d, mask, nsites*sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(packedindex_d, packedindex, nsites*sizeof(int), cudaMemcpyHostToDevice);

  int nblocks=(Nall[X]*Nall[Y]*Nall[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
 copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(nfields1*nfields2, nhalo, N_d,
  						ftmp_d, data_d, mask_d,
  						packedindex_d, packedsize, 0);
  cudaThreadSynchronize();

  cudaMemcpy(ftmp, ftmp_d, packedsize*nfields1*nfields2*sizeof(double), cudaMemcpyDeviceToHost); 

  j=0;
  for (index=0; index<nsites; index++){
    
    if(mask[index]){
 
      for (i=0;i<nfields1*nfields2;i++)
	{
	  field_tmp[i]=ftmp[i*packedsize+j];
	}
      access_function(index,field_tmp);       
      j++;

    }

  }



  /* run the GPU kernel */

  checkCUDAError("get_field_partial_from_gpu");

}


__global__ static void copy_field_partial_gpu_d(int nPerSite, int nhalo, int N[3],
					    double* f_out, double* f_in, char *mask_d, int *packedindex_d, int packedsize, int inpack) {

  int threadIndex, nsite, Nall[3];
  int i;


  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsite = Nall[X]*Nall[Y]*Nall[Z];


  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;

  //Avoid going beyond problem domain
  if ((threadIndex < Nall[X]*Nall[Y]*Nall[Z]) && mask_d[threadIndex])
    {

      for (i=0;i<nPerSite;i++)
	{
	    
	  if (inpack)
	    f_out[i*nsite+threadIndex]
	    =f_in[i*packedsize+packedindex_d[threadIndex]];
	  else
	   f_out[i*packedsize+packedindex_d[threadIndex]]
	      =f_in[i*nsite+threadIndex];
	  
	}
    }


  return;
}



void halo_gpu(int nfields1, int nfields2, int packablefield1, double * data_d)
{


  int pack_field1=packablefield1*reduced_halo;
  int nfields1packed;
  if (packablefield1){
    /* calculate number of velocity components when packed */
    int p;
    nfields1packed=0;
    for (p=0; p<NVEL; p++)
      {
	if (cv[p][0] == 1 || !reduced_halo) nfields1packed++; 
      }
  }
  else{
    nfields1packed=nfields1;
  }


  int NedgeX[3], NedgeY[3], NedgeZ[3];

  int ii,jj,kk,m,index_source,index_target;

  int nblocks;

#define OVERLAP

  const int tagf = 903;
  const int tagb = 904;
  
  MPI_Request request[4];
  MPI_Status status[4];
  MPI_Comm comm = cart_comm();


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
 nblocks=(nhalo*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
 pack_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamX>>>(nfields1,nfields2,nhalo,
						pack_field1, N_d,edgeXLOW_d,
						     edgeXHIGH_d,data_d,X);


 /* pack Y edges on accelerator */
  nblocks=(Nall[X]*nhalo*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
  pack_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamY>>>(nfields1,nfields2,nhalo,
						pack_field1, N_d,edgeYLOW_d,
						     edgeYHIGH_d,data_d,Y);

 /* pack Z edges on accelerator */
    nblocks=(Nall[X]*Nall[Y]*nhalo+DEFAULT_TPB-1)/DEFAULT_TPB;
  pack_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamZ>>>(nfields1,nfields2,nhalo,
  						pack_field1, N_d,edgeZLOW_d,
						     edgeZHIGH_d,data_d,Z);


  /* get X low edges */
  cudaMemcpyAsync(edgeXLOW, edgeXLOW_d, nhalodataX*sizeof(double),
		  cudaMemcpyDeviceToHost,streamX);
 /* get X high edges */
  cudaMemcpyAsync(edgeXHIGH, edgeXHIGH_d, nhalodataX*sizeof(double),
		  cudaMemcpyDeviceToHost,streamX);


#ifndef OVERLAP
  cudaStreamSynchronize(streamX);
#endif

 /* get Y low edges */
  cudaMemcpyAsync(edgeYLOW, edgeYLOW_d, nhalodataY*sizeof(double),
		  cudaMemcpyDeviceToHost,streamY);
 /* get Y high edges */
  cudaMemcpyAsync(edgeYHIGH, edgeYHIGH_d, nhalodataY*sizeof(double),
		  cudaMemcpyDeviceToHost,streamY);


#ifndef OVERLAP
  cudaStreamSynchronize(streamY);
#endif

  /* get Z low edges */
  cudaMemcpyAsync(edgeZLOW, edgeZLOW_d, nhalodataZ*sizeof(double),
		  cudaMemcpyDeviceToHost,streamZ);
  /* get Z high edges */
  cudaMemcpyAsync(edgeZHIGH, edgeZHIGH_d, nhalodataZ*sizeof(double),
		  cudaMemcpyDeviceToHost,streamZ);


#ifndef OVERLAP
  cudaStreamSynchronize(streamZ);
#endif


 /* wait for X data from accelerator*/
  cudaStreamSynchronize(streamX);



   if (cart_size(X) == 1) {
     /* x up */
     memcpy(haloXLOW,edgeXHIGH,nhalodataX*sizeof(double));
     
     /* x down */
     memcpy(haloXHIGH,edgeXLOW,nhalodataX*sizeof(double));
     
      }
  else
    {
      /* initiate transfers */
      MPI_Irecv(haloXLOW, nhalodataX, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), tagf, comm, &request[0]);
      MPI_Irecv(haloXHIGH, nhalodataX, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), tagb, comm, &request[1]);
      MPI_Isend(edgeXHIGH, nhalodataX, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), tagf, comm, &request[2]);
      MPI_Isend(edgeXLOW,  nhalodataX, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), tagb, comm, &request[3]);
     }


 /* wait for X halo swaps to finish */
   if (cart_size(X) > 1)       MPI_Waitall(4, request, status);


 /* put X halos back on device, and unpack */
  cudaMemcpyAsync(haloXLOW_d, haloXLOW, nhalodataX*sizeof(double),
		  cudaMemcpyHostToDevice,streamX);
  cudaMemcpyAsync(haloXHIGH_d, haloXHIGH, nhalodataX*sizeof(double),
		  cudaMemcpyHostToDevice,streamX);
  nblocks=(nhalo*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
     unpack_halo_gpu_d<<<nblocks,DEFAULT_TPB,0,streamX>>>(nfields1,nfields2,nhalo,
  						  pack_field1, N_d,data_d,haloXLOW_d,
							      haloXHIGH_d,X);



#ifndef OVERLAP
  cudaStreamSynchronize(streamX);
#endif

  /* wait for Y data from accelerator*/
  cudaStreamSynchronize(streamY);


  /* fill in corners of Y edge data  */

  for (m=0;m<(nfields1packed*nfields2);m++)
    {
      
      
      for (ii = 0; ii < nhalo; ii++) {
	for (jj = 0; jj < nhalo; jj++) {
	  for (kk = 0; kk < N[Z]; kk++) {
	    
	    
	    
	    /* xlow part of ylow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(ii,jj,kk,NedgeY);
	    
	    edgeYLOW[npackedsiteY*m+index_target] =
	      haloXLOW[npackedsiteX*m+index_source];
	    
	    /* xlow part of yhigh */
	    index_source = get_linear_index(ii,NedgeX[Y]-nhalo+jj,kk,NedgeX);
	    index_target = get_linear_index(ii,jj,kk,NedgeY);
	    
	    edgeYHIGH[npackedsiteY*m+index_target] =
	      haloXLOW[npackedsiteX*m+index_source];
	    
	    
	    /* get high X data */
	    
	    /* xhigh part of ylow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(NedgeY[X]-nhalo+ii,jj,kk,NedgeY);
	    
	    edgeYLOW[npackedsiteY*m+index_target] =
	      haloXHIGH[npackedsiteX*m+index_source];
	    
	    /* xhigh part of yhigh */
	    
	    index_source = get_linear_index(ii,NedgeX[Y]-nhalo+jj,kk,NedgeX);			index_target = get_linear_index(NedgeY[X]-nhalo+ii,jj,kk,NedgeY);
	    
	    edgeYHIGH[npackedsiteY*m+index_target] =
	      haloXHIGH[npackedsiteX*m+index_source];
	    
	    
	    
	  }
	}
	
      }
    }
  


  /* The y-direction (XZ plane) */
   if (cart_size(Y) == 1) {
  /* y up */
  memcpy(haloYLOW,edgeYHIGH,nhalodataY*sizeof(double));
  
  /* y down */
  memcpy(haloYHIGH,edgeYLOW,nhalodataY*sizeof(double));
  
      }
  else
    {
      /* initiate transfers */
      MPI_Irecv(haloYLOW, nhalodataY, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), tagf, comm, &request[0]);
      MPI_Irecv(haloYHIGH, nhalodataY, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), tagb, comm, &request[1]);
      MPI_Isend(edgeYHIGH, nhalodataY, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), tagf, comm, &request[2]);
      MPI_Isend(edgeYLOW,  nhalodataY, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), tagb, comm, &request[3]);
    }


 /* wait for Y halo swaps to finish */
    if (cart_size(Y) > 1)       MPI_Waitall(4, request, status);

 /* put Y halos back on device, and unpack */
  cudaMemcpyAsync(haloYLOW_d, haloYLOW, nhalodataY*sizeof(double),
		  cudaMemcpyHostToDevice,streamY);
  cudaMemcpyAsync(haloYHIGH_d, haloYHIGH, nhalodataY*sizeof(double),
		  cudaMemcpyHostToDevice,streamY);
  nblocks=(Nall[X]*nhalo*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
    unpack_halo_gpu_d<<<nblocks,DEFAULT_TPB,0,streamY>>>(nfields1,nfields2,nhalo,
  						  pack_field1, N_d,data_d,haloYLOW_d,
							 haloYHIGH_d,Y);



#ifndef OVERLAP
  cudaStreamSynchronize(streamY);
#endif

 
  /* wait for Z data from accelerator*/
  cudaStreamSynchronize(streamZ);

  /* fill in corners of Z edge data: from Xhalo  */
    
  for (m=0;m<(nfields1packed*nfields2);m++)
    {
      
      for (ii = 0; ii < nhalo; ii++) {
	for (jj = 0; jj < N[Y]; jj++) {
	  for (kk = 0; kk < nhalo; kk++) {
	    
	    
	    
	    /* xlow part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(ii,jj+nhalo,kk,NedgeZ);
	    
	    edgeZLOW[npackedsiteZ*m+index_target] =
	      haloXLOW[npackedsiteX*m+index_source];
	    
	    
	    /* xlow part of zhigh */
	    index_source = get_linear_index(ii,jj,NedgeX[Z]-nhalo+kk,NedgeX);
	    index_target = get_linear_index(ii,jj+nhalo,kk,NedgeZ);
	    
	    edgeZHIGH[npackedsiteZ*m+index_target] =
	      haloXLOW[npackedsiteX*m+index_source];
	    
	    
	    
	    /* xhigh part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeX);
	    index_target = get_linear_index(NedgeZ[X]-nhalo+ii,jj+nhalo,kk,
					    NedgeZ);
	    
	    edgeZLOW[npackedsiteZ*m+index_target] =
	      haloXHIGH[npackedsiteX*m+index_source];
	    
	    
	    /* xhigh part of zhigh */
	    index_source = get_linear_index(ii,jj,NedgeX[Z]-nhalo+kk,NedgeX);
	    index_target = get_linear_index(NedgeZ[X]-nhalo+ii,jj+nhalo,kk,
					    NedgeZ);
	    
	    edgeZHIGH[npackedsiteZ*m+index_target] =
	      haloXHIGH[npackedsiteX*m+index_source];
	    
	    
	  }
	}
	
	
      }
    }
  
  /* fill in corners of Z edge data: from Yhalo  */
  
  
  
  for (m=0;m<(nfields1packed*nfields2);m++)
    {
      
      
      
      for (ii = 0; ii < Nall[X]; ii++) {
	for (jj = 0; jj < nhalo; jj++) {
	  for (kk = 0; kk < nhalo; kk++) {
	    
	    
	    
	    /* ylow part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeY);
	    index_target = get_linear_index(ii,jj,kk,NedgeZ);
	    
	    edgeZLOW[npackedsiteZ*m+index_target] =
	      haloYLOW[npackedsiteY*m+index_source];
	    
	    
	    /* ylow part of zhigh */
	    index_source = get_linear_index(ii,jj,NedgeY[Z]-nhalo+kk,NedgeY);
	    index_target = get_linear_index(ii,jj,kk,NedgeZ);
	    
	    edgeZHIGH[npackedsiteZ*m+index_target] =
	      haloYLOW[npackedsiteY*m+index_source];
	    
	    
	    
	    /* yhigh part of zlow */
	    index_source = get_linear_index(ii,jj,kk,NedgeY);
	    index_target = get_linear_index(ii,NedgeZ[Y]-nhalo+jj,kk,NedgeZ);
	    
	    edgeZLOW[npackedsiteZ*m+index_target] =
	      haloYHIGH[npackedsiteY*m+index_source];
	    
	    
	    /* yhigh part of zhigh */
	    
	    index_source = get_linear_index(ii,jj,NedgeY[Z]-nhalo+kk,NedgeY);
	    index_target = get_linear_index(ii,NedgeZ[Y]-nhalo+jj,kk,NedgeZ);
	    
	    edgeZHIGH[npackedsiteZ*m+index_target] =
	      haloYHIGH[npackedsiteY*m+index_source];
	    
	    
	  }
	}
	
      }
    }
  


  /* The z-direction (xy plane) */
   if (cart_size(Z) == 1) {
  /* z up */
  memcpy(haloZLOW,edgeZHIGH,nhalodataZ*sizeof(double));

  /* z down */
  memcpy(haloZHIGH,edgeZLOW,nhalodataZ*sizeof(double));
      }
  else
    {
      MPI_Irecv(haloZLOW, nhalodataZ, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), tagf, comm, &request[0]);
      MPI_Irecv(haloZHIGH, nhalodataZ, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), tagb, comm, &request[1]);
      MPI_Isend(edgeZHIGH, nhalodataZ, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), tagf, comm, &request[2]);
      MPI_Isend(edgeZLOW,  nhalodataZ, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), tagb, comm, &request[3]);
      MPI_Waitall(4, request, status);

    }

 /* put Z halos back on device, and unpack */
  cudaMemcpyAsync(haloZLOW_d, haloZLOW, nhalodataZ*sizeof(double),
		  cudaMemcpyHostToDevice,streamZ);
  cudaMemcpyAsync(haloZHIGH_d, haloZHIGH, nhalodataZ*sizeof(double),
		  cudaMemcpyHostToDevice,streamZ);

    nblocks=(Nall[X]*Nall[Y]*nhalo+DEFAULT_TPB-1)/DEFAULT_TPB;
     unpack_halo_gpu_d<<<nblocks,DEFAULT_TPB,0,streamZ>>>(nfields1,nfields2,nhalo,
							  pack_field1, N_d,data_d,haloZLOW_d,
							  haloZHIGH_d,Z);


  /* wait for all streams to complete */
  cudaStreamSynchronize(streamX);
  cudaStreamSynchronize(streamY);
  cudaStreamSynchronize(streamZ);
  

}



/* pack edges on the accelerator */
__global__ static void pack_edge_gpu_d(int nfields1, int nfields2,
				       int nhalo, int pack_field1,
					  int N[3],
					 double* edgeLOW_d,
				       double* edgeHIGH_d, 
				       double* f_d, int dirn)
{


  /* variables to determine how vel packing is done from cv array */
  int ud=-1; /* up or down */
  int pn=-1; /* positive 1 or negative 1 factor */


  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];
 
  int Nedge[3];

  if (dirn == X){
    Nedge[X]=nhalo;
    Nedge[Y]=N[Y];
    Nedge[Z]=N[Z];
  }
  else if (dirn == Y){
    Nedge[X]=Nall[X];
    Nedge[Y]=nhalo;
    Nedge[Z]=N[Z];
  }
  else if (dirn == Z){
    Nedge[X]=Nall[X];
    Nedge[Y]=Nall[Y];
    Nedge[Z]=nhalo;
  }


  int p,m, index,ii,jj,kk;
  int packed_index,packedp;
 
 
  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      if (threadIndex==0){
	//for (p=0;p<NVEL;p++)
	//for (m=0;m<3;m++)
	  //p=0;m=1;
 	  //printf("TT3 %d %d %d %d %d\n",p,m,cv_cd[p][m],cv_d[p][m], cv_ptr[p*3+m]);

      }      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);
      
      /* LOW EDGE */
      if (dirn == X){
	index = get_linear_index_gpu_d(ii+nhalo,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn == Y){
	index = get_linear_index_gpu_d(ii,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn == Z){
	index = get_linear_index_gpu_d(ii,jj,kk+nhalo,Nall);
      }
 
      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud || !pack_field1)
	  {
	    for (m = 0; m < nfields2; m++) {
	      edgeLOW_d[nfields2*npackedsite*packedp+m*npackedsite
	      	  +packed_index]
	      	= f_d[nfields2*nsite*p+nsite*m+index];
	    }
	    packedp++;
	  }
      }
      
  
      /* HIGH EDGE */
      if (dirn == X){
	index = get_linear_index_gpu_d(Nall[X]-2*nhalo+ii,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn == Y){
        index = get_linear_index_gpu_d(ii,Nall[Y]-2*nhalo+jj,kk+nhalo,Nall);
      }
      else if (dirn == Z){
	index = get_linear_index_gpu_d(ii,jj,Nall[Z]-2*nhalo+kk,Nall);
      }

      /* copy data to packed structure */
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud*pn || !pack_field1 )
	  {
	    for (m = 0; m < nfields2; m++) {
	      
	      edgeHIGH_d[nfields2*npackedsite*packedp+m*npackedsite
			   +packed_index]
		= f_d[nfields2*nsite*p+nsite*m+index];
	      
	    }
	    packedp++;
	  }
      }
    }
  
  
}



/* unpack halos on the accelerator */
__global__ static void unpack_halo_gpu_d(int nfields1, int nfields2,
					 int nhalo, int pack_field1,
					   int N[3],
					   double* f_d, double* haloLOW_d,
					 double* haloHIGH_d, int dirn)
{


  int dirn_save=dirn;

  /* variables to determine how vel packing is done from cv array */
  int ud=1; /* up or down */
  int pn=-1; /* positive 1 or negative 1 factor */

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];
 
  int Nedge[3];

  if (dirn == X){
    Nedge[X]=nhalo;
    Nedge[Y]=N[Y];
    Nedge[Z]=N[Z];
  }
  else if (dirn == Y){
    Nedge[X]=Nall[X];
    Nedge[Y]=nhalo;
    Nedge[Z]=N[Z];
  }
  else if (dirn == Z){
    Nedge[X]=Nall[X];
    Nedge[Y]=Nall[Y];
    Nedge[Z]=nhalo;
  }


  int p,m, index,ii,jj,kk;
  int packed_index, packedp;
 
  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < npackedsite)
    {
      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);

      /* LOW HALO */
      if (dirn == X){
	index = get_linear_index_gpu_d(ii,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn == Y){
	index = get_linear_index_gpu_d(ii,jj,kk+nhalo,Nall);
      }
      else if (dirn == Z){
	index = get_linear_index_gpu_d(ii,jj,kk,Nall);
      }

      if (dirn==Y || dirn==Z){
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
      }

      if (dirn==Z){
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
      }

      
      
      /* copy packed structure data to original array */
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud || !pack_field1)
	  {
	    for (m = 0; m < nfields2; m++) {
	  
	      f_d[nfields2*nsite*p+nsite*m+index] =
	      haloLOW_d[nfields2*npackedsite*packedp+m*npackedsite
	      	    +packed_index];

	    }
	    packedp++;
	  }
      }
           
      /* HIGH HALO */
      if (dirn_save == X){
	index = get_linear_index_gpu_d(Nall[X]-nhalo+ii,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn_save == Y){
	index = get_linear_index_gpu_d(ii,Nall[Y]-nhalo+jj,kk+nhalo,Nall);	
      }
      else if (dirn_save == Z){
	index = get_linear_index_gpu_d(ii,jj,Nall[Z]-nhalo+kk,Nall);
      }

      /* copy packed structure data to original array */
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud*pn || !pack_field1 )
	  {
	    for (m = 0; m < nfields2; m++) {
	      
	      f_d[nfields2*nsite*p+nsite*m+index] =
	      haloHIGH_d[nfields2*npackedsite*packedp+m*npackedsite
	           +packed_index];
	      
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

