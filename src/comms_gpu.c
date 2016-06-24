/*****************************************************************************
 *
 * comms_gpu.c
 *
 * Specialised halo-exchange code for multi-GPU systems
 * These routines can provide a direct replacement for the
 * "traditional" comms routines in Ludwig
 * this source is orthogonal to the rest of Ludwig
 * and will only be used for benchmarking at the moment
 * 
 * Alan Gray
 * Kevin Stratford
 * 
 *****************************************************************************/


#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#include "pe.h"
#include "runtime.h"
#include "lb_model_s.h" 

__targetHost__ void init_comms_gpu(int N[3], int ndist);
__targetHost__ void finalise_comms_gpu();
__targetHost__ void halo_alternative(int nfields1, int nfields2, int packfield1, double * data_d);

#include "pe.h"
#include "coords.h"





__targetConst__ int N_cd[3];

/* forward declarations of host routines internal to this module */
static void calculate_comms_data_sizes(void);
static void allocate_comms_memory_on_gpu(void);
static void free_comms_memory_on_gpu(void);




/* forward declarations of accelerator routines internal to this module */


__targetEntry__ static void copy_field_partial_gpu_d(int nPerSite, int nhalo, int N[3],
						double* f_out, double* f_in, char *mask_d, int *packedindex_d, int packedsize, int inpack);

__targetHost__ __target__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);
__targetHost__ __target__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);




/* /\* external pointers to data on host*\/ */
/* extern double * f_; */
/* extern double * ftmp; */
/* extern double * colloid_force_tmp; */
/* extern double * velocity_d; */

/* /\* external pointers to data on accelerator*\/ */
/* extern int * cv_d; */
/* extern int * N_d; */
/* extern double * f_d; */
/* extern double * ftmp_d; */

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


#ifdef __NVCC__
/* handles for CUDA streams (for ovelapping)*/
static cudaStream_t streamX,streamY, streamZ, streamBULK;
#endif


static int reduced_halo=0;

/* constant memory symbols internal to this module */
__targetConst__ int cv_cd[NVEL][3];


double wtime(void) {
 
  struct timeval t1;
  gettimeofday(&t1, NULL);

  return (t1.tv_sec * 1000000 + t1.tv_usec)/1000000.;

}



/* get 3d coordinates from the index on the accelerator */

__targetHost__ __target__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,int index,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}

/* get linear index from 3d coordinates (device) */
 __targetHost__ __target__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}





/* get linear index from 3d coordinates (host) */
int get_linear_index(int ii,int jj,int kk,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}


#ifdef __NVCC__

cudaStream_t getXstream(){
  return streamX;
}

cudaStream_t getYstream(){
  return streamY;
}
cudaStream_t getZstream(){
  return streamZ;
}
cudaStream_t getBULKstream(){
  return streamBULK;
}
#endif




/* pack edges on the accelerator */
__targetEntry__ static void pack_edge_gpu_d(const int nfields1, const int nfields2,
				       const int nhalo, const int pack_field1,
					  const int N[3],
					 double* edgeLOW_d,
				       double* edgeHIGH_d, 
					    const double* __restrict__ f_d, 
					    const int dirn)
{




  int Nall[3];
  Nall[X]=N_cd[X]+2*nhalo;
  Nall[Y]=N_cd[Y]+2*nhalo;
  Nall[Z]=N_cd[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];
 
  int Nedge[3];

  if (dirn == X){
    Nedge[X]=nhalo;
    Nedge[Y]=N_cd[Y];
    Nedge[Z]=N_cd[Z];
  }
  else if (dirn == Y){
    Nedge[X]=Nall[X];
    Nedge[Y]=nhalo;
    Nedge[Z]=N_cd[Z];
  }
  else if (dirn == Z){
    Nedge[X]=Nall[X];
    Nedge[Y]=Nall[Y];
    Nedge[Z]=nhalo;
  }

 
  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];


  int threadIndex;
#ifdef __NVCC__
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  if (threadIndex < npackedsite)
#else
#pragma omp parallel for 
  for(threadIndex=0;threadIndex<npackedsite;threadIndex++)
#endif
    {

      /* variables to determine how vel packing is done from cv array */
      int ud=-1; /* up or down */
      int pn=-1; /* positive 1 or negative 1 factor */



      int p,m, index,ii,jj,kk;
      int packed_index,packedp;


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
      for (m = 0; m < nfields2; m++) {
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud || !pack_field1)
	  {
	      edgeLOW_d[nfields2*npackedsite*packedp+m*npackedsite
	      	  +packed_index]
	      	= f_d[LB_ADDR(nsite, nfields2, nfields1, index, m, p)];
	    
	    packedp++;
	    }
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

      for (m = 0; m < nfields2; m++) {
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud*pn || !pack_field1 )
	  {

	      
	      edgeHIGH_d[nfields2*npackedsite*packedp+m*npackedsite
			   +packed_index]
		= f_d[LB_ADDR(nsite, nfields2, nfields1, index, m, p)];
	      
	    packedp++;
	  }
      }

      }
    
  
  
    }
}


/* unpack halos on the accelerator */
__targetEntry__ static void unpack_halo_gpu_d(const int nfields1, const int nfields2,
					 const int nhalo, const int pack_field1,
					   const int N[3],
					   double* f_d, const double* __restrict__ haloLOW_d,
					      const double* __restrict__ haloHIGH_d, const int dirn_save)
{

 
  int Nall[3];
  Nall[X]=N_cd[X]+2*nhalo;
  Nall[Y]=N_cd[Y]+2*nhalo;
  Nall[Z]=N_cd[Z]+2*nhalo;
  int nsite = Nall[X]*Nall[Y]*Nall[Z];
 
  int Nedge[3];

  if (dirn_save == X){
    Nedge[X]=nhalo;
    Nedge[Y]=N_cd[Y];
    Nedge[Z]=N_cd[Z];
  }
  else if (dirn_save == Y){
    Nedge[X]=Nall[X];
    Nedge[Y]=nhalo;
    Nedge[Z]=N_cd[Z];
  }
  else if (dirn_save == Z){
    Nedge[X]=Nall[X];
    Nedge[Y]=Nall[Y];
    Nedge[Z]=nhalo;
  }


 
  int npackedsite = Nedge[X]*Nedge[Y]*Nedge[Z];
  

  int threadIndex;
#ifdef __NVCC__
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  if (threadIndex < npackedsite)
#else
#pragma omp parallel for 
    for(threadIndex=0;threadIndex<npackedsite;threadIndex++)
#endif
    {

  int dirn=dirn_save;



  int p,m, index,ii,jj,kk;
  int packed_index, packedp;


 /* variables to determine how vel packing is done from cv array */
  int ud=1; /* up or down */
  int pn=-1; /* positive 1 or negative 1 factor */

      
      packed_index = threadIndex;
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nedge);

      /* LOW HALO */
      if (dirn_save == X){
	index = get_linear_index_gpu_d(ii,jj+nhalo,kk+nhalo,Nall);
      }
      else if (dirn_save == Y){
	index = get_linear_index_gpu_d(ii,jj,kk+nhalo,Nall);
      }
      else if (dirn_save == Z){
	index = get_linear_index_gpu_d(ii,jj,kk,Nall);
      }


      if (dirn_save==Y || dirn_save==Z){
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

      if (dirn_save==Z){
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
      for (m = 0; m < nfields2; m++) {
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud || !pack_field1)
	  {
	  
	      f_d[LB_ADDR(nsite, nfields2, nfields1, index, m, p)] =
	      haloLOW_d[nfields2*npackedsite*packedp+m*npackedsite
	      	   +packed_index];

	    
	    packedp++;
	    }
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
      for (m = 0; m < nfields2; m++) {
      packedp=0;
      for (p = 0; p < nfields1; p++) {
	if (cv_cd[p][dirn] == ud*pn || !pack_field1 )
	  {
	    
	      

	       f_d[LB_ADDR(nsite, nfields2, nfields1, index, m, p)] =
	      haloHIGH_d[nfields2*npackedsite*packedp+m*npackedsite
	       +packed_index];
	      
	    
	    packedp++;

	  }
      }


    }
  
    }

}



void halo_alternative(int nfields1, int nfields2, int packablefield1, double * data_d)
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
#ifdef __NVCC__
 nblocks=(nhalo*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
 pack_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamX>>>(nfields1,nfields2,nhalo,
						pack_field1, N_cd,edgeXLOW_d,
						    edgeXHIGH_d,data_d,X);
#else
 pack_edge_gpu_d (nfields1,nfields2,nhalo,
		 pack_field1, N_cd,edgeXLOW_d,
		 edgeXHIGH_d,data_d,X);
#endif

 /* pack Y edges on accelerator */
#ifdef __NVCC__
  nblocks=(Nall[X]*nhalo*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
  pack_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamY>>>(nfields1,nfields2,nhalo,
						pack_field1, N_cd,edgeYLOW_d,
						     edgeYHIGH_d,data_d,Y);
#else
  pack_edge_gpu_d(nfields1,nfields2,nhalo,
		  pack_field1, N_cd,edgeYLOW_d,
		  edgeYHIGH_d,data_d,Y);
#endif

 /* pack Z edges on accelerator */
#ifdef __NVCC__
    nblocks=(Nall[X]*Nall[Y]*nhalo+DEFAULT_TPB-1)/DEFAULT_TPB;
  pack_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamZ>>>(nfields1,nfields2,nhalo,
  						pack_field1, N_cd,edgeZLOW_d,
						     edgeZHIGH_d,data_d,Z);
#else
  pack_edge_gpu_d(nfields1,nfields2,nhalo,
    pack_field1, N_cd,edgeZLOW_d,
		  edgeZHIGH_d,data_d,Z);
#endif

  nhalodataX = N[Y] * N[Z] * nhalo * nfields1packed*nfields2;
  nhalodataY = Nall[X] * N[Z] * nhalo * nfields1packed*nfields2;
  nhalodataZ = Nall[X] * Nall[Y] * nhalo * nfields1packed*nfields2;




#ifdef __NVCC__
  /* get X low edges */
  cudaMemcpyAsync(edgeXLOW, edgeXLOW_d, nhalodataX*sizeof(double),
		  cudaMemcpyDeviceToHost,streamX);
 /* get X high edges */
  cudaMemcpyAsync(edgeXHIGH, edgeXHIGH_d, nhalodataX*sizeof(double),
		  cudaMemcpyDeviceToHost,streamX);

#else
  /* get X low edges */
  //memcpy(edgeXLOW, edgeXLOW_d, nhalodataX*sizeof(double));

 
 /* get X high edges */
  //memcpy(edgeXHIGH, edgeXHIGH_d, nhalodataX*sizeof(double));
#endif


#ifndef OVERLAP

#ifdef __NVCC__
  cudaStreamSynchronize(streamX);
#endif

#endif


#ifdef __NVCC__
 /* get Y low edges */
  cudaMemcpyAsync(edgeYLOW, edgeYLOW_d, nhalodataY*sizeof(double),
		  cudaMemcpyDeviceToHost,streamY);
 /* get Y high edges */
  cudaMemcpyAsync(edgeYHIGH, edgeYHIGH_d, nhalodataY*sizeof(double),
		  cudaMemcpyDeviceToHost,streamY);
#else
 /* get Y low edges */
  //memcpy(edgeYLOW, edgeYLOW_d, nhalodataY*sizeof(double));
 /* get Y high edges */
  //memcpy(edgeYHIGH, edgeYHIGH_d, nhalodataY*sizeof(double));
#endif

#ifndef OVERLAP

#ifdef __NVCC__
  cudaStreamSynchronize(streamY);
#endif

#endif

#ifdef __NVCC__
  /* get Z low edges */
  cudaMemcpyAsync(edgeZLOW, edgeZLOW_d, nhalodataZ*sizeof(double),
		  cudaMemcpyDeviceToHost,streamZ);
  /* get Z high edges */
  cudaMemcpyAsync(edgeZHIGH, edgeZHIGH_d, nhalodataZ*sizeof(double),
		  cudaMemcpyDeviceToHost,streamZ);
#else
  /* get Z low edges */
  //memcpy(edgeZLOW, edgeZLOW_d, nhalodataZ*sizeof(double));
  /* get Z high edges */
  //memcpy(edgeZHIGH, edgeZHIGH_d, nhalodataZ*sizeof(double));
#endif



#ifndef OVERLAP
#ifdef __NVCC__
  cudaStreamSynchronize(streamZ);
#endif
#endif

 /* wait for X data from accelerator*/
#ifdef __NVCC__
  cudaStreamSynchronize(streamX);
#endif

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

#ifdef __NVCC__
 /* put X halos back on device, and unpack */
  cudaMemcpyAsync(haloXLOW_d, haloXLOW, nhalodataX*sizeof(double),
		  cudaMemcpyHostToDevice,streamX);
  cudaMemcpyAsync(haloXHIGH_d, haloXHIGH, nhalodataX*sizeof(double),
		  cudaMemcpyHostToDevice,streamX);
#else
 /* put X halos back on device, and unpack */
  //memcpy(haloXLOW_d, haloXLOW, nhalodataX*sizeof(double));
  //memcpy(haloXHIGH_d, haloXHIGH, nhalodataX*sizeof(double));
#endif


#ifdef __NVCC__
  nblocks=(nhalo*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
     unpack_halo_gpu_d<<<nblocks,DEFAULT_TPB,0,streamX>>>(nfields1,nfields2,
     							  nhalo,
     							  pack_field1, N_cd,
     							  data_d,haloXLOW_d,
     							  haloXHIGH_d,X);
#else
     unpack_halo_gpu_d(nfields1,nfields2,
		       nhalo,
		       pack_field1, N_cd,
		       data_d,haloXLOW_d,
		       haloXHIGH_d,X);
#endif



#ifndef OVERLAP
#ifdef __NVCC__
  cudaStreamSynchronize(streamX);
#endif
#endif

  /* wait for Y data from accelerator*/
#ifdef __NVCC__
  cudaStreamSynchronize(streamY);
#endif

  /* fill in corners of Y edge data  */

  for (m=0;m<(nfields1packed*nfields2);m++)
    {
      
      //#pragma omp parallel for collapse(3)
      for (ii = 0; ii < nhalo; ii++) {
	for (jj = 0; jj < nhalo; jj++) {
	  for (kk = 0; kk < N[Z]; kk++) {
	    
	    //printf("BB %1.16e\n",)
	    
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


#ifdef __NVCC__
 /* put Y halos back on device, and unpack */
  cudaMemcpyAsync(haloYLOW_d, haloYLOW, nhalodataY*sizeof(double),
		  cudaMemcpyHostToDevice,streamY);
  cudaMemcpyAsync(haloYHIGH_d, haloYHIGH, nhalodataY*sizeof(double),
		  cudaMemcpyHostToDevice,streamY);
#else
 /* put Y halos back on device, and unpack */
  //memcpy(haloYLOW_d, haloYLOW, nhalodataY*sizeof(double));
  //memcpy(haloYHIGH_d, haloYHIGH, nhalodataY*sizeof(double));
#endif



#ifdef __NVCC__
  nblocks=(Nall[X]*nhalo*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
    unpack_halo_gpu_d<<<nblocks,DEFAULT_TPB,0,streamY>>>(nfields1,nfields2,nhalo,
  						  pack_field1, N_cd,data_d,haloYLOW_d,
  							 haloYHIGH_d,Y);
#else
    unpack_halo_gpu_d(nfields1,nfields2,nhalo,
    pack_field1, N_cd,data_d,haloYLOW_d,
    		      haloYHIGH_d,Y);
#endif


#ifndef OVERLAP
#ifdef __NVCC__
  cudaStreamSynchronize(streamY);
#endif
#endif

 
  /* wait for Z data from accelerator*/
#ifdef __NVCC__
  cudaStreamSynchronize(streamZ);
#endif

  /* fill in corners of Z edge data: from Xhalo  */

    
  for (m=0;m<(nfields1packed*nfields2);m++)
    {
      
      //#pragma omp parallel for collapse(3)
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
      
      
      //#pragma omp parallel for collapse(3)
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

#ifdef __NVCC__
 /* put Z halos back on device and unpack*/
  cudaMemcpyAsync(haloZLOW_d, haloZLOW, nhalodataZ*sizeof(double),
		  cudaMemcpyHostToDevice,streamZ);
  cudaMemcpyAsync(haloZHIGH_d, haloZHIGH, nhalodataZ*sizeof(double),
		  cudaMemcpyHostToDevice,streamZ);
#else
 /* put Z halos back on device and unpack*/
  //memcpy(haloZLOW_d, haloZLOW, nhalodataZ*sizeof(double));
  //memcpy(haloZHIGH_d, haloZHIGH, nhalodataZ*sizeof(double));
#endif


#ifdef __NVCC__
    nblocks=(Nall[X]*Nall[Y]*nhalo+DEFAULT_TPB-1)/DEFAULT_TPB;
     unpack_halo_gpu_d<<<nblocks,DEFAULT_TPB,0,streamZ>>>(nfields1,nfields2,nhalo,
							  pack_field1, N_cd,data_d,haloZLOW_d,
							  haloZHIGH_d,Z);

#else
     unpack_halo_gpu_d(nfields1,nfields2,nhalo,
     			 pack_field1, N_cd,data_d,haloZLOW_d,
     		       haloZHIGH_d,Z);
#endif

  /* wait for all streams to complete */
#ifdef __NVCC__
  cudaStreamSynchronize(streamX);
  cudaStreamSynchronize(streamY);
  cudaStreamSynchronize(streamZ);
#endif

  return;
}





/* Allocate memory on accelerator */
static void allocate_comms_memory_on_gpu()
{

#ifdef __NVCC__
  
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


#else


  packedindex= (int*) calloc( nsites,sizeof(int)); 
  mask_=(char*) calloc( nsites,sizeof(char));
  mask_with_neighbours= (char*) calloc(nsites,sizeof(char)); 
  edgeXLOW=(double*) calloc( nhalodataX,sizeof(double));
  edgeXHIGH=(double*) calloc( nhalodataX,sizeof(double)); 
  edgeYLOW=(double*) calloc(  nhalodataY,sizeof(double));
  edgeYHIGH=(double*) calloc(  nhalodataY,sizeof(double)); 
  edgeZLOW=(double*) calloc(  nhalodataZ,sizeof(double)); 
  edgeZHIGH=(double*) calloc( nhalodataZ,sizeof(double)); 
  haloXLOW=(double*) calloc( nhalodataX,sizeof(double));
  haloXHIGH=(double*) calloc(  nhalodataX,sizeof(double)); 
  haloYLOW=(double*) calloc( nhalodataY,sizeof(double)); 
  haloYHIGH=(double*) calloc( nhalodataY,sizeof(double)); 
  haloZLOW=(double*) calloc(  nhalodataZ,sizeof(double)); 
  haloZHIGH=(double*) calloc(  nhalodataZ,sizeof(double)); 


#endif


  
#ifdef __NVCC__
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

#else

  /* edgeXLOW_d=(double*) calloc(nhalodataX,sizeof(double)); */
  /* edgeXHIGH_d=(double*) calloc(nhalodataX,sizeof(double)); */
  /* edgeYLOW_d=(double*) calloc(nhalodataY,sizeof(double)); */
  /* edgeYHIGH_d=(double*) calloc(nhalodataY,sizeof(double)); */
  /* edgeZLOW_d=(double*) calloc(nhalodataZ,sizeof(double)); */
  /* edgeZHIGH_d=(double*) calloc(nhalodataZ,sizeof(double)); */
  
  /* haloXLOW_d=(double*) calloc(nhalodataX,sizeof(double)); */
  /* haloXHIGH_d=(double*) calloc(nhalodataX,sizeof(double)); */
  /* haloYLOW_d=(double*) calloc(nhalodataY,sizeof(double)); */
  /* haloYHIGH_d=(double*) calloc(nhalodataY,sizeof(double)); */
  /* haloZLOW_d=(double*) calloc(nhalodataZ,sizeof(double)); */
  /* haloZHIGH_d=(double*) calloc(nhalodataZ,sizeof(double)); */

  edgeXLOW_d=edgeXLOW;
  edgeXHIGH_d=edgeXHIGH;
  edgeYLOW_d=edgeYLOW;
  edgeYHIGH_d=edgeYHIGH;
  edgeZLOW_d=edgeZLOW;
  edgeZHIGH_d=edgeZHIGH;
  
  haloXLOW_d=haloXLOW;
  haloXHIGH_d=haloXHIGH;
  haloYLOW_d=haloYLOW;
  haloYHIGH_d=haloYHIGH;
  haloZLOW_d=haloZLOW;
  haloZHIGH_d=haloZHIGH;

  mask_d=(char*) calloc(nsites,sizeof(char));
  packedindex_d=(int*) calloc(nsites,sizeof(int));

#endif

  return;
}


/* Free memory on accelerator */
static void free_comms_memory_on_gpu()
{


#ifdef __NVCC__
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

#else

  free(packedindex);
  free(mask_);
  free(mask_with_neighbours);

  /* free(edgeXLOW); */
  /* free(edgeXHIGH); */
  /* free(edgeYLOW); */
  /* free(edgeYHIGH); */
  /* free(edgeZLOW); */
  /* free(edgeZHIGH); */

  /* free(haloXLOW); */
  /* free(haloXHIGH); */
  /* free(haloYLOW); */
  /* free(haloYHIGH); */
  /* free(haloZLOW); */
  /* free(haloZHIGH); */


  /* free memory on accelerator */

  free(mask_d);
  free(packedindex_d);

  free(edgeXLOW_d);
  free(edgeXHIGH_d);
  free(edgeYLOW_d);
  free(edgeYHIGH_d);
  free(edgeZLOW_d);
  free(edgeZHIGH_d);

  free(haloXLOW_d);
  free(haloXHIGH_d);
  free(haloYLOW_d);
  free(haloYHIGH_d);
  free(haloZLOW_d);
  free(haloZHIGH_d);


#endif

}




/* Perform tasks necessary to initialise accelerator */
void init_comms_gpu(int Nin[3], int ndistin)
{

  ndist=ndistin;

  N[X]=Nin[X];N[Y]=Nin[Y];N[Z]=Nin[Z];
  calculate_comms_data_sizes();
  allocate_comms_memory_on_gpu();


  char string[FILENAME_MAX];

  RUN_get_string_parameter("reduced_halo", string, FILENAME_MAX);
  if (strcmp(string, "yes") == 0) reduced_halo = 1;
  
  /* create CUDA streams (for ovelapping)*/
#ifdef __NVCC__
  cudaStreamCreate(&streamX);
  cudaStreamCreate(&streamY);
  cudaStreamCreate(&streamZ);
  cudaStreamCreate(&streamBULK);
#endif

  copyConstToTarget(cv_cd, cv, NVEL*3*sizeof(int)); 
  copyConstToTarget(N_cd, N, 3*sizeof(int));


}

void finalise_comms_gpu()
{
  free_comms_memory_on_gpu();

#ifdef __NVCC__
  cudaStreamDestroy(streamX);
  cudaStreamDestroy(streamY);
  cudaStreamDestroy(streamZ);
  cudaStreamDestroy(streamBULK);
#endif
}


/* calculate sizes of data - needed for memory copies to accelerator */
static void calculate_comms_data_sizes()
{
  coords_nlocal(N);  
  nhalo = coords_nhalo();  

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

  if (n1 < 5) n1=5;

  nhalodataX = N[Y] * N[Z] * nhalo * n1;
  nhalodataY = Nall[X] * N[Z] * nhalo * n1;
  nhalodataZ = Nall[X] * Nall[Y] * nhalo * n1;

  return;
}
