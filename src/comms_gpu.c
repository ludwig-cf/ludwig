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


/* KEVIN */

#include "kernel.h"


typedef struct field_halo_s field_halo_t;
typedef struct field_halo_param_s field_halo_param_t;

struct field_halo_param_s {
  int nhalo;                /* coords_nhalo() */
  int nswap;                /* Width of actual halo swap <= nhalo */
  int nsite;                /* total allocated nall[X]*nall[Y]*nall[Z] */
  int nfel;                 /* Field elements per site (double) */
  int nlocal[3];            /* local domain extent */
  int nall[3];              /* ... including 2*coords_nhalo */
  int hext[3][3];           /* halo extents ... see below */
  int hsz[3];               /* halo size in lattice sites each direction */
};

struct field_halo_s {
  field_halo_param_t * param; 
  double * fxlo;
  double * fxhi;
  double * fylo;
  double * fyhi;
  double * fzlo;
  double * fzhi;
  double * hxlo;
  double * hxhi;
  double * hylo;
  double * hyhi;
  double * hzlo;
  double * hzhi;
  cudaStream_t stream[3];   /* Stream for each of X,Y,Z */
  field_halo_t * target;    /* Device memory */
};

static __constant__ field_halo_param_t const_param;

int halo_swap_gpu(field_halo_t * halo, double * f_d);

__global__ static void field_halo_pack(field_halo_t * halo, int id, double * data);
__global__ static void field_halo_unpack(field_halo_t * halo, int id, double * data);
__host__ __target__ static void field_halo_coords(field_halo_t * halo, int index,
						 int * ic, int * jc, int * kc);
__host__ __target__ static int field_halo_index(field_halo_t * halo,
						int ic, int jc, int kc);

/*****************************************************************************
 *
 *  halo_init_extra
 *
 *****************************************************************************/

__host__ int field_halo_create(int nhcomm, int nfel, field_halo_t ** phalo) {

  int sz;
  int rank;
  int nhalo;
  int ndevice;
  unsigned int mflag = cudaHostAllocDefault;

  field_halo_t * halo = NULL;

  assert(phalo);

  halo = (field_halo_t *) calloc(1, sizeof(field_halo_t));
  assert(halo);

  halo->param = (field_halo_param_t *) calloc(1, sizeof(field_halo_param_t));
  assert(halo->param);

  /* Template for distributions, which is used to allocate buffers;
   * assumed to be large enough for any halo transfer... */

  nhalo = coords_nhalo();

  halo->param->nhalo = nhalo;
  halo->param->nswap = nhcomm;
  halo->param->nfel = nfel;
  coords_nlocal(halo->param->nlocal);
  coords_nall(halo->param->nall);

  halo->param->nsite = halo->param->nall[X]*halo->param->nall[Y]*halo->param->nall[Z];

  /* Halo extents:  hext[X] = {1, nall[Y], nall[Z]}
                    hext[Y] = {nall[X], 1, nall[Z]}
                    hext[Z] = {nall[X], nall[Y], 1} */

  halo->param->hext[X][X] = halo->param->nswap;
  halo->param->hext[X][Y] = halo->param->nall[Y];
  halo->param->hext[X][Z] = halo->param->nall[Z];
  halo->param->hext[Y][X] = halo->param->nall[X];
  halo->param->hext[Y][Y] = halo->param->nswap;
  halo->param->hext[Y][Z] = halo->param->nall[Z];
  halo->param->hext[Z][X] = halo->param->nall[X];
  halo->param->hext[Z][Y] = halo->param->nall[Y];
  halo->param->hext[Z][Z] = halo->param->nswap;

  halo->param->hsz[X] = nhcomm*halo->param->hext[X][Y]*halo->param->hext[X][Z];
  halo->param->hsz[Y] = nhcomm*halo->param->hext[Y][X]*halo->param->hext[Y][Z];
  halo->param->hsz[Z] = nhcomm*halo->param->hext[Z][X]*halo->param->hext[Z][Y];

  /* Host buffers, actual and halo regions */

  sz = halo->param->hsz[X]*nfel*sizeof(double);
  cudaHostAlloc((void **) &halo->fxlo, sz, mflag);
  cudaHostAlloc((void **) &halo->fxhi, sz, mflag);
  cudaHostAlloc((void **) &halo->hxlo, sz, mflag);
  cudaHostAlloc((void **) &halo->hxhi, sz, mflag);

  sz = halo->param->hsz[Y]*nfel*sizeof(double);
  cudaHostAlloc((void **) &halo->fylo, sz, mflag);
  cudaHostAlloc((void **) &halo->fyhi, sz, mflag);
  cudaHostAlloc((void **) &halo->hylo, sz, mflag);
  cudaHostAlloc((void **) &halo->hyhi, sz, mflag);

  sz = halo->param->hsz[Z]*nfel*sizeof(double);
  cudaHostAlloc((void **) &halo->fzlo, sz, mflag);
  cudaHostAlloc((void **) &halo->fzhi, sz, mflag);
  cudaHostAlloc((void **) &halo->hzlo, sz, mflag);
  cudaHostAlloc((void **) &halo->hzhi, sz, mflag);

  /* Device buffers: allocate or alias */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    halo->target = halo;
  }
  else {
    double * tmp;

    /* Target structure */
    targetCalloc((void **) &halo->target, sizeof(field_halo_t));

    /* Buffers */
    sz = halo->param->hsz[X]*nfel*sizeof(double);

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fxlo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fxhi, &tmp, sizeof(double *));

    targetCalloc((void **) & tmp, sz);
    copyToTarget(&halo->target->hxlo, &tmp, sizeof(double *));
    targetCalloc((void **) & tmp, sz);
    copyToTarget(&halo->target->hxhi, &tmp, sizeof(double *));

    sz = halo->param->hsz[Y]*nfel*sizeof(double);

    targetCalloc((void ** ) &tmp, sz);
    copyToTarget(&halo->target->fylo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fyhi, &tmp, sizeof(double *));

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hylo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hyhi, &tmp, sizeof(double *));

    sz = halo->param->hsz[Z]*nfel*sizeof(double);

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fzlo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->fzhi, &tmp, sizeof(double *));

    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hzlo, &tmp, sizeof(double *));
    targetCalloc((void **) &tmp, sz);
    copyToTarget(&halo->target->hzhi, &tmp, sizeof(double *));

    /* Device constants */
    assert(0);
  }

  *phalo = 0;

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_free
 *
 *****************************************************************************/

__host__ int field_halo_free(field_halo_t * halo) {

  int ndevice;

  assert(halo);

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    assert(0); /* buffers */
    targetFree(halo->target);
  }

  cudaFreeHost(halo->fxlo);
  cudaFreeHost(halo->fxhi);
  cudaFreeHost(halo->fylo);
  cudaFreeHost(halo->fyhi);
  cudaFreeHost(halo->fzlo);
  cudaFreeHost(halo->fzhi);

  cudaFreeHost(halo->hxlo);
  cudaFreeHost(halo->hxhi);
  cudaFreeHost(halo->hylo);
  cudaFreeHost(halo->hyhi);
  cudaFreeHost(halo->hzlo);
  cudaFreeHost(halo->hzhi);
  free(halo->param);
  free(halo);

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_swap
 *
 *  "data" needs to be a device pointer
 *
 *****************************************************************************/

__host__ int field_halo_swap(field_halo_t * halo, double * data) {

  int ncount;
  int ndevice;
  int ic, jc, kc;
  int ih, jh, kh;
  int ixlo, ixhi;
  int iylo, iyhi;
  int izlo, izhi;  
  int nblocks;
  int m, mc, p;
  int nd, nh;
  int hsz[3];
  dim3 nblk, ntpb;

  MPI_Comm comm = cart_comm();

  MPI_Request req_x[4];
  MPI_Request req_y[4];
  MPI_Request req_z[4];
  MPI_Status  status[4];

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  assert(halo);

  targetGetDeviceCount(&ndevice);

  /* hsz[] is just shorthand for local halo sizes */
  /* An offset nd is required if nswap < nhalo */

  hsz[X] = halo->param->hsz[X];
  hsz[Y] = halo->param->hsz[Y];
  hsz[Z] = halo->param->hsz[Z];
  nh = halo->param->nhalo;
  nd = nh - halo->param->nswap;

  /* POST ALL RELEVANT Irecv() ahead of time */

  for (p = 0; p < 4; p++) {
    req_x[p] = MPI_REQUEST_NULL;
    req_y[p] = MPI_REQUEST_NULL;
    req_z[p] = MPI_REQUEST_NULL;
  }

  if (cart_size(X) > 1) {
    ncount = halo->param->hsz[X]*halo->param->nfel;
    MPI_Irecv(halo->hxlo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), ftagx, comm, req_x);
    MPI_Irecv(halo->hxhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), btagx, comm, req_x + 1);
  }

  if (cart_size(Y) > 1) {
    ncount = halo->param->hsz[Y]*halo->param->nfel;
    MPI_Irecv(halo->hylo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), ftagy, comm, req_y);
    MPI_Irecv(halo->hyhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), btagy, comm, req_y + 1);
  }

  if (cart_size(Z) > 1) {
    ncount = halo->param->hsz[Z]*halo->param->nfel;
    MPI_Irecv(halo->hzlo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), ftagz, comm, req_z);
    MPI_Irecv(halo->hzhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), btagz, comm, req_z + 1);
  }

  /* pack X edges on accelerator */
  /* pack Y edges on accelerator */
  /* pack Z edges on accelerator */

  kernel_launch_param(hsz[X], &nblk, &ntpb);
  __host_launch4s(field_halo_pack, nblk, ntpb, 0, halo->stream[X],
		  halo->target, X, data);

  if (ndevice > 0) {
    ncount = halo->param->hsz[X]*halo->param->nfel;
    cudaMemcpyAsync(halo->fxlo, halo->target->fxlo, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[X]);
    cudaMemcpyAsync(halo->fxhi, halo->target->fxhi, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[X]);
  }

  kernel_launch_param(hsz[Y], &nblk, &ntpb);
  __host_launch4s(field_halo_pack, nblk, ntpb, 0, halo->stream[Y],
		  halo->target, Y, data);

  if (ndevice > 0) {
    ncount = halo->param->hsz[Y]*halo->param->nfel;
    cudaMemcpyAsync(halo->fylo, halo->target->fylo, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Y]);
    cudaMemcpyAsync(halo->fyhi, halo->target->fyhi, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Y]);
  }

  kernel_launch_param(hsz[Z], &nblk, &ntpb);
  __host_launch4s(field_halo_pack, nblk, ntpb, 0, halo->stream[Z],
		  halo->target, Z, data);

  if (ndevice > 0) {
    ncount = halo->param->hsz[Z]*halo->param->nfel;
    cudaMemcpyAsync(halo->fzlo, halo->target->fzlo, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Z]);
    cudaMemcpyAsync(halo->fzhi, halo->fzhi, ncount*sizeof(double),
		    cudaMemcpyDeviceToHost, halo->stream[Z]);
  }


 /* Wait for X; copy or MPI recvs; put X halos back on device, and unpack */

  cudaStreamSynchronize(halo->stream[X]);
  ncount = halo->param->hsz[X]*halo->param->nfel;

  if (cart_size(X) == 1) {
    /* note these copies do not alias for ndevice == 1 */
    /* fxhi -> hxlo */
    cudaMemcpyAsync(halo->target->hxlo, halo->fxhi, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[X]);
    /* fxlo -> hxhi */
    cudaMemcpyAsync(halo->target->hxhi, halo->fxlo, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[X]);
  }
  else {
    MPI_Isend(halo->fxhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), ftagx, comm, req_x + 2);
    MPI_Isend(halo->fxlo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), btagx, comm, req_x + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_x, &mc, status);
      if (mc == 0 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hxlo, halo->hxlo, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[X]);
      }
      if (mc == 1 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hxhi, halo->hxhi, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[X]);
      }
    }
  }

  kernel_launch_param(hsz[X], &nblk, &ntpb);
  __host_launch4s(field_halo_unpack, nblk, ntpb, 0, halo->stream[X],
		  halo->target, X, data);

  /* Now wait for Y data to arrive from device */
  /* Fill in 4 corners of Y edge data from X halo */

  cudaStreamSynchronize(halo->stream[Y]);

  ih = halo->param->hext[Y][X] - nh;
  jh = halo->param->hext[X][Y] - nh - halo->param->nswap;

  for (ic = 0; ic < halo->param->nswap; ic++) {
    for (jc = 0; jc < halo->param->nswap; jc++) {
      for (kc = 0; kc < halo->param->nall[Z]; kc++) {

        ixlo = get_linear_index(     ic, nh + jc, kc, halo->param->hext[X]);
        iylo = get_linear_index(nd + ic,      jc, kc, halo->param->hext[Y]);
        ixhi = get_linear_index(ih + ic,      jc, kc, halo->param->hext[Y]);
        iyhi = get_linear_index(ic,      jh + jc, kc, halo->param->hext[X]);

        for (p = 0; p < halo->param->nfel; p++) {
          halo->fylo[hsz[Y]*p + iylo] = halo->hxlo[hsz[X]*p + ixlo];
          halo->fyhi[hsz[Y]*p + iylo] = halo->hxlo[hsz[X]*p + iyhi];
          halo->fylo[hsz[Y]*p + ixhi] = halo->hxhi[hsz[X]*p + ixlo];
          halo->fyhi[hsz[Y]*p + ixhi] = halo->hxhi[hsz[X]*p + iyhi];
        }
      }
    }
  }


  /* Swap in Y, send data back to device and unpack */

  ncount = halo->param->hsz[Y]*halo->param->nfel;

  if (cart_size(Y) == 1) {
    /* fyhi -> hylo */
    cudaMemcpyAsync(halo->target->hylo, halo->fyhi, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Y]);
    /* fylo -> hyhi */
    cudaMemcpyAsync(halo->target->hyhi, halo->fylo,ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Y]);
  }
  else {
    MPI_Isend(halo->fyhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), ftagy, comm, req_y + 2);
    MPI_Isend(halo->fylo, ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), btagy, comm, req_y + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_y, &mc, status);
      if (mc == 0 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hylo, halo->hylo, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Y]);
      }
      if (mc == 1 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hyhi, halo->hyhi, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Y]);
      }
    }
  }

  kernel_launch_param(hsz[Y], &nblk, &ntpb);
  __host_launch4s(field_halo_unpack, nblk, ntpb, 0, halo->stream[Y],
		  halo->target, Y, data);

  /* Wait for Z data from device */
  /* Fill in 4 corners of Z edge data from X halo  */

  cudaStreamSynchronize(halo->stream[Z]);

  ih = halo->param->hext[Z][X] - nh;
  kh = halo->param->hext[X][Z] - nh - halo->param->nswap;

  for (ic = 0; ic < halo->param->nswap; ic++) {
    for (jc = 0; jc < halo->param->nall[Y]; jc++) {
      for (kc = 0; kc < halo->param->nswap; kc++) {

        ixlo = get_linear_index(     ic, jc, nh + kc, halo->param->hext[X]);
        izlo = get_linear_index(nd + ic, jc,      kc, halo->param->hext[Z]);
        ixhi = get_linear_index(     ic, jc, kh + kc, halo->param->hext[X]);
        izhi = get_linear_index(ih + ic, jc,      kc, halo->param->hext[Z]);

        for (p = 0; p < halo->param->nfel; p++) {
          halo->fzlo[hsz[Z]*p + izlo] = halo->hxlo[hsz[X]*p + ixlo];
          halo->fzhi[hsz[Z]*p + izlo] = halo->hxlo[hsz[X]*p + ixhi];
          halo->fzlo[hsz[Z]*p + izhi] = halo->hxhi[hsz[X]*p + ixlo];
          halo->fzhi[hsz[Z]*p + izhi] = halo->hxhi[hsz[X]*p + ixhi];
        }
      }
    }
  }

  /* Fill in 4 strips in X of Z edge data: from Y halo  */

  jh = halo->param->hext[Z][Y] - nh;
  kh = halo->param->hext[Y][Z] - nh - halo->param->nswap;
  
  for (ic = 0; ic < halo->param->nall[X]; ic++) {
    for (jc = 0; jc < halo->param->nswap; jc++) {
      for (kc = 0; kc < halo->param->nswap; kc++) {

        iylo = get_linear_index(ic,      jc, nh + kc, halo->param->hext[Y]);
        izlo = get_linear_index(ic, nd + jc,      kc, halo->param->hext[Z]);
        iyhi = get_linear_index(ic,      jc, kh + kc, halo->param->hext[Y]);
        izhi = get_linear_index(ic, jh + jc,      kc, halo->param->hext[Z]);

        for (p = 0; p < halo->param->nfel; p++) {
          halo->fzlo[hsz[Z]*p + izlo] = halo->hylo[hsz[Y]*p + iylo];
          halo->fzhi[hsz[Z]*p + izlo] = halo->hylo[hsz[Y]*p + iyhi];
          halo->fzlo[hsz[Z]*p + izhi] = halo->hyhi[hsz[Y]*p + iylo];
          halo->fzhi[hsz[Z]*p + izhi] = halo->hyhi[hsz[Y]*p + iyhi];
        }
      }
    }
  }

  /* The z-direction swap  */

  ncount = halo->param->hsz[Z]*halo->param->nfel;

  if (cart_size(Z) == 1) {
    /* fzhi -> hzlo */
    cudaMemcpyAsync(halo->target->hzlo, halo->fzhi, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Z]);
    /* fzlo -> hzhi */
    cudaMemcpyAsync(halo->target->hzhi, halo->fzlo, ncount*sizeof(double),
		    cudaMemcpyHostToDevice, halo->stream[Z]);
  }
  else {
    MPI_Isend(halo->fzhi, ncount, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), ftagz, comm, req_z + 2);
    MPI_Isend(halo->fzlo,  ncount, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), btagz, comm, req_z + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_z, &mc, status);
      if (mc == 0 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hzlo, halo->hzlo, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Z]);
      }
    }
    if (mc == 1 && ndevice > 0) {
	cudaMemcpyAsync(halo->target->hzhi, halo->hzhi, ncount*sizeof(double),
			cudaMemcpyHostToDevice, halo->stream[Z]);
    }
  }

  kernel_launch_param(hsz[Z], &nblk, &ntpb);
  __host_launch4s(field_halo_unpack, nblk, ntpb, 0, halo->stream[Z],
		  halo->target, Z, data);

  cudaStreamSynchronize(halo->stream[X]);
  cudaStreamSynchronize(halo->stream[Y]);
  cudaStreamSynchronize(halo->stream[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  field_halo_pack
 *
 *  Move data to halo buffer on device for coordinate
 *  direction id at both low and high ends.
 *
 *****************************************************************************/

__global__
static void field_halo_pack(field_halo_t * halo, int id, double * data) {

  int kindex;

  __target_simt_parallel_for(kindex, halo->param->hsz[id], 1) {

    int nh;
    int nfel;
    int ia, indexl, indexh, ic, jc, kc;
    int hsz;
    int ho; /* high end offset */
    double * __restrict__ buflo;
    double * __restrict__ bufhi;

    nfel = halo->param->nfel;
    hsz = halo->param->hsz[id];

    /* Load two buffers for this site */
    /* Use full nhalo to address full data */

    nh = halo->param->nhalo;
    field_halo_coords(halo, kindex, &ic, &jc, &kc);

    if (id == X) {
      ho = nh + halo->param->nlocal[X] - halo->param->nswap;
      indexl = field_halo_index(halo, nh + ic, jc, kc);
      indexh = field_halo_index(halo, ho + ic, jc, kc);
      buflo = halo->fxlo;
      bufhi = halo->fxhi;
    }
    if (id == Y) {
      ho = nh + halo->param->nlocal[Y] - halo->param->nswap;
      indexl = get_linear_index_gpu_d(ic, nh + jc, kc, halo->param->nall);
      indexh = get_linear_index_gpu_d(ic, ho + jc, kc, halo->param->nall);
      buflo = halo->fylo;
      bufhi = halo->fyhi;
    }
    if (id == Z) {
      ho = nh + halo->param->nlocal[Z] - halo->param->nswap;
      indexl = get_linear_index_gpu_d(ic, jc, nh + kc, halo->param->nall);
      indexh = get_linear_index_gpu_d(ic, jc, ho + kc, halo->param->nall);
      buflo = halo->fzlo;
      bufhi = halo->fzhi;
    }

    /* Low end, and high end */

    for (ia = 0; ia < nfel; ia++) {
      buflo[hsz*ia + kindex] = data[addr_rank1(nsites, nfel, indexl, ia)];
    }

    for (ia = 0; ia < nfel; ia++) {
      bufhi[hsz*ia + kindex] = data[addr_rank1(nsites, nfel, indexh, ia)];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  field_halo_unpack
 *
 *  Unpack halo buffers to the distribution on device for direction id.
 *
 *****************************************************************************/

__global__
static void field_halo_unpack(field_halo_t * halo, int id, double * data) {

  int kindex;

  /* Unpack buffer this site. */

  __target_simt_parallel_for(kindex, halo->param->hsz[id], 1) {

    int nfel;
    int hsz;
    int ia, indexl, indexh;
    int nh;                          /* Full halo width */
    int ic, jc, kc;                  /* Lattice ooords */
    int lo, ho;                      /* Offset for low, high end */
    double * __restrict__ buflo;
    double * __restrict__ bufhi;


    nfel = halo->param->nfel;
    hsz = halo->param->hsz[id];

    nh = halo->param->nhalo;
    field_halo_coords(halo, kindex, &ic, &jc, &kc);

    if (id == X) {
      lo = nh - halo->param->nswap;
      ho = nh + halo->param->nlocal[X];
      indexl = field_halo_index(halo, lo + ic, jc, kc);
      indexh = field_halo_index(halo, ho + ic, jc, kc);
      buflo = halo->hxlo;
      bufhi = halo->hxhi;
    }

    if (id == Y) {
      lo = nh - halo->param->nswap;
      ho = nh + halo->param->nlocal[Y];
      indexl = field_halo_index(halo, ic, lo + jc, kc);
      indexh = field_halo_index(halo, ic, ho + jc, kc);
      buflo = halo->hylo;
      bufhi = halo->hyhi;
    }

    if (id == Z) {
      lo = nh - halo->param->nswap;
      ho = nh + halo->param->nlocal[Z];
      indexl = field_halo_index(halo, ic, jc, lo + kc);
      indexh = field_halo_index(halo, ic, jc, ho + kc);
      buflo = halo->hzlo;
      bufhi = halo->hzhi;
    } 

    /* Low end, then high end */

    for (ia = 0; ia < nfel; ia++) {
      data[addr_rank1(nsites, nfel, indexl, ia)] = buflo[hsz*ia + kindex];
    }

    for (ia = 0; ia < nfel; ia++) {
      data[addr_rank1(nsites, nfel, indexh, ia)] = bufhi[hsz*ia + kindex];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  field_halo_coords
 *
 *  For given kernel index, work out where we are in (ic, jc, kc)
 *
 *****************************************************************************/

__host__ __target__
static void field_halo_coords(field_halo_t * halo, int index,
			     int * ic, int * jc, int * kc) {
  int xstr;
  int ystr;

  assert(halo);

  ystr = halo->param->nall[Z];
  xstr = ystr*halo->param->nall[Y];

  *ic = index/xstr;
  *jc = (index - *ic*xstr)/ystr;
  *kc = index - *ic*xstr - *jc*ystr;

  return;
}

/*****************************************************************************
 *
 *  field_halo_index
 *
 *  A special case of coords_index().
 *
 *****************************************************************************/

__host__ __target__
static int field_halo_index(field_halo_t * halo, int ic, int jc, int kc) {

  int xstr;
  int ystr;

  assert(halo);

  ystr = halo->param->nall[Z];
  xstr = ystr*halo->param->nall[Y];

  return (ic*xstr + jc*ystr + kc);
}
