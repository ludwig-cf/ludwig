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


#include "model.h" 
#include <time.h>
#include <sys/time.h>
#include "targetDP.h"



#ifdef LB_DATA_SOA


__targetHost__ int  RUN_get_string_parameter(const char *, char *, const int);


__targetHost__ void init_comms_gpu(int N[3], int ndist);
__targetHost__ void finalise_comms_gpu();
__targetHost__ void halo_SoA(int nfields1, int nfields2, int packfield1, double * data_d);

#ifdef KEVIN_GPU
int halo_init_extra(void);
#endif

#include "pe.h"
#include "coords.h"





__targetConst__ int N_cd[3];

/* forward declarations of host routines internal to this module */
static void calculate_comms_data_sizes(void);
static void allocate_comms_memory_on_gpu(void);
static void free_comms_memory_on_gpu(void);




/* forward declarations of accelerator routines internal to this module */
__targetEntry__ static void pack_edge_gpu_d(int nfields1, int nfields2,
				       int nhalo, int nreduced,
					 int N[3],
					 double* fedgeLOW_d,
				       double* fedgeHIGH_d, 
					    double* f_d, int dirn);

__targetEntry__ static void unpack_halo_gpu_d(int nfields1, int nfields2,
					 int nhalo, int nreduced,
					   int N[3],
					   double* f_d, double* fhaloLOW_d,
					      double* fhaloHIGH_d, int dirn);


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


#ifdef CUDA
/* handles for CUDA streams (for ovelapping)*/
static cudaStream_t streamX,streamY, streamZ, streamBULK;
#endif


static int reduced_halo=0;

/* constant memory symbols internal to this module */
__targetConst__ int cv_cd[NVEL][3];
//__constant__ int N_cd[3];

/* void getXYZstreamptr(void* Xptr,void* Yptr,void* Zptr){ */
/*   //void getXstreamptr(void* ptr){   */
/*   *Xptr=streamX; Yptr=streamY;Zptr=streamZ; */
/*   return ; */
/* } */


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


#ifdef CUDA

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
__targetEntry__ static void pack_edge_gpu_d(int nfields1, int nfields2,
				       int nhalo, int pack_field1,
					  int N[3],
					 double* edgeLOW_d,
				       double* edgeHIGH_d, 
					    double* f_d, int dirn)
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
#ifdef CUDA
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
__targetEntry__ static void unpack_halo_gpu_d(int nfields1, int nfields2,
					 int nhalo, int pack_field1,
					   int N[3],
					   double* f_d, double* haloLOW_d,
					      double* haloHIGH_d, int dirn_save)
{


  //if (dirn==X) return;
  //if (dirn==Y) return;
  //if (dirn==Z) return;


 
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
#ifdef CUDA
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



void halo_SoA(int nfields1, int nfields2, int packablefield1, double * data_d)
{



  
  int pack_field1=packablefield1*reduced_halo;
  int nfields1packed;

  double t1, t2;

  //t1=wtime();
  
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


  //t2=wtime();
  
  //t1=wtime();

  int NedgeX[3], NedgeY[3], NedgeZ[3];

  int ii,jj,kk,m,index_source,index_target;

  int nblocks;

  //#define OVERLAP


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
#ifdef CUDA
 nblocks=(nhalo*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
 pack_edge_gpu_d<<<nblocks,DEFAULT_TPB,0,streamX>>>(nfields1,nfields2,nhalo,
						pack_field1, N_cd,edgeXLOW_d,
						    edgeXHIGH_d,data_d,X);
#else
 pack_edge_gpu_d (nfields1,nfields2,nhalo,
		 pack_field1, N_cd,edgeXLOW_d,
		 edgeXHIGH_d,data_d,X);
#endif

  //t2=wtime();
  
  //printf("A1 %1.16e\n",t2-t1);

  //t1=wtime();


 /* pack Y edges on accelerator */
#ifdef CUDA
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
#ifdef CUDA
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




#ifdef CUDA
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

#ifdef CUDA
  cudaStreamSynchronize(streamX);
#endif

#endif

  //t2=wtime();
  
  //printf("A2 %1.16e\n",t2-t1);

  //t1=wtime();

#ifdef CUDA
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

#ifdef CUDA
  cudaStreamSynchronize(streamY);
#endif

#endif

  //t2=wtime();
  
  //printf("A3 %1.16e\n",t2-t1);

  //t1=wtime();

#ifdef CUDA
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
#ifdef CUDA
  cudaStreamSynchronize(streamZ);
#endif
#endif

  //t2=wtime();
  
  //printf("A4 %1.16e\n",t2-t1);

  //t1=wtime();



 /* wait for X data from accelerator*/
#ifdef CUDA
  cudaStreamSynchronize(streamX);
#endif

  //t2=wtime();
  
  //printf("B %1.16e\n",t2-t1);

  //t1=wtime();

  //HACK
  //if (nfields1==19) 
  //launch_bulk_calc_gpu();

  // collide_bulk_gpu(1);




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


  //t2=wtime();
  
  //printf("C %1.16e\n",t2-t1);

  //t1=wtime();


#ifdef CUDA
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


#ifdef CUDA
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
#ifdef CUDA
  cudaStreamSynchronize(streamX);
#endif
#endif

  /* wait for Y data from accelerator*/
#ifdef CUDA
  cudaStreamSynchronize(streamY);
#endif

  //t2=wtime();
  
  //printf("D %1.16e\n",t2-t1);

  //t1=wtime();

  /* fill in corners of Y edge data  */

  for (m=0;m<(nfields1packed*nfields2);m++)
    {
      
      
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
  


  //t2=wtime();
  
  //printf("E %1.16e\n",t2-t1);

  //t1=wtime();


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


  //t2=wtime();
  
  //printf("F %1.16e\n",t2-t1);

  //t1=wtime();


#ifdef CUDA
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



#ifdef CUDA
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
#ifdef CUDA
  cudaStreamSynchronize(streamY);
#endif
#endif

 
  /* wait for Z data from accelerator*/
#ifdef CUDA
  cudaStreamSynchronize(streamZ);
#endif

  /* fill in corners of Z edge data: from Xhalo  */


  //t2=wtime();
  
  //printf("H %1.16e\n",t2-t1);

  //t1=wtime();
    
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
  
  

  //t2=wtime();
  
  //printf("I %1.16e\n",t2-t1);

  //t1=wtime();

  
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
  


  //t2=wtime();
  
  //printf("J %1.16e\n",t2-t1);

  //t1=wtime();


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

  //t2=wtime();
  
  //printf("J1 %1.16e\n",t2-t1);

  //t1=wtime();


#ifdef CUDA
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


#ifdef CUDA
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
#ifdef CUDA
  cudaStreamSynchronize(streamX);
  cudaStreamSynchronize(streamY);
  cudaStreamSynchronize(streamZ);
#endif

  //t2=wtime();
  
  //printf("L %1.16e\n",t2-t1);

  //t1=wtime();
  

}





/* Allocate memory on accelerator */
static void allocate_comms_memory_on_gpu()
{

#ifdef CUDA
  
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


  
#ifdef CUDA  
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



  //   checkCUDAError("allocate_memory_on_gpu");

}


/* Free memory on accelerator */
static void free_comms_memory_on_gpu()
{


#ifdef CUDA
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
#ifdef CUDA
  cudaStreamCreate(&streamX);
  cudaStreamCreate(&streamY);
  cudaStreamCreate(&streamZ);
  cudaStreamCreate(&streamBULK);
#endif

  //  cudaMemcpyToSymbol(cv_cd, cv, NVEL*3*sizeof(int), 0, cudaMemcpyHostToDevice); 
  //cudaMemcpyToSymbol(N_cd, N, 3*sizeof(int), 0, cudaMemcpyHostToDevice);  
  //checkCUDAError("Init GPU");  


  copyConstToTarget(cv_cd, cv, NVEL*3*sizeof(int)); 
  copyConstToTarget(N_cd, N, 3*sizeof(int));
  
#ifdef KEVIN_GPU
  halo_init_extra();
#endif


}

void finalise_comms_gpu()
{
  free_comms_memory_on_gpu();

#ifdef CUDA
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
  //  ndist = distribution_ndist();
  //ndist = lb_ndist();
  //nop = phi_nop();

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
  //if (nop > n1) n1=nop;
  
  //HACK
  //make sure enough space for 5-component order parameter
  
  if (n1 < 5) n1=5;

  nhalodataX = N[Y] * N[Z] * nhalo * n1;
  nhalodataY = Nall[X] * N[Z] * nhalo * n1;
  nhalodataZ = Nall[X] * Nall[Y] * nhalo * n1;

  //  printf("KKK %d %d\n",nhalodataX, n1);


}




#ifdef CUDA





/* void fill_mask_with_neighbours(char *mask) */
/* { */

/*   int i, ib[3], p; */

/*   for (i=0; i<nsites; i++) */
/*     mask_with_neighbours[i]=0; */


/*   for (i=0; i<nsites; i++){ */
/*     if(mask[i]){ */
/*       mask_with_neighbours[i]=1; */
/*       coords_index_to_ijk(i, ib); */
/*       /\* if not a halo *\/ */
/*       int halo = (ib[X] < 1 || ib[Y] < 1 || ib[Z] < 1 || */
/* 		  ib[X] > N[X] || ib[Y] > N[Y] || ib[Z] > N[Z]); */
      
/*       if (!halo){ */
	
/* 	for (p=1; p<NVEL; p++){ */
/* 	  int indexn = coords_index(ib[X] + cv[p][X], ib[Y] + cv[p][Y], */
/* 				    ib[Z] + cv[p][Z]); */
/* 	  mask_with_neighbours[indexn]=1; */
/* 	} */
/*       } */
/*     } */
    
/*   } */
  
  

/* } */




/* void put_field_partial_on_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *)){ */

/*   char *mask; */
/*   int i; */
/*   int index; */
/*   double field_tmp[50]; */
  
/*   if(include_neighbours){ */
/*     fill_mask_with_neighbours(mask_); */
/*     mask=mask_with_neighbours; */
/*   } */
/*   else{ */
/*     mask=mask_; */
/*   } */



/*   int packedsize=0; */
/*   for (index=0; index<nsites; index++){ */
/*     if(mask[index]) packedsize++; */
/*   } */


/*   int j=0; */
/*   for (index=0; index<nsites; index++){ */
    
/*     if(mask[index]){ */
 
/*       access_function(index,field_tmp); */
      
/*       for (i=0;i<(nfields1*nfields2);i++) */
/* 	{ */
/* 	  ftmp[i*packedsize+j]=field_tmp[i]; */
/* 	} */
      
/*       packedindex[index]=j; */
/*       j++; */

/*     } */

/*   } */

/*   cudaMemcpy(ftmp_d, ftmp, packedsize*nfields1*nfields2*sizeof(double), cudaMemcpyHostToDevice); */
/*   cudaMemcpy(mask_d, mask, nsites*sizeof(char), cudaMemcpyHostToDevice); */
/*   cudaMemcpy(packedindex_d, packedindex, nsites*sizeof(int), cudaMemcpyHostToDevice); */

/*   /\* run the GPU kernel *\/ */

/*   int nblocks=(Nall[X]*Nall[Y]*Nall[Z]+DEFAULT_TPB-1)/DEFAULT_TPB; */
/*   copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(nfields1*nfields2, nhalo, N_d, */
/*   						data_d, ftmp_d, mask_d, */
/*   						packedindex_d, packedsize, 1); */
/*   cudaThreadSynchronize(); */
/*   checkCUDAError("put_partial_field_on_gpu"); */

/* } */


/* /\* copy part of velocity_ from accelerator to host, using mask structure *\/ */
/* void get_field_partial_from_gpu(int nfields1, int nfields2, int include_neighbours,double *data_d, void (* access_function)(const int, double *)) */
/* { */


/*   char *mask; */
/*   int i; */
/*   int index; */
/*   double field_tmp[50]; */

/*   if(include_neighbours){ */
/*     fill_mask_with_neighbours(mask_); */
/*     mask=mask_with_neighbours; */
/*   } */
/*   else{ */
/*     mask=mask_; */
/*   } */

/*   int j=0; */
/*   for (i=0; i<nsites; i++){ */
/*     if(mask[i]){ */
/*       packedindex[i]=j; */
/*       j++; */
/*     } */
    
/*   } */

/*   int packedsize=j; */

/*   cudaMemcpy(mask_d, mask, nsites*sizeof(char), cudaMemcpyHostToDevice); */
/*   cudaMemcpy(packedindex_d, packedindex, nsites*sizeof(int), cudaMemcpyHostToDevice); */

/*   int nblocks=(Nall[X]*Nall[Y]*Nall[Z]+DEFAULT_TPB-1)/DEFAULT_TPB; */
/*  copy_field_partial_gpu_d<<<nblocks,DEFAULT_TPB>>>(nfields1*nfields2, nhalo, N_d, */
/*   						ftmp_d, data_d, mask_d, */
/*   						packedindex_d, packedsize, 0); */
/*   cudaThreadSynchronize(); */

/*   cudaMemcpy(ftmp, ftmp_d, packedsize*nfields1*nfields2*sizeof(double), cudaMemcpyDeviceToHost);  */

/*   j=0; */
/*   for (index=0; index<nsites; index++){ */
    
/*     if(mask[index]){ */
 
/*       for (i=0;i<nfields1*nfields2;i++) */
/* 	{ */
/* 	  field_tmp[i]=ftmp[i*packedsize+j]; */
/* 	} */
/*       access_function(index,field_tmp);        */
/*       j++; */

/*     } */

/*   } */



/*   /\* run the GPU kernel *\/ */

/*   checkCUDAError("get_field_partial_from_gpu"); */

/* } */


/* __global__ static void copy_field_partial_gpu_d(int nPerSite, int nhalo, int N[3], */
/* 					    double* f_out, double* f_in, char *mask_d, int *packedindex_d, int packedsize, int inpack) { */

/*   int threadIndex, nsite, Nall[3]; */
/*   int i; */


/*   Nall[X]=N[X]+2*nhalo; */
/*   Nall[Y]=N[Y]+2*nhalo; */
/*   Nall[Z]=N[Z]+2*nhalo; */

/*   nsite = Nall[X]*Nall[Y]*Nall[Z]; */


/*   /\* CUDA thread index *\/ */
/*   threadIndex = blockIdx.x*blockDim.x+threadIdx.x; */

/*   //Avoid going beyond problem domain */
/*   if ((threadIndex < Nall[X]*Nall[Y]*Nall[Z]) && mask_d[threadIndex]) */
/*     { */

/*       for (i=0;i<nPerSite;i++) */
/* 	{ */
	    
/* 	  if (inpack) */
/* 	    f_out[i*nsite+threadIndex] */
/* 	    =f_in[i*packedsize+packedindex_d[threadIndex]]; */
/* 	  else */
/* 	   f_out[i*packedsize+packedindex_d[threadIndex]] */
/* 	      =f_in[i*nsite+threadIndex]; */
	  
/* 	} */
/*     } */


/*   return; */
/* } */

#ifdef KEVIN_GPU

typedef struct cuda_halo_s cuda_halo_t;

struct cuda_halo_s {
  int nhalo;                /* coords_nhalo() */
  int nswap;                /* Width of actual halo swap <= nhalo */
  int nsite;                /* total nsites[X]*nsites[Y]*nsites[Z] */
  int nfel;                 /* Field elements per site (double) */
  int nlocal[3];            /* local domain extent */
  int nsites[3];            /* ... including 2*coords_nhalo */
  int hext[3][3];           /* halo extents ... see below */
  int hsz[3];               /* halo size in lattice sites each direction */
  cuda_halo_t * d;          /* Device pointer for this host struct */
};

/* static cuda_halo_t host;*/
static cuda_halo_t * fhalo;
static cuda_halo_t * uhalo;
static cuda_halo_t * qhalo;

/* Edge and halo buffers */

static double * fxlo, * hxlo, * fxlo_d, * hxlo_d;
static double * fxhi, * hxhi, * fxhi_d, * hxhi_d;
static double * fylo, * hylo, * fylo_d, * hylo_d;
static double * fyhi, * hyhi, * fyhi_d, * hyhi_d;
static double * fzlo, * hzlo, * fzlo_d, * hzlo_d;
static double * fzhi, * hzhi, * fzhi_d, * hzhi_d;

int halo_swap_gpu(cuda_halo_t * halo, double * f_d);

__global__
static void halo_pack_gpu_d(cuda_halo_t * halo, int id,
                              double * flo_d,
                              double * fhi_d,
		              double * f_d);

__global__
static void halo_unpack_gpu_d(cuda_halo_t * halo, int id,
		              double * f_d,
                              double * hlo_d,
                              double * hhi_d);

/*****************************************************************************
 *
 *  halo_init_extra
 *
 *****************************************************************************/

int halo_init_extra(void) {

  int rank;
  unsigned int mflag = cudaHostAllocDefault;

  cuda_halo_t tmp;

  /* Template for distributions, which is used to allocate buffers;
   * assumed to be large enough for any halo transfer... */

  tmp.nhalo = coords_nhalo();
  tmp.nswap = 1;
  tmp.nfel = NVEL;
  coords_nlocal(tmp.nlocal);
  coords_nsite_local(tmp.nsites);
  tmp.nsite = tmp.nsites[X]*tmp.nsites[Y]*tmp.nsites[Z];

  /* Halo extents:  hext[X] = {1, nsites[Y], nsites[Z]}
                    hext[Y] = {nsites[X], 1, nsites[Z]}
                    hext[Z] = {nsites[X], nsites[Y], 1} */

  tmp.hext[X][X] = tmp.nswap;
  tmp.hext[X][Y] = tmp.nsites[Y];
  tmp.hext[X][Z] = tmp.nsites[Z];
  tmp.hext[Y][X] = tmp.nsites[X];
  tmp.hext[Y][Y] = tmp.nswap;
  tmp.hext[Y][Z] = tmp.nsites[Z];
  tmp.hext[Z][X] = tmp.nsites[X];
  tmp.hext[Z][Y] = tmp.nsites[Y];
  tmp.hext[Z][Z] = tmp.nswap;

  tmp.hsz[X] = tmp.nswap*tmp.hext[X][Y]*tmp.hext[X][Z];
  tmp.hsz[Y] = tmp.nswap*tmp.hext[Y][X]*tmp.hext[Y][Z];
  tmp.hsz[Z] = tmp.nswap*tmp.hext[Z][X]*tmp.hext[Z][Y];

  fhalo = (cuda_halo_t *) calloc(1, sizeof(cuda_halo_t));
  *fhalo = tmp;

  cudaMalloc((void **) &fhalo->d, sizeof(cuda_halo_t));
  cudaMemcpy(fhalo->d, fhalo, sizeof(cuda_halo_t), cudaMemcpyHostToDevice);

  /* Velocities */

  uhalo = (cuda_halo_t *) calloc(1, sizeof(cuda_halo_t));
  *uhalo = tmp;
  uhalo->nfel = 3;

  cudaMalloc((void **) &uhalo->d, sizeof(cuda_halo_t));
  cudaMemcpy(uhalo->d, uhalo, sizeof(cuda_halo_t), cudaMemcpyHostToDevice);

  /* Order parameter (here q) */

  qhalo = (cuda_halo_t *) calloc(1, sizeof(cuda_halo_t));
  *qhalo = tmp;
  qhalo->nswap = 2;
  qhalo->nfel = 5;

  qhalo->hext[X][X] = qhalo->nswap;
  qhalo->hext[Y][Y] = qhalo->nswap;
  qhalo->hext[Z][Z] = qhalo->nswap;

  qhalo->hsz[X] = qhalo->nswap*qhalo->hext[X][Y]*qhalo->hext[X][Z];
  qhalo->hsz[Y] = qhalo->nswap*qhalo->hext[Y][X]*qhalo->hext[Y][Z];
  qhalo->hsz[Z] = qhalo->nswap*qhalo->hext[Z][X]*qhalo->hext[Z][Y];

  cudaMalloc((void **) &qhalo->d, sizeof(cuda_halo_t));
  cudaMemcpy(qhalo->d, qhalo, sizeof(cuda_halo_t), cudaMemcpyHostToDevice);

  /* Host buffers, actual and halo regions */

  cudaHostAlloc((void **) &fxlo, tmp.hsz[X]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &fxhi, tmp.hsz[X]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &fylo, tmp.hsz[Y]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &fyhi, tmp.hsz[Y]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &fzlo, tmp.hsz[Z]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &fzhi, tmp.hsz[Z]*NVEL*sizeof(double), mflag);

  cudaHostAlloc((void **) &hxlo, tmp.hsz[X]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &hxhi, tmp.hsz[X]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &hylo, tmp.hsz[Y]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &hyhi, tmp.hsz[Y]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &hzlo, tmp.hsz[Z]*NVEL*sizeof(double), mflag);
  cudaHostAlloc((void **) &hzhi, tmp.hsz[Z]*NVEL*sizeof(double), mflag);

  /* Device buffers */

  cudaMalloc((void **) &fxlo_d, tmp.hsz[X]*NVEL*sizeof(double));
  cudaMalloc((void **) &fxhi_d, tmp.hsz[X]*NVEL*sizeof(double));
  cudaMalloc((void **) &fylo_d, tmp.hsz[Y]*NVEL*sizeof(double));
  cudaMalloc((void **) &fyhi_d, tmp.hsz[Y]*NVEL*sizeof(double));
  cudaMalloc((void **) &fzlo_d, tmp.hsz[Z]*NVEL*sizeof(double));
  cudaMalloc((void **) &fzhi_d, tmp.hsz[Z]*NVEL*sizeof(double));

  cudaMalloc((void **) &hxlo_d, tmp.hsz[X]*NVEL*sizeof(double));
  cudaMalloc((void **) &hxhi_d, tmp.hsz[X]*NVEL*sizeof(double));
  cudaMalloc((void **) &hylo_d, tmp.hsz[Y]*NVEL*sizeof(double));
  cudaMalloc((void **) &hyhi_d, tmp.hsz[Y]*NVEL*sizeof(double));
  cudaMalloc((void **) &hzlo_d, tmp.hsz[Z]*NVEL*sizeof(double));
  cudaMalloc((void **) &hzhi_d, tmp.hsz[Z]*NVEL*sizeof(double));

  checkCUDAError("halo_init_extra malloc() failed\n");

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("THREADS PER BLOCK %2d %2d %2d\n", DEFAULT_TPB_X, DEFAULT_TPB_Y,
           DEFAULT_TPB_Z);
  }

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_gpu
 *
 *****************************************************************************/

int dist_halo_gpu(double * f_d) {

  halo_swap_gpu(fhalo, f_d);

  return 0;
}

int u_halo_gpu(double * u_d) {

  halo_swap_gpu(uhalo, u_d);

  return 0;
}

int q_halo_gpu(double * phi) {

  halo_swap_gpu(qhalo, phi);

  return 0;
}

/*****************************************************************************
 *
 *  halo_swap_gpu
 *
 *  For halo type described by halo, with corresponding data f_d.
 *
 *   Note: I've tried to squeeze fixed overheads to a minimum;
 *   the sends and receives in each direction are finished in
 *   one go, as the sends are probably progressing the message
 *   (there appears to be no benefit defering the wait on the
 *   sends).
 *
 *****************************************************************************/

int halo_swap_gpu(cuda_halo_t * halo, double * f_d) {

  int ic, jc, kc;
  int ih, jh, kh;
  int ixlo, ixhi;
  int iylo, iyhi;
  int izlo, izhi;  
  int nblocks;
  int m, mc, p;
  int nd, nh;
  int hsz[3];

  MPI_Comm comm = cart_comm();

  MPI_Request req_x[4];
  MPI_Request req_y[4];
  MPI_Request req_z[4];
  MPI_Status  status[4];

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  /* This is just shorthand for local halo sizes */
  /* An offset nd is required if nswap < nhalo */

  hsz[X] = halo->hsz[X];
  hsz[Y] = halo->hsz[Y];
  hsz[Z] = halo->hsz[Z];
  nh = halo->nhalo;
  nd = nh - halo->nswap;

  /* POST ALL RELEVANT Irecv() ahead of time */

  for (p = 0; p < 4; p++) {
    req_x[p] = MPI_REQUEST_NULL;
    req_y[p] = MPI_REQUEST_NULL;
    req_z[p] = MPI_REQUEST_NULL;
  }

  if (cart_size(X) > 1) {
    MPI_Irecv(hxlo, hsz[X]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), ftagx, comm, req_x);
    MPI_Irecv(hxhi, hsz[X]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), btagx, comm, req_x + 1);
  }

  if (cart_size(Y) > 1) {
    MPI_Irecv(hylo, hsz[Y]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), ftagy, comm, req_y);
    MPI_Irecv(hyhi, hsz[Y]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), btagy, comm, req_y + 1);
  }

  if (cart_size(Z) > 1) {
    MPI_Irecv(hzlo, hsz[Z]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), ftagz, comm, req_z);
    MPI_Irecv(hzhi, hsz[Z]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), btagz, comm, req_z + 1);
  }

  /* pack X edges on accelerator */
  /* pack Y edges on accelerator */
  /* pack Z edges on accelerator */

  nblocks = (hsz[X] + DEFAULT_TPB - 1) / DEFAULT_TPB;
  halo_pack_gpu_d<<<nblocks, DEFAULT_TPB, 0, streamX>>>
	(halo->d, X, fxlo_d, fxhi_d, f_d);

  cudaMemcpyAsync(fxlo, fxlo_d, hsz[X]*halo->nfel*sizeof(double),
		  cudaMemcpyDeviceToHost, streamX);
  cudaMemcpyAsync(fxhi, fxhi_d, hsz[X]*halo->nfel*sizeof(double),
		  cudaMemcpyDeviceToHost, streamX);

  nblocks = (hsz[Y] + DEFAULT_TPB - 1) / DEFAULT_TPB;
  halo_pack_gpu_d<<<nblocks, DEFAULT_TPB, 0, streamY>>>
	(halo->d, Y, fylo_d, fyhi_d, f_d);

  cudaMemcpyAsync(fylo, fylo_d, hsz[Y]*halo->nfel*sizeof(double),
		  cudaMemcpyDeviceToHost, streamY);
  cudaMemcpyAsync(fyhi, fyhi_d, hsz[Y]*halo->nfel*sizeof(double),
		  cudaMemcpyDeviceToHost, streamY);

  nblocks = (hsz[Z] + DEFAULT_TPB - 1) / DEFAULT_TPB;
  halo_pack_gpu_d<<<nblocks, DEFAULT_TPB, 0, streamZ>>>
	(halo->d, Z, fzlo_d, fzhi_d, f_d);

  cudaMemcpyAsync(fzlo, fzlo_d, hsz[Z]*halo->nfel*sizeof(double),
		  cudaMemcpyDeviceToHost, streamZ);
  cudaMemcpyAsync(fzhi, fzhi_d, hsz[Z]*halo->nfel*sizeof(double),
		  cudaMemcpyDeviceToHost, streamZ);


 /* Wait for X; copy or MPI recvs; put X halos back on device, and unpack */

  cudaStreamSynchronize(streamX);

  if (cart_size(X) == 1) {
    /* fxhi -> hxlo */
    memcpy(hxlo, fxhi, hsz[X]*halo->nfel*sizeof(double)); 
    cudaMemcpyAsync(hxlo_d, hxlo, hsz[X]*halo->nfel*sizeof(double),
		    cudaMemcpyHostToDevice, streamX);
    /* fxlo -> hxhi */
    memcpy(hxhi, fxlo, hsz[X]*halo->nfel*sizeof(double));
    cudaMemcpyAsync(hxhi_d, hxhi, hsz[X]*halo->nfel*sizeof(double),
		    cudaMemcpyHostToDevice, streamX);
  }
  else {
    MPI_Isend(fxhi, hsz[X]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(FORWARD,X), ftagx, comm, req_x + 2);
    MPI_Isend(fxlo, hsz[X]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(BACKWARD,X), btagx, comm, req_x + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_x, &mc, status);
      if (mc == 0) cudaMemcpyAsync(hxlo_d, hxlo,
	hsz[X]*halo->nfel*sizeof(double), cudaMemcpyHostToDevice, streamX);
      if (mc == 1) cudaMemcpyAsync(hxhi_d, hxhi,
	hsz[X]*halo->nfel*sizeof(double), cudaMemcpyHostToDevice, streamX);
    }
  }

  nblocks = (hsz[X] + DEFAULT_TPB - 1) / DEFAULT_TPB;
  halo_unpack_gpu_d<<<nblocks, DEFAULT_TPB, 0, streamX>>>
	(halo->d, X, f_d, hxlo_d, hxhi_d);

  /* Now wait for Y data to arrive from device */
  /* Fill in 4 corners of Y edge data from X halo */

  cudaStreamSynchronize(streamY);

  ih = halo->hext[Y][X] - nh;
  jh = halo->hext[X][Y] - nh - halo->nswap;

  for (ic = 0; ic < halo->nswap; ic++) {
    for (jc = 0; jc < halo->nswap; jc++) {
      for (kc = 0; kc < halo->nsites[Z]; kc++) {

        ixlo = get_linear_index(     ic, nh + jc, kc, halo->hext[X]);
        iylo = get_linear_index(nd + ic,      jc, kc, halo->hext[Y]);
        ixhi = get_linear_index(ih + ic,      jc, kc, halo->hext[Y]);
        iyhi = get_linear_index(ic,      jh + jc, kc, halo->hext[X]);

        for (p = 0; p < halo->nfel; p++) {
          fylo[hsz[Y]*p + iylo] = hxlo[hsz[X]*p + ixlo];
          fyhi[hsz[Y]*p + iylo] = hxlo[hsz[X]*p + iyhi];
          fylo[hsz[Y]*p + ixhi] = hxhi[hsz[X]*p + ixlo];
          fyhi[hsz[Y]*p + ixhi] = hxhi[hsz[X]*p + iyhi];
        }
      }
    }
  }


  /* Swap in Y, send data back to device and unpack */

  if (cart_size(Y) == 1) {
    /* fyhi -> hylo */
    memcpy(hylo, fyhi, hsz[Y]*halo->nfel*sizeof(double));
    cudaMemcpyAsync(hylo_d, hylo, hsz[Y]*halo->nfel*sizeof(double),
		  cudaMemcpyHostToDevice, streamY);
    /* fylo -> hyhi */
    memcpy(hyhi, fylo, hsz[Y]*halo->nfel*sizeof(double));
    cudaMemcpyAsync(hyhi_d, hyhi, hsz[Y]*halo->nfel*sizeof(double),
		  cudaMemcpyHostToDevice, streamY);
  }
  else {
    MPI_Isend(fyhi, hsz[Y]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(FORWARD,Y), ftagy, comm, req_y + 2);
    MPI_Isend(fylo, hsz[Y]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Y), btagy, comm, req_y + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_y, &mc, status);
      if (mc == 0) cudaMemcpyAsync(hylo_d, hylo,
	hsz[Y]*halo->nfel*sizeof(double), cudaMemcpyHostToDevice, streamY);
      if (mc == 1) cudaMemcpyAsync(hyhi_d, hyhi,
	hsz[Y]*halo->nfel*sizeof(double), cudaMemcpyHostToDevice, streamY);
    }
  }


  nblocks = (hsz[Y] + DEFAULT_TPB - 1) / DEFAULT_TPB;
  halo_unpack_gpu_d<<<nblocks, DEFAULT_TPB, 0, streamY>>>
	(halo->d, Y, f_d, hylo_d, hyhi_d);

  /* Wait for Z data from device */
  /* Fill in 4 corners of Z edge data from X halo  */

  cudaStreamSynchronize(streamZ);

  ih = halo->hext[Z][X] - nh;
  kh = halo->hext[X][Z] - nh - halo->nswap;

  for (ic = 0; ic < halo->nswap; ic++) {
    for (jc = 0; jc < halo->nsites[Y]; jc++) {
      for (kc = 0; kc < halo->nswap; kc++) {

        ixlo = get_linear_index(     ic, jc, nh + kc, halo->hext[X]);
        izlo = get_linear_index(nd + ic, jc,      kc, halo->hext[Z]);
        ixhi = get_linear_index(     ic, jc, kh + kc, halo->hext[X]);
        izhi = get_linear_index(ih + ic, jc,      kc, halo->hext[Z]);

        for (p = 0; p < halo->nfel; p++) {
          fzlo[hsz[Z]*p + izlo] = hxlo[hsz[X]*p + ixlo];
          fzhi[hsz[Z]*p + izlo] = hxlo[hsz[X]*p + ixhi];
          fzlo[hsz[Z]*p + izhi] = hxhi[hsz[X]*p + ixlo];
          fzhi[hsz[Z]*p + izhi] = hxhi[hsz[X]*p + ixhi];
        }
      }
    }
  }

  /* Fill in 4 strips in X of Z edge data: from Y halo  */

  jh = halo->hext[Z][Y] - nh;
  kh = halo->hext[Y][Z] - nh - halo->nswap;
  
  for (ic = 0; ic < halo->nsites[X]; ic++) {
    for (jc = 0; jc < halo->nswap; jc++) {
      for (kc = 0; kc < halo->nswap; kc++) {

        iylo = get_linear_index(ic,      jc, nh + kc, halo->hext[Y]);
        izlo = get_linear_index(ic, nd + jc,      kc, halo->hext[Z]);
        iyhi = get_linear_index(ic,      jc, kh + kc, halo->hext[Y]);
        izhi = get_linear_index(ic, jh + jc,      kc, halo->hext[Z]);

        for (p = 0; p < halo->nfel; p++) {
          fzlo[hsz[Z]*p + izlo] = hylo[hsz[Y]*p + iylo];
          fzhi[hsz[Z]*p + izlo] = hylo[hsz[Y]*p + iyhi];
          fzlo[hsz[Z]*p + izhi] = hyhi[hsz[Y]*p + iylo];
          fzhi[hsz[Z]*p + izhi] = hyhi[hsz[Y]*p + iyhi];
        }
      }
    }
  }

  /* The z-direction swap  */

  if (cart_size(Z) == 1) {
    /* fzhi -> hzlo */
    memcpy(hzlo, fzhi, hsz[Z]*halo->nfel*sizeof(double));
    cudaMemcpyAsync(hzlo_d, hzlo, hsz[Z]*halo->nfel*sizeof(double),
		  cudaMemcpyHostToDevice, streamZ);
    /* fzlo -> hzhi */
    memcpy(hzhi, fzlo, hsz[Z]*halo->nfel*sizeof(double));
    cudaMemcpyAsync(hzhi_d, hzhi, hsz[Z]*halo->nfel*sizeof(double),
		  cudaMemcpyHostToDevice, streamZ);
  }
  else {
    MPI_Isend(fzhi, hsz[Z]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(FORWARD,Z), ftagz, comm, req_z + 2);
    MPI_Isend(fzlo,  hsz[Z]*halo->nfel, MPI_DOUBLE,
	      cart_neighb(BACKWARD,Z), btagz, comm, req_z + 3);

    for (m = 0; m < 4; m++) {
      MPI_Waitany(4, req_z, &mc, status);
      if (mc == 0) cudaMemcpyAsync(hzlo_d, hzlo,
	hsz[Z]*halo->nfel*sizeof(double), cudaMemcpyHostToDevice, streamZ);
      if (mc == 1) cudaMemcpyAsync(hzhi_d, hzhi,
	hsz[Z]*halo->nfel*sizeof(double), cudaMemcpyHostToDevice, streamZ);
    }
  }

  nblocks = (hsz[Z] + DEFAULT_TPB - 1) / DEFAULT_TPB;
  halo_unpack_gpu_d<<<nblocks, DEFAULT_TPB, 0, streamZ>>>
	(halo->d, Z, f_d, hzlo_d, hzhi_d);

  cudaStreamSynchronize(streamX);
  cudaStreamSynchronize(streamY);
  cudaStreamSynchronize(streamZ);

  return 0;
}

/*****************************************************************************
 *
 *  halo_pack_gpu_d
 *
 *  Move data to halo buffer on device for coordinate
 *  direction id at both low and high ends.
 *
 *****************************************************************************/

__global__
static void halo_pack_gpu_d(cuda_halo_t * halo, int id,
		            double * flo_d,
			    double * fhi_d, 
			    double * f_d) {
  int threadIndex;
  int nh;
  int p, indexl, indexh, ii,jj,kk;
  int ho; /* high end offset */

  threadIndex = blockIdx.x*blockDim.x + threadIdx.x;
  
  if (threadIndex >= halo->hsz[id]) return;

  /* Load two buffers for this site */
  /* Use full nhalo to address full f_d */

  nh = halo->nhalo;
  get_coords_from_index_gpu_d(&ii, &jj, &kk, threadIndex, halo->hext[id]);

  if (id == X) {
    ho = nh + halo->nlocal[X] - halo->nswap;
    indexl = get_linear_index_gpu_d(nh + ii, jj, kk, halo->nsites);
    indexh = get_linear_index_gpu_d(ho + ii, jj, kk, halo->nsites);
  }
  if (id == Y) {
    ho = nh + halo->nlocal[Y] - halo->nswap;
    indexl = get_linear_index_gpu_d(ii, nh + jj, kk, halo->nsites);
    indexh = get_linear_index_gpu_d(ii, ho + jj, kk, halo->nsites);
  }
  if (id == Z) {
    ho = nh + halo->nlocal[Z] - halo->nswap;
    indexl = get_linear_index_gpu_d(ii, jj, nh + kk, halo->nsites);
    indexh = get_linear_index_gpu_d(ii, jj, ho + kk, halo->nsites);
  }

  /* Low end, and high end */

  for (p = 0; p < halo->nfel; p++) {
    flo_d[halo->hsz[id]*p + threadIndex] = f_d[halo->nsite*p + indexl];
  }

  for (p = 0; p < halo->nfel; p++) {
    fhi_d[halo->hsz[id]*p + threadIndex] = f_d[halo->nsite*p + indexh];
  }

  return;
}

/*****************************************************************************
 *
 *  halo_unpack_gpu_d
 *
 *  Unpack halo buffers to the distribution on device for direction id.
 *
 *****************************************************************************/

__global__
static void halo_unpack_gpu_d(cuda_halo_t * halo, int id,
		              double * f_d,
                              double * hlo_d,
			      double * hhi_d) {
  int threadIndex;
  int p, indexl, indexh;
  int nh;                          /* Full halo width */
  int ic, jc, kc;                  /* Lattice ooords */
  int lo, ho;                      /* Offset for low, high end */

  threadIndex = blockIdx.x*blockDim.x + threadIdx.x;
  if (threadIndex >= halo->hsz[id]) return;

  /* Unpack buffer this site. */

  nh = halo->nhalo;
  get_coords_from_index_gpu_d(&ic, &jc, &kc, threadIndex, halo->hext[id]);

  if (id == X) {
    lo = nh - halo->nswap;
    ho = nh + halo->nlocal[X];
    indexl = get_linear_index_gpu_d(lo + ic, jc, kc, halo->nsites);
    indexh = get_linear_index_gpu_d(ho + ic, jc, kc, halo->nsites);
  }

  if (id == Y) {
    lo = nh - halo->nswap;
    ho = nh + halo->nlocal[Y];
    indexl = get_linear_index_gpu_d(ic, lo + jc, kc, halo->nsites);
    indexh = get_linear_index_gpu_d(ic, ho + jc, kc, halo->nsites);
  }

  if (id == Z) {
    lo = nh - halo->nswap;
    ho = nh + halo->nlocal[Z];
    indexl = get_linear_index_gpu_d(ic, jc, lo + kc, halo->nsites);
    indexh = get_linear_index_gpu_d(ic, jc, ho + kc, halo->nsites);
  } 

  /* Low end, then high end */

  for (p = 0; p < halo->nfel; p++) {
    f_d[halo->nsite*p + indexl] = hlo_d[halo->hsz[id]*p + threadIndex];
  }

  for (p = 0; p < halo->nfel; p++) {
    f_d[halo->nsite*p + indexh] = hhi_d[halo->hsz[id]*p + threadIndex];
  }

  return;
}

#endif

#endif

#endif 
