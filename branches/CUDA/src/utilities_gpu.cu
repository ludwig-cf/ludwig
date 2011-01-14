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

#include "utilities_gpu.h"
#include "util.h"
#include "model.h"
#include "timer.h"

/* external pointers to data on host*/
extern double * f_;
extern const double ma_[NVEL][NVEL];
extern const double mi_[NVEL][NVEL];
extern const double wv[NVEL];
extern const int cv[NVEL][3];
extern const double q_[NVEL][3][3];


/* accelerator memory address pointers for required data structures */
double * f_d;
double * ftmp_d;
double * fedge_d;
double * fhalo_d;
double * ma_d;
double * mi_d;
double * d_d;
int * cv_d;
double * q_d;
double * wv_d;
char * site_map_status_d;
double * force_d;
double * velocity_d;
double * phi_site_d;
double * grad_phi_site_d;
double * delsq_phi_site_d;
int * N_d;
double * force_global_d;

/* host memory address pointers for temporary staging of data */
char * site_map_status_temp;
double * force_temp;
double * velocity_temp;
double * phi_site_temp;
double * grad_phi_site_temp;
double * delsq_phi_site_temp;
double * fedge;
double * fhalo;

/* data size variables */
static int ndata;
static int nhalo;
static int nsites;
static int ndist;
static  int N[3];
static  int Nall[3];
static int maxN;
static int nedgedata;
static int nhalodata;



/* Perform tasks necessary to initialise accelerator */
void initialise_gpu()
{

  double force_global[3];

  int ic,jc,kc,index;
  
  //cudaSetDevice(4);

  calculate_data_sizes();
  allocate_memory_on_gpu();

  /* get global force from physics module */
  fluid_body_force(force_global);

  /* get temp host copies of force and site_map_status arrays */
  for (ic=0; ic<=N[X]; ic++)
    {
      for (jc=0; jc<=N[Y]; jc++)
	{
	  for (kc=0; kc<=N[Z]; kc++)
	    {
	      index = coords_index(ic, jc, kc); 

	      site_map_status_temp[index] = site_map_get_status(ic, jc, kc);

	    }
	}
    }




  /* copy data from host to accelerator */
  cudaMemcpy(N_d, N, 3*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpy(ma_d, ma_, NVEL*NVEL*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(mi_d, mi_, NVEL*NVEL*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_d, d_, 3*3*sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpy(cv_d, cv, NVEL*3*sizeof(int), cudaMemcpyHostToDevice); 
  cudaMemcpy(wv_d, wv, NVEL*sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpy(q_d, q_, NVEL*3*3*sizeof(double), cudaMemcpyHostToDevice); 
  cudaMemcpy(force_global_d, force_global, 3*sizeof(double), \
	     cudaMemcpyHostToDevice);
  cudaMemcpy(site_map_status_d, site_map_status_temp, nsites*sizeof(char), \
	     cudaMemcpyHostToDevice);


}

/* Perform tasks necessary to finalise accelerator */
void finalise_gpu()
{
  free_memory_on_gpu();
}




/* calculate sizes of data - needed for memory copies to accelerator */
static void calculate_data_sizes()
{
  coords_nlocal(N);  
  nhalo = coords_nhalo();  
  ndist = distribution_ndist();

  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsites = Nall[X]*Nall[Y]*Nall[Z];
  ndata = nsites * ndist * NVEL;
  maxN = max(N[X],N[Y]);
  maxN = max(maxN,N[Z]);
  nedgedata = maxN * maxN * nhalo * 12 * ndist * NVEL;
  nhalodata = (maxN+2*nhalo) * (maxN+2*nhalo) * nhalo * 12 * ndist * NVEL;
}



/* Allocate memory on accelerator */
static void allocate_memory_on_gpu()
{

  /* temp arrays for staging data on  host */
  force_temp = (double *) malloc(nsites*3*sizeof(double));
  velocity_temp = (double *) malloc(nsites*3*sizeof(double));
  site_map_status_temp = (char *) malloc(nsites*sizeof(char));
  phi_site_temp = (double *) malloc(nsites*sizeof(double));
  grad_phi_site_temp = (double *) malloc(nsites*3*sizeof(double));
  delsq_phi_site_temp = (double *) malloc(nsites*sizeof(double));
  fedge = (double *) malloc(nedgedata*sizeof(double));
  fhalo = (double *) malloc(nhalodata*sizeof(double));

  /* arrays on accelerator */
  cudaMalloc((void **) &f_d, ndata*sizeof(double));
  cudaMalloc((void **) &ftmp_d, ndata*sizeof(double));
  cudaMalloc((void **) &fedge_d, nedgedata*sizeof(double));
  cudaMalloc((void **) &fhalo_d, nhalodata*sizeof(double));
  cudaMalloc((void **) &site_map_status_d, nsites*sizeof(char));
  cudaMalloc((void **) &ma_d, NVEL*NVEL*sizeof(double));
  cudaMalloc((void **) &mi_d, NVEL*NVEL*sizeof(double));
  cudaMalloc((void **) &d_d, 3*3*sizeof(double));
  cudaMalloc((void **) &cv_d, NVEL*3*sizeof(int));
  cudaMalloc((void **) &wv_d, NVEL*sizeof(double));
  cudaMalloc((void **) &q_d, NVEL*3*3*sizeof(double));
  cudaMalloc((void **) &force_d, nsites*3*sizeof(double));
  cudaMalloc((void **) &velocity_d, nsites*3*sizeof(double));
  cudaMalloc((void **) &phi_site_d, nsites*sizeof(double));
  cudaMalloc((void **) &delsq_phi_site_d, nsites*sizeof(double));
  cudaMalloc((void **) &grad_phi_site_d, nsites*3*sizeof(double));
  cudaMalloc((void **) &N_d, sizeof(int)*3);
  cudaMalloc((void **) &force_global_d, sizeof(double)*3);
  //checkCUDAError("allocate_memory_on_gpu");

}


/* Free memory on accelerator */
static void free_memory_on_gpu()
{

  /* free temp memory on host */
  free(force_temp);
  free(velocity_temp);
  free(site_map_status_temp);
  free(phi_site_temp);
  free(grad_phi_site_temp);
  free(delsq_phi_site_temp);
  free(fedge);
  free(fhalo);

  /* free memory on accelerator */
  cudaFree(f_d);
  cudaFree(ftmp_d);
  cudaFree(fedge_d);
  cudaFree(fhalo_d);
  cudaFree(ma_d);
  cudaFree(mi_d);
  cudaFree(d_d);
  cudaFree(cv_d);
  cudaFree(wv_d);
  cudaFree(q_d);
  cudaFree(site_map_status_d);
  cudaFree(force_d);
  cudaFree(velocity_d);
  cudaFree(phi_site_d);
  cudaFree(delsq_phi_site_d);
  cudaFree(grad_phi_site_d);
  cudaFree(N_d);
  cudaFree(force_global_d);

}



/* copy f_ from host to accelerator */
void put_f_on_gpu()
{
  /* copy data from CPU to accelerator */
  cudaMemcpy(f_d, f_, ndata*sizeof(double), cudaMemcpyHostToDevice);

  //checkCUDAError("put_f_on_gpu");

}

/* copy force from host to accelerator */
void put_force_on_gpu()
{

  int index, i, ic, jc, kc;
  double force[3];
	      

  /* get temp host copies of arrays */
  for (ic=1; ic<=N[X]; ic++)
    {
      for (jc=1; jc<=N[Y]; jc++)
	{
	  for (kc=1; kc<=N[Z]; kc++)
	    {
	      index = coords_index(ic, jc, kc); 

	      hydrodynamics_get_force_local(index,force);
	      	      
	      for (i=0;i<3;i++)
		{
		  force_temp[index*3+i]=force[i];
		}
	    }
	}
    }


  /* copy data from CPU to accelerator */
  cudaMemcpy(force_d, force_temp, nsites*3*sizeof(double), \
	     cudaMemcpyHostToDevice);

  //checkCUDAError("put_force_on_gpu");

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


void get_velocity_from_gpu()
{
  int index,i, ic,jc,kc; 
  double velocity[3];

  cudaMemcpy(velocity_temp, velocity_d, nsites*3*sizeof(double), 
	    cudaMemcpyDeviceToHost);
  //checkCUDAError("get_velocity_from_gpu");

  /* copy velocity from temporary array back to hydrodynamics module */
  for (ic=1; ic<=N[X]; ic++)
    {
      for (jc=1; jc<=N[Y]; jc++)
	{
	  for (kc=1; kc<=N[Z]; kc++)
	    {
	      index = coords_index(ic, jc, kc); 
	      for (i=0;i<3;i++)
		{		 
		  velocity[i]=velocity_temp[index*3+i];
		}     
	      hydrodynamics_set_velocity(index,velocity);
	    }
	}
    }

}

/* copy phi from host to accelerator */
void put_phi_on_gpu()
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


	      phi_site_temp[index]=phi_get_phi_site(index);
	      phi_gradients_grad(index, grad_phi);
	      delsq_phi_site_temp[index] = phi_gradients_delsq(index);
	      	      
	      for (i=0;i<3;i++)
		{
		  grad_phi_site_temp[index*3+i]=grad_phi[i];
		}
	    }
	}
    }


  /* copy data from CPU to accelerator */
  cudaMemcpy(phi_site_d, phi_site_temp, nsites*sizeof(double), \
	     cudaMemcpyHostToDevice);
  cudaMemcpy(delsq_phi_site_d, delsq_phi_site_temp, nsites*sizeof(double), \
	     cudaMemcpyHostToDevice);
  cudaMemcpy(grad_phi_site_d, grad_phi_site_temp, nsites*3*sizeof(double), \
	     cudaMemcpyHostToDevice);


  //checkCUDAError("put_phi_on_gpu");

}



/* copy phi from accelerator to host */
void get_phi_site_from_gpu()
{

  int index, i, ic, jc, kc;
	      

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






/* copy f_ edges from accelerator to host */
void get_f_edges_from_gpu()
{

 int ii, jj, kk, p, m, index;
 int offset, packed_index;
 static dim3 BlockDims;
 static dim3 GridDims;
 
 int npackedsite = maxN*maxN;


  /* pack edges on accelerator */

  /* set up CUDA grid */
  #define BLOCKSIZE 256
  /* 1D decomposition - use x grid and block dimension only */
  BlockDims.x=BLOCKSIZE;
  GridDims.x=(N[X]*N[Y]*N[Z]+BlockDims.x-1)/BlockDims.x;

  /* run the kernel */
  TIMER_start(EDGEPACK);
  pack_edges_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,maxN,
  				      N_d,fedge_d,f_d);

  cudaThreadSynchronize();
  TIMER_stop(EDGEPACK);


  /* copy data from accelerator to host */
  TIMER_start(EDGEGET);
  cudaMemcpy(fedge, fedge_d, nedgedata*sizeof(double), cudaMemcpyDeviceToHost);
  TIMER_stop(EDGEGET);


  /* Unpack edges */ 
  
  TIMER_start(EDGEUNPACK);
  
  for (ii = 0; ii < N[X]; ii++) {
    for (jj = 0; jj < N[Y]; jj++) {
      for (kk = 0; kk < N[Z]; kk++) {
	
	/* only operate on edge sites */
	if (  (ii >= 0 && ii < nhalo) ||		\
	      (jj >= 0 && jj < nhalo) ||		\
	      (kk >= 0 && kk < nhalo) ||		\
	      (ii >= (N[X]-nhalo) && ii < N[X]) ||	\
	      (jj >= (N[Y]-nhalo) && jj < N[Y]) ||	\
	      (kk >= (N[Z]-nhalo) && kk < N[Z])		\
	      )
	  {
	    	    
	    /* get location of data in packed array */
	    get_packed_index_offset(&packed_index,&offset,ii,jj,
				    kk,nhalo,N,ndist);	    
	    
	    /* get index for original array */
	    index = coords_index(ii+1,jj+1,kk+1);
	    
	    /* copy edge data from packed array to original array */
	    
	    for (p = 0; p < NVEL; p++) { 
	      for (m = 0; m < ndist; m++) { 
		
		f_[ndist*nsites*p+nsites*m+index] = 
		  fedge[offset+ndist*npackedsite*p+m*npackedsite+packed_index]; 
		
	      } 
	    }
	    
	  }
	
      }
    }
  }
  
  TIMER_stop(EDGEUNPACK);
  
  
  //checkCUDAError("get_f_edges_from_gpu");

}




/* copy f_ halos from host to accelerator */
void put_f_halos_on_gpu()
{

  int ii, jj, kk, p, m, index;
  int offset, packed_index;
  static dim3 BlockDims;
  static dim3 GridDims;

  int npackedsite = (maxN+2*nhalo)*(maxN+2*nhalo);


  /* Pack haloes */ 
  TIMER_start(HALOPACK);
  for (ii = 0; ii < Nall[X]; ii++) {
    for (jj = 0; jj < Nall[Y]; jj++) {
      for (kk = 0; kk < Nall[Z]; kk++) {
	
	
	/* only threads which are operating on halo sites are active*/
	if (  (ii >= 0 && ii < nhalo) ||	\
	      (jj >= 0 && jj < nhalo) ||		 \
	      (kk >= 0 && kk < nhalo) ||		 \
	      (ii >= (Nall[X]-nhalo) && ii < Nall[X] ) ||	\
	      (jj >= (Nall[Y]-nhalo) && jj < Nall[Y] ) ||	\
	      (kk >= (Nall[Z]-nhalo) && kk < Nall[Z] )		\
	      )
	  {
	    
	    
	    /* get location of data in packed array */
	    get_packed_index_offset(&packed_index,&offset,ii,jj,
				    kk,nhalo,Nall,ndist);	    
	    
	    /* get index for original array */
	    index = get_linear_index(ii,jj,kk,Nall);
	    
	    /* copy edge data from original array to packed array */
	    for (p = 0; p < NVEL; p++) { 
	      for (m = 0; m < ndist; m++) { 

		fhalo[offset+ndist*npackedsite*p+m*npackedsite+packed_index] = f_[ndist*nsites*p+nsites*m+index];	      
	      } 
	    }
	    
	  }
	
      }
    }
  }
  TIMER_stop(HALOPACK);


  /* copy data from host to accelerator */
  TIMER_start(HALOPUT);
  cudaMemcpy(fhalo_d, fhalo, nhalodata*sizeof(double), cudaMemcpyHostToDevice);
  TIMER_stop(HALOPUT);

  /* set up CUDA grid */
  #define BLOCKSIZE 256
  /* 1D decomposition - use x grid and block dimension only */
  BlockDims.x=BLOCKSIZE;
  GridDims.x=(Nall[X]*Nall[Y]*Nall[Z]+BlockDims.x-1)/BlockDims.x;

  /* run the kernel */
  TIMER_start(HALOUNPACK);
  unpack_halos_gpu_d<<<GridDims.x,BlockDims.x>>>(ndist,nhalo,maxN,
  			      N_d,f_d,fhalo_d);

  cudaThreadSynchronize();
  TIMER_stop(HALOUNPACK);

  //checkCUDAError("get_f_edges_from_gpu");

}







/* get packed index and offset on the host */
 static void   get_packed_index_offset(int *packed_index,int *offset,
				       int ii,int jj,int kk,int nhalo,
				       int N[3],int ndist)
{

  
  int mpack,npack,extent,site_size;
  

  /* get indices and offsets for packed data structure */
  /* six planes (of depth nhalo) corresponding to edges of cuboid */
  extent=max(N[X],N[Y]);
  extent=max(extent,N[Z]);
  site_size=ndist*NVEL*extent*extent;
  
  if (  (ii >= 0 && ii < nhalo) )
    {
      mpack=jj;
      npack=kk;
      *offset=ii*site_size; 
    }
  
  if (  (jj >= 0 && jj < nhalo) )
    {
      mpack=ii;
      npack=kk;
      *offset=(nhalo+jj)*site_size;
    }
  if (  (kk >= 0 && kk < nhalo) )
    {
      mpack=ii;
      npack=jj;
      *offset=(2*nhalo+kk)*site_size;
    }
  if ( ii >= (N[X]-nhalo) && ii < N[X] )
    {
      mpack=jj;
      npack=kk;
      *offset=(3*nhalo+(ii-N[X]+nhalo))*site_size;
    }
  if ( jj >= (N[Y]-nhalo) && jj < N[Y] )
    {
      mpack=ii;
      npack=kk;
      *offset=(4*nhalo+(jj-N[Y]+nhalo))*site_size;
    }
  if ( kk >= (N[Z]-nhalo) && kk < N[Z] )
    {
      mpack=ii;
      npack=jj;
      *offset=(5*nhalo+(kk-N[Z]+nhalo))*site_size;
    }
  
  *packed_index = mpack*extent + npack;
  
  
 } 

/* get linear index from 3d coordinates (host) */
static int get_linear_index(int ii,int jj,int kk,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

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

/* get packed index and offset on the accelerator */
 __device__ static void   get_packed_index_offset_gpu_d(int *packed_index,int *offset,int ii,int jj,int kk,int nhalo,int N[3],int ndist)
{

  
  int mpack,npack,extent,site_size;
  

  /* get indices and offsets for packed data structure */
  /* six planes (of depth nhalo) corresponding to edges of cuboid */
  extent=max(N[X],N[Y]);
  extent=max(extent,N[Z]);
  site_size=ndist*NVEL*extent*extent;
  
  if (  (ii >= 0 && ii < nhalo) )
    {
      mpack=jj;
      npack=kk;
      *offset=ii*site_size; 
    }
  
  if (  (jj >= 0 && jj < nhalo) )
    {
      mpack=ii;
      npack=kk;
      *offset=(nhalo+jj)*site_size;
    }
  if (  (kk >= 0 && kk < nhalo) )
    {
      mpack=ii;
      npack=jj;
      *offset=(2*nhalo+kk)*site_size;
    }
  if ( ii >= (N[X]-nhalo) && ii < N[X] )
    {
      mpack=jj;
      npack=kk;
      *offset=(3*nhalo+(ii-N[X]+nhalo))*site_size;
    }
  if ( jj >= (N[Y]-nhalo) && jj < N[Y] )
    {
      mpack=ii;
      npack=kk;
      *offset=(4*nhalo+(jj-N[Y]+nhalo))*site_size;
    }
  if ( kk >= (N[Z]-nhalo) && kk < N[Z] )
    {
      mpack=ii;
      npack=jj;
      *offset=(5*nhalo+(kk-N[Z]+nhalo))*site_size;
    }
  
  *packed_index = mpack*extent + npack;
  
 
 } 

/* get linear index from 3d coordinates (device) */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}

/* pack edges on the accelerator */
__global__ static void pack_edges_gpu_d(int ndist, int nhalo, int maxN, int N[3], double* fedge_d, double* f_d) {

  int p,m, index,ii,jj,kk;
  int packed_index, offset;

  int nsite = (N[X]+2*nhalo)*(N[Y]+2*nhalo)*(N[Z]+2*nhalo);
  int npackedsite = maxN*maxN;

  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;
  
  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;
  
  if (threadIndex < N[X]*N[Y]*N[Z])
    {
      
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,N);

              
      /* only threads which are operating on edge sites are active*/
      if (  (ii >= 0 && ii < nhalo) ||	\
	    (jj >= 0 && jj < nhalo) ||	\
	    (kk >= 0 && kk < nhalo) ||		  \
	    (ii >= (N[X]-nhalo) && ii < N[X]) ||  \
	    (jj >= (N[Y]-nhalo) && jj < N[Y]) ||  \
	    (kk >= (N[Z]-nhalo) && kk < N[Z])	  \
	    )
	{
	  
	  index = get_linear_index_gpu_d(ii+nhalo,jj+nhalo,kk+nhalo,Nall);
	  
	  get_packed_index_offset_gpu_d(&packed_index,&offset,ii,jj,kk,nhalo,N,ndist);
	  
	  
	  /* copy data to packed structure */
	  for (p = 0; p < NVEL; p++) {
	    for (m = 0; m < ndist; m++) {
	      
	      fedge_d[offset+ndist*npackedsite*p+m*npackedsite+packed_index] 
		= f_d[ndist*nsite*p+nsite*m+index];
	      
	      
	    }	   
	  }
	}       
     }
  
}

/* unpack halos on the accelerator */
__global__ static void unpack_halos_gpu_d(int ndist, int nhalo, int maxN, 
					  int N[3], double* f_d, 
					  double* fhalo_d) {


  int ii,jj,kk,p,m,index;
  int packed_index, offset;

  int nsite = (N[X]+2*nhalo)*(N[Y]+2*nhalo)*(N[Z]+2*nhalo);
  int npackedsite = (maxN+2*nhalo)*(maxN+2*nhalo);

  int threadIndex = blockIdx.x*blockDim.x+threadIdx.x;


  int Nall[3];
  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  if (threadIndex < Nall[X]*Nall[Y]*Nall[Z] )
    {
      
      get_coords_from_index_gpu_d(&ii,&jj,&kk,threadIndex,Nall);
      
      /* only threads which are operating on halo sites are active*/
      if (  (ii >= 0 && ii < nhalo) ||	\
	    (jj >= 0 && jj < nhalo) ||			 \
	    (kk >= 0 && kk < nhalo) ||			 \
	    (ii >= (Nall[X]-nhalo) && ii < Nall[X] ) ||	 \
	    (jj >= (Nall[Y]-nhalo) && jj < Nall[Y] ) ||	 \
	    (kk >= (Nall[Z]-nhalo) && kk < Nall[Z] )	 \
	    )
	{
	  
	  
	  index = get_linear_index_gpu_d(ii,jj,kk,Nall);
	  
	  get_packed_index_offset_gpu_d(&packed_index,&offset,ii,jj,kk,nhalo,Nall,ndist);
	  
	  
	  /* copy data from packed structure */
	  for (p = 0; p < NVEL; p++) {
	    for (m = 0; m < ndist; m++) {

	      f_d[ndist*nsite*p+nsite*m+index] 
	      = fhalo_d[offset+ndist*npackedsite*p+m*npackedsite+packed_index];

	    }
	  }
	  
	}
      
    }
  
  
  
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
