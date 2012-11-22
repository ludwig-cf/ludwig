/*****************************************************************************
 *
 *  propagation_gpu.c
 *
 *  Propagation schemes for the different models.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *   Adapted to run on GPU: Alan Gray
 * 
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

/* below define needed to stop repeated declaration of distribution_ndist */
#define INCLUDING_FROM_GPU_SOURCE 

#include "model.h"
#include "propagation_gpu.h"

extern "C" void checkCUDAError(const char *msg);

/*****************************************************************************
 *
 *  propagation
 *
 *  Driver routine for the propagation stage.
 *
 *****************************************************************************/

__global__ static void propagate_d3q19_3D_gpu_d(int ndist, int nhalo, int N[3],
						double* fnew_d, double* fold_d);

__global__ static void propagate_d3q19_3Dsm_gpu_d(int ndist, int nhalo, int N[3],
						double* fnew_d, double* fold_d);


void propagation_gpu() {


  int ndist, nhalo;
  int N[3];

  ndist = distribution_ndist();
  nhalo = coords_nhalo();
  coords_nlocal(N); 


  cudaFuncSetCacheConfig(propagate_d3q19_gpu_d,cudaFuncCachePreferShared);

  /* copy f to ftmp on accelerator */
  //copy_f_to_ftmp_on_gpu();
  
  double *tmpptr=ftmp_d;
  ftmp_d=f_d;
  f_d=tmpptr;




  /* run the kernel */
  if (NVEL == 9){
    printf("propagate_d2q9 not yet supported in GPU mode\n");
    exit(1);
  }
  if (NVEL == 15){
    printf("propagate_d3q15 not yet supported in GPU mode\n");
    exit(1);
  }
  if (NVEL == 19){
  /* set up CUDA grid */
  /* 1D decomposition - use x grid and block dimension only */
  int nblocks=(N[X]*N[Y]*N[Z]+DEFAULT_TPB-1)/DEFAULT_TPB;
  propagate_d3q19_gpu_d<<<nblocks,DEFAULT_TPB>>>(ndist,nhalo,
      					      N_d,f_d,ftmp_d);


/*     #define TPB_X 4 */
/*     #define TPB_Y 8 */
/*     #define TPB_Z 16 */

/*     dim3 ThreadsPerBlock, BlocksPerGrid; */

/*     ThreadsPerBlock.x=TPB_X; */
/*     ThreadsPerBlock.y=TPB_Y; */
/*     ThreadsPerBlock.z=TPB_Z; */
  
/*     BlocksPerGrid.x=(N[X]+ThreadsPerBlock.x-1)/ThreadsPerBlock.x; */
/*     BlocksPerGrid.y=(N[Y]+ThreadsPerBlock.y-1)/ThreadsPerBlock.y; */
/*     BlocksPerGrid.z=(N[Z]+ThreadsPerBlock.z-1)/ThreadsPerBlock.z; */

/*     propagate_d3q19_3Dsm_gpu_d<<<BlocksPerGrid,ThreadsPerBlock>>>(ndist,nhalo, */
/*    					      N_d,f_d,ftmp_d); */

    cudaThreadSynchronize();

    checkCUDAError("propagation");
    // exit(1);

  }  
  return;
}





/*****************************************************************************
 *
 *  propagate_d3q19_gpu_d
 *
 *  Follows the velocities defined in d3q19.c
 *
 *  Adapted to run on GPU: Alan Gray
 *
 *****************************************************************************/


/* __device__ static void load_to_sm_gpu_d(double f_sm[TPB_X+2][TPB_Y+2][TPB_Z+2], int p, int n, int pstr, int nsite, int* Nall, double* fold_d){ */


/*   int ioffset=blockIdx.x*blockDim.x+1; */
/*   int joffset=blockIdx.y*blockDim.y+1; */
/*   int koffset=blockIdx.z*blockDim.z+1; */
  
  
  
/*   int ism,jsm,ksm,i_,j_,k_; */
/*   /\* load data to shared memory *\/ */
  
  
/*   int bX=blockDim.x; */
/*   int bY=blockDim.y; */
/*   int bZ=blockDim.z; */
  
/*   int tiX=threadIdx.x; */
/*   int tiY=threadIdx.y; */
/*   int tiZ=threadIdx.z; */
  
/*   int bS=bX*bY*bZ; */
/*   int bSsm=(bX+2)*(bY+2)*(bZ+2); */
  
/*   int yfac = (bZ+2); */
/*   int xfac = (bZ+2)*(bY+2); */
  
/*   int iter; */
  
/*   //printf("hello\n",); */
  
/*   for(iter=0; iter<=(bSsm/bS); iter++){ */
	  
/*     int ii = iter*bS + tiX*bY*bZ + tiY*bZ + tiZ; */
    
/*     if (ii<bSsm){ */
      
      
      
      
/*       ism = ii/xfac; */
/*       jsm = (ii-xfac*ism)/yfac; */
/*       ksm = ii-ism*xfac-jsm*yfac; */
      
/*       i_= (ioffset-1)+ism; */
/*       j_= (joffset-1)+jsm; */
/*       k_= (koffset-1)+ksm; */
      
/*       int itmp=get_linear_index_gpu_d(i_,j_,k_,Nall); */
      
/*       f_sm[ism][jsm][ksm]=fold_d[p*pstr+n*nsite+itmp]; */
/*     } */
/*   } */
  
/*   syncthreads(); */
  
/*   /\* end load data to shared memory *\/ */
  
  
/* } */

/* __global__ static void propagate_d3q19_3Dsm_gpu_d(int ndist, int nhalo, int N[3], */
/* 					     double* fnew_d, double* fold_d) { */

/*   int ii, jj, kk, index, n, threadIndex, nsite, Nall[3], p; */
/*   int xstr, ystr, zstr, pstr, xfac, yfac; */
/*   int ism, jsm, ksm; */

/*   __shared__ double f_sm[TPB_X+2][TPB_Y+2][TPB_Z+2]; */


/*   Nall[X]=N[X]+2*nhalo; */
/*   Nall[Y]=N[Y]+2*nhalo; */
/*   Nall[Z]=N[Z]+2*nhalo; */

/*   nsite = Nall[X]*Nall[Y]*Nall[Z]; */

/*   zstr = 1; */
/*   ystr = zstr*Nall[Z]; */
/*   xstr = ystr*Nall[Y]; */
/*   pstr = nsite*ndist; */


/*   /\* CUDA thread index *\/ */

/*   int xIndex = blockIdx.x*blockDim.x+threadIdx.x; */
/*   int yIndex = blockIdx.y*blockDim.y+threadIdx.y; */
/*   int zIndex = blockIdx.z*blockDim.z+threadIdx.z; */



/*   /\* Avoid going beyond problem domain *\/ */
/*   if (xIndex < N[X] && yIndex < N[Y] && zIndex < N[Z]) */
/*      { */

/*        index = get_linear_index_gpu_d(xIndex+1,yIndex+1,zIndex+1,Nall); */
    
/*       for (n = 0; n < ndist; n++) { */

/* /\* 	int ioffset=blockIdx.x*blockDim.x+1; *\/ */
/* /\* 	int joffset=blockIdx.y*blockDim.y+1; *\/ */
/* /\* 	int koffset=blockIdx.z*blockDim.z+1; *\/ */



/* /\* 	int ism,jsm,ksm,i_,j_,k_; *\/ */
/* /\* 	/\\* load data to shared memory *\\/ *\/ */
	

/* /\* 	int bX=blockDim.x; *\/ */
/* /\* 	int bY=blockDim.y; *\/ */
/* /\* 	int bZ=blockDim.z; *\/ */
	
/* /\* 	int tiX=threadIdx.x; *\/ */
/* /\* 	int tiY=threadIdx.y; *\/ */
/* /\* 	int tiZ=threadIdx.z; *\/ */
		
/* /\* 	int bS=bX*bY*bZ; *\/ */
/* /\* 	int bSsm=(bX+2)*(bY+2)*(bZ+2); *\/ */
	
/* /\* 	int yfac = (bZ+2); *\/ */
/* /\* 	int xfac = (bZ+2)*(bY+2); *\/ */

/* /\* 	int iter; *\/ */

/* /\* 	//printf("hello\n",); *\/ */
	
/* /\* 	for(iter=0; iter<=(bSsm/bS); iter++){ *\/ */
	  
/* /\* 	  int ii = iter*bS + tiX*bY*bZ + tiY*bZ + tiZ; *\/ */
	  
/* /\* 	  if (ii<bSsm){ *\/ */



	    
/* /\* 	    ism = ii/xfac; *\/ */
/* /\* 	    jsm = (ii-xfac*ism)/yfac; *\/ */
/* /\* 	    ksm = ii-ism*xfac-jsm*yfac; *\/ */
	    
/* /\* 	    i_= (ioffset-1)+ism; *\/ */
/* /\* 	    j_= (joffset-1)+jsm; *\/ */
/* /\* 	    k_= (koffset-1)+ksm; *\/ */

/* /\* 	    int itmp=get_linear_index_gpu_d(i_,j_,k_,Nall); *\/ */
	    
/* /\* 	    for (p=0; p<NVEL; p++) *\/ */
/* /\* 	      f_sm[p][ism][jsm][ksm]=fold_d[p*pstr+n*nsite+itmp]; *\/ */

/* /\* 	  } *\/ */
/* /\* 	} *\/ */
	
	
/* /\* 	syncthreads(); *\/ */
	
/* /\* 	/\\* end load data to shared memory *\\/ *\/ */



	

	



/* 	//fnew_d[0*pstr+n*nsite+index]=fold_d[0*pstr+n*nsite+index]; */
/* 	ism=threadIdx.x+1; */
/* 	jsm=threadIdx.y+1; */
/* 	ksm=threadIdx.z+1; */


/* 	load_to_sm_gpu_d(f_sm,0,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[0*pstr+n*nsite+index]=f_sm[ism][jsm][ksm]; */

/* 	/\* Distributions moving forward in memory. *\/ */
/* 	load_to_sm_gpu_d(f_sm,9,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[9*pstr+n*nsite+index]=f_sm[ism][jsm][ksm-1]; */

/* 	load_to_sm_gpu_d(f_sm,8,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[8*pstr+n*nsite+index]=f_sm[ism][jsm-1][ksm+1]; */

/* 	load_to_sm_gpu_d(f_sm,7,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[7*pstr+n*nsite+index]=f_sm[ism][jsm-1][ksm]; */

/* 	load_to_sm_gpu_d(f_sm,6,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[6*pstr+n*nsite+index]=f_sm[ism][jsm-1][ksm-1]; */

/* 	load_to_sm_gpu_d(f_sm,5,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[5*pstr+n*nsite+index]=f_sm[ism-1][jsm+1][ksm]; */

/* 	load_to_sm_gpu_d(f_sm,4,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[4*pstr+n*nsite+index]=f_sm[ism-1][jsm][ksm+1]; */

/* 	load_to_sm_gpu_d(f_sm,3,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[3*pstr+n*nsite+index]=f_sm[ism-1][jsm][ksm]; */

/* 	load_to_sm_gpu_d(f_sm,2,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[2*pstr+n*nsite+index]=f_sm[ism-1][jsm][ksm-1]; */

/* 	load_to_sm_gpu_d(f_sm,1,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[1*pstr+n*nsite+index]=f_sm[ism-1][jsm-1][ksm]; */


/* 	/\* Distributions moving backward in memory. *\/ */

/* 	load_to_sm_gpu_d(f_sm,10,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[10*pstr+n*nsite+index]=f_sm[ism][jsm][ksm+1]; */

/* 	load_to_sm_gpu_d(f_sm,11,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[11*pstr+n*nsite+index]=f_sm[ism][jsm+1][ksm-1]; */

/* 	load_to_sm_gpu_d(f_sm,12,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[12*pstr+n*nsite+index]=f_sm[ism][jsm+1][ksm]; */

/* 	load_to_sm_gpu_d(f_sm,13,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[13*pstr+n*nsite+index]=f_sm[ism][jsm+1][ksm+1]; */

/* 	load_to_sm_gpu_d(f_sm,14,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[14*pstr+n*nsite+index]=f_sm[ism+1][jsm-1][ksm]; */

/* 	load_to_sm_gpu_d(f_sm,15,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[15*pstr+n*nsite+index]=f_sm[ism+1][jsm][ksm-1]; */

/* 	load_to_sm_gpu_d(f_sm,16,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[16*pstr+n*nsite+index]=f_sm[ism+1][jsm][ksm]; */

/* 	load_to_sm_gpu_d(f_sm,17,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[17*pstr+n*nsite+index]=f_sm[ism+1][jsm][ksm+1]; */

/* 	load_to_sm_gpu_d(f_sm,18,n,pstr,nsite,Nall,fold_d);	 */
/* 	fnew_d[18*pstr+n*nsite+index]=f_sm[ism+1][jsm+1][ksm]; */


	
/*       } */


/*      } */


   
/*    return; */
/* } */


/* __global__ static void propagate_d3q19_3D_gpu_d(int ndist, int nhalo, int N[3], */
/* 					     double* fnew_d, double* fold_d) { */

/*   int ii, jj, kk, index, n, threadIndex, nsite, Nall[3]; */
/*   int xstr, ystr, zstr, pstr, xfac, yfac; */


/*   Nall[X]=N[X]+2*nhalo; */
/*   Nall[Y]=N[Y]+2*nhalo; */
/*   Nall[Z]=N[Z]+2*nhalo; */

/*   nsite = Nall[X]*Nall[Y]*Nall[Z]; */

/*   zstr = 1; */
/*   ystr = zstr*Nall[Z]; */
/*   xstr = ystr*Nall[Y]; */
/*   pstr = nsite*ndist;  */


/*   /\* CUDA thread index *\/ */

/*   int xIndex = blockIdx.x*blockDim.x+threadIdx.x; */
/*   int yIndex = blockIdx.y*blockDim.y+threadIdx.y; */
/*   int zIndex = blockIdx.z*blockDim.z+threadIdx.z; */

/*   /\* Avoid going beyond problem domain *\/ */
/*   if (xIndex < N[X] && yIndex < N[Y] && zIndex < N[Z]) */
/*      { */

/*        index = get_linear_index_gpu_d(xIndex+1,yIndex+1,zIndex+1,Nall); */
    
/*       for (n = 0; n < ndist; n++) { */
	
/* 	fnew_d[0*pstr+n*nsite+index]=fold_d[0*pstr+n*nsite+index]; */

/* 	/\* Distributions moving forward in memory. *\/ */

/* 	fnew_d[9*pstr+n*nsite+index]=fold_d[9*pstr+n*nsite+index+                    (-1)]; */
/* 	fnew_d[8*pstr+n*nsite+index]=fold_d[8*pstr+n*nsite+index+          (-1)*ystr+(+1)]; */
/* 	fnew_d[7*pstr+n*nsite+index]=fold_d[7*pstr+n*nsite+index+         +(-1)*ystr     ]; */
/* 	fnew_d[6*pstr+n*nsite+index]=fold_d[6*pstr+n*nsite+index+         +(-1)*ystr+(-1)]; */
/* 	fnew_d[5*pstr+n*nsite+index]=fold_d[5*pstr+n*nsite+index+(-1)*xstr+(+1)*ystr     ]; */
/* 	fnew_d[4*pstr+n*nsite+index]=fold_d[4*pstr+n*nsite+index+(-1)*xstr          +(+1)]; */
/* 	fnew_d[3*pstr+n*nsite+index]=fold_d[3*pstr+n*nsite+index+(-1)*xstr               ]; */
/* 	fnew_d[2*pstr+n*nsite+index]=fold_d[2*pstr+n*nsite+index+(-1)*xstr          +(-1)]; */
/* 	fnew_d[1*pstr+n*nsite+index]=fold_d[1*pstr+n*nsite+index+(-1)*xstr+(-1)*ystr     ]; */
	
/* 	/\* Distributions moving backward in memory. *\/   */
	
/* 	fnew_d[10*pstr+n*nsite+index]=fold_d[10*pstr+n*nsite+index+                    (+1)]; */
/* 	fnew_d[11*pstr+n*nsite+index]=fold_d[11*pstr+n*nsite+index+          (+1)*ystr+(-1)]; */
/* 	fnew_d[12*pstr+n*nsite+index]=fold_d[12*pstr+n*nsite+index+         +(+1)*ystr     ]; */
/* 	fnew_d[13*pstr+n*nsite+index]=fold_d[13*pstr+n*nsite+index+         +(+1)*ystr+(+1)]; */
/* 	fnew_d[14*pstr+n*nsite+index]=fold_d[14*pstr+n*nsite+index+(+1)*xstr+(-1)*ystr     ]; */
/* 	fnew_d[15*pstr+n*nsite+index]=fold_d[15*pstr+n*nsite+index+(+1)*xstr          +(-1)]; */
/* 	fnew_d[16*pstr+n*nsite+index]=fold_d[16*pstr+n*nsite+index+(+1)*xstr               ]; */
/* 	fnew_d[17*pstr+n*nsite+index]=fold_d[17*pstr+n*nsite+index+(+1)*xstr          +(+1)]; */
/* 	fnew_d[18*pstr+n*nsite+index]=fold_d[18*pstr+n*nsite+index+(+1)*xstr+(+1)*ystr     ]; */

	
/*       }  */


/*      } */


   
/*    return; */
/* } */


__global__ static void propagate_d3q19_gpu_d(int ndist, int nhalo, int N[3],
					     double* fnew_d, double* fold_d) {

  int ii, jj, kk, index, n, threadIndex, nsite, Nall[3];
  int xstr, ystr, zstr, pstr, xfac, yfac;


  Nall[X]=N[X]+2*nhalo;
  Nall[Y]=N[Y]+2*nhalo;
  Nall[Z]=N[Z]+2*nhalo;

  nsite = Nall[X]*Nall[Y]*Nall[Z];

  zstr = 1;
  ystr = zstr*Nall[Z];
  xstr = ystr*Nall[Y];
  pstr = nsite*ndist;


  /* CUDA thread index */
  threadIndex = blockIdx.x*blockDim.x+threadIdx.x;



  /* Avoid going beyond problem domain */
   if (threadIndex < N[X]*N[Y]*N[Z])
     {

      /* calculate index from CUDA thread index */
      yfac = N[Z];
      xfac = N[Y]*yfac;
       
      ii = threadIndex/xfac;
      jj = ((threadIndex-xfac*ii)/yfac);
      kk = (threadIndex-ii*xfac-jj*yfac);
      
      index = get_linear_index_gpu_d(ii+nhalo,jj+nhalo,kk+nhalo,Nall);
    
      for (n = 0; n < ndist; n++) {
	
	fnew_d[0*pstr+n*nsite+index]=fold_d[0*pstr+n*nsite+index];

	/* Distributions moving forward in memory. */

	fnew_d[9*pstr+n*nsite+index]=fold_d[9*pstr+n*nsite+index+                    (-1)];
	fnew_d[8*pstr+n*nsite+index]=fold_d[8*pstr+n*nsite+index+          (-1)*ystr+(+1)];
	fnew_d[7*pstr+n*nsite+index]=fold_d[7*pstr+n*nsite+index+         +(-1)*ystr     ];
	fnew_d[6*pstr+n*nsite+index]=fold_d[6*pstr+n*nsite+index+         +(-1)*ystr+(-1)];
	fnew_d[5*pstr+n*nsite+index]=fold_d[5*pstr+n*nsite+index+(-1)*xstr+(+1)*ystr     ];
	fnew_d[4*pstr+n*nsite+index]=fold_d[4*pstr+n*nsite+index+(-1)*xstr          +(+1)];
	fnew_d[3*pstr+n*nsite+index]=fold_d[3*pstr+n*nsite+index+(-1)*xstr               ];
	fnew_d[2*pstr+n*nsite+index]=fold_d[2*pstr+n*nsite+index+(-1)*xstr          +(-1)];
	fnew_d[1*pstr+n*nsite+index]=fold_d[1*pstr+n*nsite+index+(-1)*xstr+(-1)*ystr     ];
	
	/* Distributions moving backward in memory. */
	
	fnew_d[10*pstr+n*nsite+index]=fold_d[10*pstr+n*nsite+index+                    (+1)];
	fnew_d[11*pstr+n*nsite+index]=fold_d[11*pstr+n*nsite+index+          (+1)*ystr+(-1)];
	fnew_d[12*pstr+n*nsite+index]=fold_d[12*pstr+n*nsite+index+         +(+1)*ystr     ];
	fnew_d[13*pstr+n*nsite+index]=fold_d[13*pstr+n*nsite+index+         +(+1)*ystr+(+1)];
	fnew_d[14*pstr+n*nsite+index]=fold_d[14*pstr+n*nsite+index+(+1)*xstr+(-1)*ystr     ];
	fnew_d[15*pstr+n*nsite+index]=fold_d[15*pstr+n*nsite+index+(+1)*xstr          +(-1)];
	fnew_d[16*pstr+n*nsite+index]=fold_d[16*pstr+n*nsite+index+(+1)*xstr               ];
	fnew_d[17*pstr+n*nsite+index]=fold_d[17*pstr+n*nsite+index+(+1)*xstr          +(+1)];
	fnew_d[18*pstr+n*nsite+index]=fold_d[18*pstr+n*nsite+index+(+1)*xstr+(+1)*ystr     ];

	
      }


     }


   
   return;
}


/* get linear index from 3d coordinates */
 __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3])
{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}


