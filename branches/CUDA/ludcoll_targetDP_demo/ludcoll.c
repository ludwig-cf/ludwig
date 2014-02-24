/*
 * ludcoll.c: Ludwig collision benchmark main file. 
 * Alan Gray, November 2013
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "ludcoll.h"


/* Constants*/

TARGET_CONST int N_cd[3];
TARGET_CONST int Nall_cd[3];
TARGET_CONST int nhalo_cd;
TARGET_CONST int nsites_cd;
TARGET_CONST int nop_cd;
TARGET_CONST double rtau_shear_d;
TARGET_CONST double rtau_bulk_d;
TARGET_CONST double rtau_d[NVEL];
TARGET_CONST double wv_cd[NVEL];
TARGET_CONST double ma_cd[NVEL][NVEL];
TARGET_CONST double mi_cd[NVEL][NVEL];
TARGET_CONST double q_cd[NVEL][3][3];
TARGET_CONST int cv_cd[NVEL][3];
TARGET_CONST double d_cd[3][3];
TARGET_CONST double a_d;
TARGET_CONST double b_d;
TARGET_CONST double kappa_d;
TARGET_CONST double rtau2_d;
TARGET_CONST double rcs2_d;
TARGET_CONST double force_global_cd[3];


/* pointers for lattice data arrays */

double * f_d;
double * ftmp_d;
double * force_d;
double * velocity_d;
double * phi_site_d;
double * grad_phi_site_d;
double * delsq_phi_site_d;


char *siteMask;



/* workspace for benchmark setup and validation */

double fieldtemp[NDATA], fieldtemp2[NDATA], fieldtemp3[NDATA];

int main() {

  double t1, t2;
  int i,j,k;

  /* lattice parameters */
  int nhalo=1;

  int N[3],Nall[3];
  N[X]=ND;N[Y]=ND;N[Z]=ND;
  Nall[X]=N[X]+2*nhalo;  Nall[Y]=N[Y]+2*nhalo;  Nall[Z]=N[Z]+2*nhalo;

  int nsites=Nall[X]*Nall[Y]*Nall[Z];

  int nFieldsDist=NVEL*NDIST;
  //int ndata=nsites*nFieldsDist;



  /* Allocate memory on target */
  targetCalloc((void **) &f_d, nsites*nFieldsDist*sizeof(double));
  targetCalloc((void **) &ftmp_d, nsites*nFieldsDist*sizeof(double));
  targetCalloc((void **) &phi_site_d, nsites*sizeof(double));
  targetCalloc((void **) &delsq_phi_site_d, nsites*sizeof(double));
  targetCalloc((void **) &grad_phi_site_d, nsites*3*sizeof(double));
  targetCalloc((void **) &force_d, nsites*3*sizeof(double));
  targetCalloc((void **) &velocity_d, nsites*3*sizeof(double));
  checkTargetError("malloc");

  
  //set up site mask
  siteMask = (char*) calloc(nsites,sizeof(char));
  if(!siteMask){
    printf("siteMask malloc failed\n");
    exit(1);
  }

  // set all non-halo sites to 1
  for (i=nhalo;i<(Nall[X]-nhalo);i++)
    for (j=nhalo;j<(Nall[Y]-nhalo);j++)
      for (k=nhalo;k<(Nall[Z]-nhalo);k++)
	siteMask[i*Nall[Z]*Nall[Y]+j*Nall[Z]+k]=1;





  /* read input data from disk and copy to target */
  FILE *fileptr;
  int datasize;

  readData(nsites,nFieldsDist,"f_input.bin",f_d);
  readData(nsites,1,"phi_site_input.bin",phi_site_d);
  readData(nsites,3,"grad_phi_site_input.bin",grad_phi_site_d);
  readData(nsites,1,"delsq_phi_site_input.bin",delsq_phi_site_d);
  readData(nsites,3,"force_input.bin",force_d);

  checkTargetError("memcpy");


  printf("Setting up constants...\n");


  double rtau_shear,rtau_bulk,rtau_[NVEL],wv[NVEL],ma_[NVEL*NVEL],mi_[NVEL*NVEL],
    d_[3*3],q_[NVEL*3*3],rtau2,rcs2,a_,b_,kappa_;
  int cv[NVEL*3];
  
  
  printf("Reading constants_input.bin...\n");
  fileptr=fopen("constants_input.bin","rb");
  printf("...Done\n");
  fread(&rtau_shear, sizeof(double), 1, fileptr);
  fread(&rtau_bulk, sizeof(double), 1, fileptr);
  fread(rtau_, NVEL*sizeof(double), 1, fileptr);
  fread(wv, NVEL*sizeof(double), 1, fileptr);
  fread(ma_, NVEL*NVEL*sizeof(double), 1, fileptr);
  fread(mi_, NVEL*NVEL*sizeof(double), 1, fileptr);
  fread(d_, 3*3*sizeof(double), 1, fileptr);
  fread(cv, NVEL*3*sizeof(int), 1, fileptr);
  fread(q_, NVEL*3*3*sizeof(double), 1, fileptr);
  fread(&rtau2, sizeof(double), 1, fileptr);
  fread(&rcs2, sizeof(double), 1, fileptr);
  fread(&a_, sizeof(double), 1, fileptr);
  fread(&b_, sizeof(double), 1, fileptr);
  fread(&kappa_, sizeof(double), 1, fileptr);
  fclose(fileptr);


  copyConstantDoubleToTarget(&rtau_shear_d, &rtau_shear, sizeof(double)); 
  copyConstantDoubleToTarget(&rtau_bulk_d, &rtau_bulk, sizeof(double));
  copyConstantDouble1DArrayToTarget(rtau_d, rtau_, NVEL*sizeof(double)); 
  copyConstantDouble1DArrayToTarget(wv_cd, wv, NVEL*sizeof(double));
  copyConstantDouble2DArrayToTarget( (double **) ma_cd, ma_, NVEL*NVEL*sizeof(double));
  copyConstantDouble2DArrayToTarget((double **) mi_cd, mi_, NVEL*NVEL*sizeof(double));
  copyConstantDouble2DArrayToTarget((double **) d_cd, d_, 3*3*sizeof(double));
  copyConstantInt2DArrayToTarget((int **) cv_cd,cv, NVEL*3*sizeof(int)); 
  copyConstantDouble3DArrayToTarget((double ***) q_cd, q_, NVEL*3*3*sizeof(double)); 
  copyConstantDoubleToTarget(&rtau2_d, &rtau2, sizeof(double));
  copyConstantDoubleToTarget(&rcs2_d, &rcs2, sizeof(double));
  copyConstantDoubleToTarget(&a_d, &a_, sizeof(double));
  copyConstantDoubleToTarget(&b_d, &b_, sizeof(double));
  copyConstantDoubleToTarget(&kappa_d, &kappa_, sizeof(double));
  copyConstantInt1DArrayToTarget(N_cd,N, 3*sizeof(int)); 
  copyConstantInt1DArrayToTarget(Nall_cd,Nall, 3*sizeof(int)); 
  copyConstantIntToTarget(&nhalo_cd,&nhalo, sizeof(int)); 
  copyConstantIntToTarget(&nsites_cd,&nsites, sizeof(int)); 
  double force_global[3]; 
  force_global[0]=0.; force_global[1]=0.; force_global[2]=0.; 
  copyConstantDouble1DArrayToTarget(force_global_cd,force_global, 3*sizeof(double)); 

  checkTargetError("constants");
  printf("... Done\n");
  
  

  /* swap input and output f pointers on targe */
  double *tmpptr=ftmp_d;
  ftmp_d=f_d;
  f_d=tmpptr;





  /* execute main kernel */

  printf("Starting Kernel Launch...\n");
  t1=omp_get_wtime();

  //  collision TARGET_LAUNCH(N[X]*N[Y]*N[Z]) (f_d, ftmp_d,
  collision TARGET_LAUNCH(nsites) (f_d, ftmp_d,
					   phi_site_d,
					   grad_phi_site_d,
					   delsq_phi_site_d,
					   force_d,
					   velocity_d);
  
  syncTarget();

  t2=omp_get_wtime();

  printf("... Done\n");
  checkTargetError("collision");

  printf("Time: %1.16e s\n",t2-t1);

  /* collect results from target */

  datasize=nsites*nFieldsDist*sizeof(double);
  //copyFromTarget(fieldtemp, f_d, datasize);

  copyFromTargetMasked(fieldtemp, f_d, nsites,nFieldsDist,siteMask);

  fileptr=fopen("f_output.bin","wb");
  fwrite(fieldtemp, datasize, 1, fileptr);
  fclose(fileptr);

  datasize=NALL*NALL*NALL*3*sizeof(double);
  //copyFromTarget(fieldtemp, velocity_d, datasize);
  copyFromTargetMasked(fieldtemp, velocity_d, nsites,3,siteMask);

  fileptr=fopen("v_output.bin","wb");
      fwrite(fieldtemp, datasize, 1, fileptr);
  fclose(fileptr);


  /* validate results */

  printf("\n\nValidating distribution:\n");


  datasize=nsites*nFieldsDist*sizeof(double);
  fileptr=fopen("f_output.bin","rb");
  fread(fieldtemp2, datasize, 1, fileptr);
  fclose(fileptr);
  fileptr=fopen("f_outputref.bin","rb");
  fread(fieldtemp3, datasize, 1, fileptr);
  fclose(fileptr);
  compData(fieldtemp2, fieldtemp3,nsites*nFieldsDist);

  printf("\n\nValidating velocity:\n");
  datasize=NALL*NALL*NALL*3*sizeof(double);
  fileptr=fopen("v_output.bin","rb");
  fread(fieldtemp2, datasize, 1, fileptr);
  fclose(fileptr);
  fileptr=fopen("v_outputref.bin","rb");
  fread(fieldtemp3, datasize, 1, fileptr);
  fclose(fileptr);
  compData(fieldtemp2, fieldtemp3,NALL*NALL*NALL*3);



  /* free data */

  targetFree(f_d);
  targetFree(ftmp_d);
  targetFree(phi_site_d);
  targetFree(delsq_phi_site_d);
  targetFree(grad_phi_site_d);
  targetFree(force_d);
  targetFree(velocity_d);

  free(siteMask);

  return 0;
}



/* entry point function for execution on target */
TARGET_ENTRY void collision( double* __restrict__ f_d, 
			     const double* __restrict__ ftmp_d, 
			     const double* __restrict__ phi_site_d,		
			     const double* __restrict__ grad_phi_site_d,	
			     const double* __restrict__ delsq_phi_site_d,	
			     const double* __restrict__ force_d, 
			     double* __restrict__ velocity_d) 
{
  

  
  /* thread parallel code section */
  int tpIndex;
  //TARGET_TLP(tpIndex,N_cd[X]*N_cd[Y]*N_cd[Z]) 
        TARGET_TLP(tpIndex,Nall_cd[X]*Nall_cd[Y]*Nall_cd[Z])
    {
      
      int ii,jj,kk, siteIndex;
      
  
      /* get latttice site index from thread parallel index */
      //get_coords_from_index(&ii,&jj,&kk,tpIndex,N_cd);      
      //siteIndex = get_linear_index(ii+nhalo_cd,jj+nhalo_cd,kk+nhalo_cd,Nall_cd);
      siteIndex=tpIndex;

      /* execute collision kernel for this thread (lattice site) */
      collision_site
	(f_d, ftmp_d,
	 phi_site_d,
	 grad_phi_site_d,
	 delsq_phi_site_d,
	 force_d,
	 velocity_d, siteIndex);
            
      
    }
  
  return;
  
}


/* benchmark data input */
void readData(int nsites,int nfields, char* filename, double* targetData){

  FILE *fileptr;
  
  printf("Reading %s ...\n",filename);
  fileptr=fopen(filename,"rb");
  fread(fieldtemp, nsites*nfields*sizeof(double), 1, fileptr);
  fclose(fileptr);
  printf("... Done\n");

  //copyToTarget(targetData,fieldtemp, nsites*nfields*sizeof(double));

  copyToTargetMasked(targetData,fieldtemp,nsites,nfields,siteMask);

}


/* benchmark output validation */
void compData(double *data, double *dataRef, int size){

  
  double maxdiff=0., diff;
  
  int i;

  for (i=0; i<size; i++){    

    if (data[i]!=data[i] || dataRef[i]!=dataRef[i])
      {
	printf("encountered NAN at index %d:\n",i);
	exit(1);
      }



    diff = fabs(data[i]-dataRef[i]);
    
    if ( diff > TOLERANCE ){
      printf("data mismatch:\n");
      printf("data[%d]=%1.16e, dataRef[%d]=%1.16e\n",i,data[i],i,dataRef[i]);
      
      exit(1);
    }
    
    if (diff > maxdiff) maxdiff=diff;
    

  }


  printf("Data matches reference within tolerance of %e\n", TOLERANCE);
  printf("Max diff = %e\n", maxdiff);



}




/* get 3d coordinates from linear index */
TARGET void get_coords_from_index(int *ii,int *jj,int *kk,int index,int N[3])

{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;
  
  *ii = index/xfac;
  *jj = ((index-xfac*(*ii))/yfac);
  *kk = (index-(*ii)*xfac-(*jj)*yfac);

  return;

}


/* get linear index from 3d coordinates */
 TARGET int get_linear_index(int ii,int jj,int kk,int N[3])
{
  
  int yfac = N[Z];
  int xfac = N[Y]*yfac;

  return ii*xfac + jj*yfac + kk;

}

