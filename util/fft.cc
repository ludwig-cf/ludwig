#include <ostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include <fftw3.h> 
#include <stdio.h>       
#include <string.h>

#define pi 3.141592653589793

#define cmf_in
#define cmf_out

using namespace std;

int Lx=128,Ly=128,Lz=128; 	// box size

int griddata_out  = 1;
int blockdata_out = 0;

double ***op,****ft,***sq, norm;
int imin,imax,jmin,jmax,kmin,kmax;

char inputfilename[250],outputfilename[250],logfilename[250]; 
char line1[250],line2[250],filetrunc[250];

long int timestep;
void allocate();
void structure_factor();
void domain_scale(double * lt);
void output();

fftw_complex *in, *out;
fftw_plan plan;

void fftw_execute(fftw_plan plan);
fftw_plan fftw_plan_dft_3d(int nx, int ny, int nz, fftw_complex *in, fftw_complex *out, int sign, unsigned flags);

int compare (const void* a, const void* b){
     double x = *(double *)a,
            y = *(double *)b;
   return x<=y ? (x<y ? -1:0) : 1;
}


int main(int argc, char* argv[]){

  int a,i,j,k,n;
  double lt;
  string line; 

  allocate();

  sprintf(inputfilename,"%s",argv[1]);
  sprintf(line1,"%s",argv[1]);

  char *pch;
  pch = strtok(line1,".");

  do{
     pch = strtok(NULL,".");
     if(pch==NULL) break;
     sprintf(line2,".%s",pch);
     strcat(line1,line2);
  }while(1);

  sprintf(filetrunc,"%s",line1);
  sprintf(outputfilename,"ft.%s",filetrunc);

  ifstream opinput(inputfilename,ios::in);

  if (!opinput){
     cout << "Can't open order parameter input file" << endl;
     exit(0);
  }

  /* Read order parameter file */
//  cout << "Reading order parameter" << endl; 
  
  /* Skip header */
  for (n=0; n<10; n++){
    getline(opinput,line);
  }


#ifdef cmf_in
    i=-1;
    j=0;
    k=0;
#endif

#ifndef cmf_in
    i=0;
    j=0;
    k=-1;
#endif

  while(!opinput.eof()){

#ifdef cmf_in
     i++;
     if(i==Lx){
	j++;
	i=0;	
	if(j==Ly){
	   k++;
	   j=0;
	   if(k==Lz){break;}   

	}
     }
#endif

#ifndef cmf_in
     k++;
     if(k==Lz){
	j++;
	k=0;	
	if(j==Ly){
	   i++;
	   j=0;
	   if(i==Lx){break;}   

	}
     }
#endif

     opinput >> op[i][j][k];

  }
//  cout << "Order parameter input done" << endl; 


//  cout << "Perform FFT" << endl; 

  a=0;

  for(i=0; i<Lx; i++){
     for(j=0; j<Ly; j++){
	for(k=0; k<Lz; k++){

	   in[a][0]=op[i][j][k];
	   in[a++][1]=0;

	}
     }
  }

  plan=fftw_plan_dft_3d(Lx,Ly,Lz,in,out,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_execute(plan);

  /* Normalization */
  norm=sqrt(Lx*Ly*Lz);
  //norm=1.0;

  a=0;

  for(i=0; i<Lx; i++){
     for(j=0; j<Ly; j++){
	for(k=0; k<Lz; k++){

	   ft[i][j][k][0]=out[a][0]/norm;
	   ft[i][j][k][1]=out[a++][1]/norm;

	}
     }
  }


//  cout << "FFT done" << endl; 

  /* Calulate derived quantities */

  structure_factor();
  domain_scale(&lt);

  cout << lt << endl;

  /*
  cout << "Output" << endl; 
  output();
  cout << "Output done" << endl; 
  */

  fftw_destroy_plan(plan);
  fftw_free(in); 
  fftw_free(out);

}


void allocate(){

  int i,j,k;

  op = new double**[Lx];
  ft = new double***[Lx];
  sq = new double**[Lx];

  for (i=0; i<Lx; i++){

     op[i]=new double*[Ly];
     ft[i]=new double**[Ly];
     sq[i]=new double*[Ly];

     for (j=0; j<Ly; j++){

	op[i][j]=new double[Lz];
	ft[i][j]=new double*[Lz];
	sq[i][j]=new double[Lz];

	for (k=0; k<Lz; k++){
	   ft[i][j][k]=new double[2];
	}
     }
  }

  in = (fftw_complex*) fftw_malloc(Lx*Ly*Lz * sizeof(fftw_complex));
  out = (fftw_complex*) fftw_malloc(Lx*Ly*Lz * sizeof(fftw_complex));

}


void structure_factor(){

  int i,j,k;
  long int n,N;

  N=Lx*Ly*Lz;

  /* Unscattered contribution */
  sq[0][0][0]=ft[0][0][0][0]*ft[0][0][0][0]-ft[0][0][0][1]*ft[0][0][0][1];
  sq[0][0][0]=0.0;

  /* kx & ky & kz != 0 */
  for(i=1; i<Lx; i++){
     for(j=1; j<Ly; j++){
	for(k=1; k<Lz; k++){
	   sq[i][j][k]=ft[i][j][k][0]*ft[Lx-i][Ly-j][Lz-k][0]-ft[i][j][k][1]*ft[Lx-i][Ly-j][Lz-k][1];
	}
     }
  }

  /* kx || ky || kz == 0 */
  for(i=1; i<Lx; i++){
     for(j=1; j<Ly; j++){
	sq[i][j][0]=ft[i][j][0][0]*ft[Lx-i][Ly-j][0][0]-ft[i][j][0][1]*ft[Lx-i][Ly-j][0][1];
     }
  }

  for(j=1; j<Ly; j++){
     for(k=1; k<Lz; k++){
	sq[0][j][k]=ft[0][j][k][0]*ft[0][Ly-j][Lz-k][0]-ft[0][j][k][1]*ft[0][Ly-j][Lz-k][1];
     }
  }

  for(i=1; i<Lx; i++){
     for(k=1; k<Lz; k++){
	sq[i][0][k]=ft[i][0][k][0]*ft[Lx-i][0][Lz-k][0]-ft[i][0][k][1]*ft[Lx-i][0][Lz-k][1];
     }
  }

  for(j=1; j<Ly; j++){
     sq[0][j][0]=ft[0][j][0][0]*ft[0][Ly-j][0][0]-ft[0][j][0][1]*ft[0][Ly-j][0][1];
  }

  for(k=1; k<Lz; k++){
     sq[0][0][k]=ft[0][0][k][0]*ft[0][0][Lz-k][0]-ft[0][0][k][1]*ft[0][0][Lz-k][1];
  }

  for(i=1; i<Lx; i++){
	sq[i][0][0]=ft[i][0][0][0]*ft[Lx-i][0][0][0]-ft[i][0][0][1]*ft[Lx-i][0][0][1];
  }

}

void domain_scale(double * lt){

  int i, j, k;
  int ip, jp, kp; 
  int i1,j1,k1,i2,j2,k2;

  double fi, fj, fk; 
  double num, denom;

  i1=-(int)floor(Lx/2.0);
  j1=-(int)floor(Ly/2.0);
  k1=-(int)floor(Lz/2.0);

  i2=(int)ceil(Lx/2.0);
  j2=(int)ceil(Ly/2.0);
  k2=(int)ceil(Lz/2.0);

  num = 0.0;
  denom = 0.0;

  fi = pow(2.0*pi/Lx,2);
  fj = pow(2.0*pi/Ly,2);
  fk = pow(2.0*pi/Lz,2);

  for(i=i1; i<i2; i++){ 
    for(j=j1; j<j2; j++){
      for(k=k1; k<k2; k++){

	ip=i;	
	jp=j;
	kp=k;

	if(i<0){ip=i+Lx;}
	if(j<0){jp=j+Ly;}
	if(k<0){kp=k+Lz;}

	num   += sq[ip][jp][kp];
	denom += sqrt(i*i*fi+j*j*fj+k*k*fk)*sq[ip][jp][kp];

      }
    }
  }

  * lt = 2.0*pi*num/denom;

}

void output(){

  int i,ip,j,jp,k,kp;
  int i1,j1,k1,i2,j2,k2;

  ofstream ftoutput(outputfilename, ios::out);

  ftoutput.precision(3);
  ftoutput.setf(ios::scientific);

  i1=-(int)floor(Lx/2.0);
  j1=-(int)floor(Ly/2.0);
  k1=-(int)floor(Lz/2.0);

  i2=(int)ceil(Lx/2.0);
  j2=(int)ceil(Ly/2.0);
  k2=(int)ceil(Lz/2.0);


#ifdef cmf_out
  for(k=k1; k<k2; k++){
    for(j=j1; j<j2; j++){
      for(i=i1; i<i2; i++){
#endif

#ifndef cmf_out
  for(i=i1; i<i2; i++){
    for(k=k1; k<k2; k++){
      for(j=j1; j<j2; j++){
#endif

	ip=i;	
	jp=j;
	kp=k;

	if(i<0){ip=i+Lx;}
	if(j<0){jp=j+Ly;}
	if(k<0){kp=k+Lz;}

	if(griddata_out) ftoutput <<  sq[ip][jp][kp] << endl;  
	if(blockdata_out) ftoutput << i << "  " << j << "  " << k << "  " << sq[ip][jp][kp] << endl;  

      }

      if(blockdata_out) ftoutput << endl;

    }
  }

}


