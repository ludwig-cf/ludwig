/*****************************************************************************
 *
 *  util_ellipsoid.c
 *
 *  Utility functions for ellipsoids.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <ctype.h>
#include <string.h>

#include "util.h"
#include "util_ellipsoid.h"

/*****************************************************************************
 *
 *  Orthonormalise a vector b to a given vector a
 *
 *****************************************************************************/

 __host__ __device__ void orthonormalise_vector_b_to_a(double *a, double *b){

  double proj,mag;
  /*projecting b onto a*/
  proj = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  b[0] = b[0] - proj*a[0];
  b[1] = b[1] - proj*a[1];
  b[2] = b[2] - proj*a[2];
  /*Normalising b */
  mag = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
  b[0]=b[0]/mag;
  b[1]=b[1]/mag;
  b[2]=b[2]/mag;
  return ;
}


/*****************************************************************************
 *
 *  Normalise a vector a unit vector
 *
 *****************************************************************************/

__host__ __device__ void normalise_unit_vector(double *a ,const int n){
  
  double magsum = 0.0;
  for(int i = 0; i < n; i++) {magsum+=a[i]*a[i];}
  double mag = sqrt(magsum);
  for(int i = 0; i < n; i++) {a[i]=a[i]/mag;}
  return ; 
}

/*****************************************************************************
 *
 *  matrix_product
 *
 *****************************************************************************/

__host__ __device__
void matrix_product(const double a[3][3], const double b[3][3], double result[3][3]) {

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      result[i][j]=0.0;
      for(int k=0; k < 3; k++){
        result[i][j]+=a[i][k]*b[k][j];
      }   
    }   
  }
  return;
}

/*****************************************************************************
 *
 *  matrix_transpose
 *
 *****************************************************************************/

__host__ __device__
void matrix_transpose(const double a[3][3], double result[3][3]) {

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      result[i][j]=a[j][i];
    }   
  }
  return;
}

/*****************************************************************************
 *
 *  Printing vector on the screen
 *
 ****************************************************************************/
__host__ __device__ void print_vector_onscreen(const double *a, const int n){
  for(int i = 0; i < n; i++) printf("%22.15e, ",a[i]);
  printf("\n");
  return ;
  }
/*****************************************************************************
 *
 *  Printing matrix on the screen
 *
 ****************************************************************************/
__host__ __device__ void print_matrix_onscreen(const double a[3][3]){
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      printf("%22.15e, ",a[i][j]);
    }
    printf("\n");
  }
  return ;
  }
/*****************************************************************************
 *
 *  Multiplying quaternions
 *
 ****************************************************************************/
__host__ __device__ void quaternion_product(const double a[4], const double b[4], double result[4]) {
  result[1]= a[0]*b[1] - a[3]*b[2] + a[2]*b[3] + a[1]*b[0];
  result[2]= a[3]*b[1] + a[0]*b[2] - a[1]*b[3] + a[2]*b[0];
  result[3]=-a[2]*b[1] + a[1]*b[2] + a[0]*b[3] + a[3]*b[0];
  result[0]=-a[1]*b[1] - a[2]*b[2] - a[3]*b[3] + a[0]*b[0];
  return;
  }
/*****************************************************************************
 *
 *  Determining a quaternion from the angular velocity
 *
 ****************************************************************************/
__host__ __device__ void quaternion_from_omega(const double omega[3], const double f, double qbar[4]) {

 double omag;
 omag=sqrt(omega[0]*omega[0]+omega[1]*omega[1]+omega[2]*omega[2]);
 if(omag>1e-12) {
   qbar[0]=cos(omag*f);
   for(int i = 0; i < 3; i++) {qbar[i+1]=sin(omag*f)*(omega[i]/omag);}
 }
 else {
  qbar[0]=1.0;
  for (int i = 1; i < 4; i++) {qbar[i] = 0.0;}
 }
  
  return;
  }
/*****************************************************************************
 *
 *  Rotate to world frame by quaternion
 *
 ****************************************************************************/
__host__ __device__ void rotate_toworldframe_quaternion(const double q[4], const double a[3], double b[3]) {
 double pseudoa[4],qinv[4],binter[4],pseudob[4];
 pseudoa[0]=0.0;
 for(int i = 1; i < 4; i++) {pseudoa[i] = a[i-1];}
 qinv[0]=q[0]; 
 for(int i = 1; i < 4; i++) {qinv[i] =-q[i];}
 quaternion_product(q,pseudoa,binter);
 quaternion_product(binter,qinv,pseudob);
 for(int i = 0; i < 3; i++) {b[i] = pseudob[i+1];}
 return;
 }

/*****************************************************************************
 *
 *  Calculate the moment of inerita tensor from the specified quaternion
 *
 ****************************************************************************/
__host__ __device__ void inertia_tensor_quaternion(const double q[4], const double moment[3], double mI[3][3]) {

  /*Construting the moment of inertia tensor in the principal coordinates*/
  double mIP[3][3];
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      mIP[i][j] = 0.0;
    }
  }
  for(int i = 0; i < 3; i++) {mIP[i][i] = moment[i];}
  /*Rotating it to the body frame, eqn 13*/
  double Mi[3],Mdi[3],Mdd[3][3];
  /*First column transform and fill as first row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][0];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[0][i] = Mdi[i];}
  /*Second column transform and fill as second row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][1];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[1][i] = Mdi[i];}
  /*Third column transform and fill as third row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][2];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[2][i] = Mdi[i];}
  /*Repeat the entire procedure*/
  /*First column transform and fill as first row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][0];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[0][i] = Mdi[i];}
  /*Second column transform and fill as second row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][1];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[1][i] = Mdi[i];}
  /*Third column transform and fill as third row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][2];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[2][i] = Mdi[i];}
  return;
  }
/*****************************************************************************
 *
 *  Rotate to body frame by quaternion
 *
 ****************************************************************************/
__host__ __device__ void rotate_tobodyframe_quaternion(const double q[4], const double a[3], double b[3]) {
 double pseudoa[4],qinv[4],binter[4],pseudob[4];
 pseudoa[0]=0.0;
 for(int i = 1; i < 4; i++) {pseudoa[i] = a[i-1];}
 qinv[0]=q[0]; 
 for(int i = 1; i < 4; i++) {qinv[i] =-q[i];}
 quaternion_product(qinv,pseudoa,binter);
 quaternion_product(binter,q,pseudob);
 for(int i = 0; i < 3; i++) {b[i] = pseudob[i+1];}
 return;
 }

/*****************************************************************************
 *
 *  Determining the quaternions
 *
 ****************************************************************************/
__host__ __device__ void quaternions_from_vectors(const double a[3], const double b[3], double q[4]) {
  double R[3][3];
  rotationmatrix_from_vectors(a,b,R);
  quaternions_from_dcm(R,q);
  normalise_unit_vector(q,4);
  return;
  }
/*****************************************************************************
 *
 *  Determining the rotation matrix from given quaternions
 *
 ****************************************************************************/
__host__ __device__ void rotationmatrix_from_quaternions(const double q[4], double R[3][3]) {
  R[0][0]=1.0-2.0*(q[2]*q[2]+q[3]*q[3]);
  R[1][0]=2.0*(q[1]*q[2]+q[3]*q[0]);
  R[2][0]=2.0*(q[1]*q[3]-q[2]*q[0]);
  R[0][1]=2.0*(q[2]*q[1]-q[3]*q[0]);
  R[1][1]=1.0-2.0*(q[3]*q[3]+q[1]*q[1]);
  R[2][1]=2.0*(q[2]*q[3]+q[1]*q[0]);
  R[0][2]=2.0*(q[3]*q[1]+q[2]*q[0]);
  R[1][2]=2.0*(q[3]*q[2]-q[1]*q[0]);
  R[2][2]=1.0-2.0*(q[1]*q[1]+q[2]*q[2]);
  return;
  }
/*****************************************************************************
 *
 *  Determining the rotation matrix from given two orientation vectors
 *
 ****************************************************************************/
__host__ __device__ void rotationmatrix_from_vectors(const double a[3], const double b[3], double R[3][3]) {
  double c[3];
  double inertialv1[3]={1.0,0.0,0.0};
  double inertialv2[3]={0.0,1.0,0.0};
  double inertialv3[3]={0.0,0.0,1.0};

  cross_product(a,b,c);
  normalise_unit_vector(c,3);
  /*Calculating the DCM matrix*/
  R[0][0]=dot_product(inertialv1,a);
  R[0][1]=dot_product(inertialv1,b);
  R[0][2]=dot_product(inertialv1,c);
  R[1][0]=dot_product(inertialv2,a);
  R[1][1]=dot_product(inertialv2,b);
  R[1][2]=dot_product(inertialv2,c);
  R[2][0]=dot_product(inertialv3,a);
  R[2][1]=dot_product(inertialv3,b);
  R[2][2]=dot_product(inertialv3,c);
  return;
  }

/*****************************************************************************
 *
 *  Determining quaternions from Euler angles
 *
 ****************************************************************************/
__host__ __device__ void quaternions_from_eulerangles(const double phi, const double theta, const double psi, double q[4]){

  q[0]=cos(theta/2.0)*cos((phi+psi)/2.0);
  q[1]=sin(theta/2.0)*cos((phi-psi)/2.0);
  q[2]=sin(theta/2.0)*sin((phi-psi)/2.0);
  q[3]=cos(theta/2.0)*sin((phi+psi)/2.0);
  return;
  }

/*****************************************************************************
 *
 *  Determining Euler angles from the rotation matrix
 *
 ****************************************************************************/
__host__ __device__ void eulerangles_from_dcm(const double R[3][3], double *phi, double *theta, double *psi){

  if(R[2][2]==1.0){
    *theta=0.0;
    *phi=0.0;
    *psi=atan2(R[0][1],R[0][0]);
  }
  else if(R[2][2]==-1.0){
    *theta=M_PI;
    *psi=0.0;
    *phi=atan2(R[0][1],R[0][0]);
  }
  else {
    *phi=atan2(R[2][0],-R[2][1]);
    *theta=acos(R[2][2]);
    *psi=atan2(R[0][2],R[1][2]);
  }

  return;
  }

/*****************************************************************************
 *
 *  Determining quaternions from the rotation matrix
 *
 ****************************************************************************/
__host__ __device__ void quaternions_from_dcm(const double R[3][3], double q[4]){
  
  double trq=R[0][0]+R[1][1]+R[2][2];
  q[0]=0.5*sqrt(1.0+trq);
  if(fabs(q[0])<1e-12){
    q[1]=sqrt((1.0+R[0][0])/2.0);
    q[2]=R[0][1]/(2.0*q[1]);
    q[3]=R[1][2]/(2.0*q[2]);
  }
  else {
  q[1]=(R[1][2]-R[2][1])/(4.0*q[0]);
  q[2]=(R[2][0]-R[0][2])/(4.0*q[0]);
  q[3]=(R[0][1]-R[1][0])/(4.0*q[0]);
  }

  return;
  }

/*****************************************************************************
 *
 *  Rotating a vector by a matrix
 *
 ****************************************************************************/
__host__ __device__ void rotate_byRmatrix(const double R[3][3], const double x[3], double xcap[3]    ){
  for(int i=0; i < 3; i++){
    xcap[i]=0.0;
    for(int j = 0; j < 3; j++) {
    xcap[i]+=R[i][j]*x[j];
    }
  }
  return ;
  }

/*****************************************************************************
 *
 *  Copy a vector to another
 *
 ****************************************************************************/
__host__ __device__ void copy_vectortovector(const double a[3], double b[3], int n){
  for(int i=0; i < n; i++){
    b[i]=a[i];
  }
  return ;
  }

/*****************************************************************************
*
*  Jeffery's predictions for a spheroid
*
*****************************************************************************/
__host__ __device__ void Jeffery_omega_predicted(double const r, double const quater[4], double const gammadot, double opred[3]) {

  double beta;
  double phi1,the1;
  double v1[3]={1.0,0.0,0.0};
  double v2[3]={0.0,1.0,0.0};
  double v3[3]={0.0,0.0,1.0};
  double p[3],pdot[3];
  double pdoty,phiar;
  double pcpdot[3];
  double pxj,pyj,pzj,pxdotj,pydotj,pzdotj;
  double op[3]={0.0,0.0,0.0};
  double omp;

  beta=(r*r-1.0)/(r*r+1.0);
  /*Determining p, the orientation of the long axis*/
  rotate_tobodyframe_quaternion(quater,v1,p);
  /*Determing pdot in Guazzeli's convention*/
  pdoty=p[0]*v2[0]+p[1]*v2[1]+p[2]*v2[2];
  phiar=(p[0]-pdoty*v2[0]*v2[0])*v3[0]+
        (p[1]-pdoty*v2[1]*v2[1])*v3[1]+
        (p[2]-pdoty*v2[2]*v2[2])*v3[2];
  the1=acos(-pdoty);
  phi1=acos(phiar);
  pxj= sin(the1)*sin(phi1);
  pyj= sin(the1)*cos(phi1);
  pzj=-cos(the1);
  pxdotj= gammadot*((beta+1.0)*pyj/2.0-beta*pxj*pxj*pyj);
  pydotj= gammadot*((beta-1.0)*pxj/2.0-beta*pyj*pyj*pxj);
  pzdotj=-gammadot*beta*pxj*pyj*pzj;
  /*Determing pdot in Ludwig's convention*/
  pdot[0]=pxdotj;
  pdot[1]=-pzdotj;
  pdot[2]=pydotj;
  /*Determing the spinning velocity*/
  op[1]= gammadot/2.0;
  omp=dot_product(op,p);
  /*Determining the tumbling velocity*/
  cross_product(p,pdot,pcpdot);
  /*Determining the total angular velocity*/
  opred[0]=omp*p[0]+pcpdot[0];
  opred[1]=omp*p[1]+pcpdot[1];
  opred[2]=omp*p[2]+pcpdot[2];

  return ;
  }


/*****************************************************************************/
