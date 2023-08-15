/*****************************************************************************
 *
 *  util_ellipsoid.c
 *
 *  Utility functions for ellipsoids.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Sumesh Thampi
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <math.h>

#include "util.h"
#include "util_vector.h"
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
 *  matrix_product
 *
 *****************************************************************************/

void matrix_product(const double a[3][3], const double b[3][3],
		    double result[3][3]) {

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result[i][j] = 0.0;
      for (int k = 0; k < 3; k++) {
        result[i][j] += a[i][k]*b[k][j];
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

void matrix_transpose(const double a[3][3], double result[3][3]) {

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      result[i][j] = a[j][i];
    }
  }
  return;
}

/*****************************************************************************
 *
 *  Determining a quaternion from the angular velocity
 *
 *  See, e.g., Zhao and Wachem Acta Mech 224 3091--3109 (2013)
 *  Eq 49
 *  The factor (1/2) is replaced by the general fraction f.
&
 ****************************************************************************/

__host__ __device__ void quaternion_from_omega(const double omega[3],
					       const double f, double qbar[4]) {

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
 *  Calculate the moment of inerita tensor from the specified quaternion
 *
 *  The moment of interia tensor in the lab frame is moment[3].
 *  This must be rotated by the quaternion.
 *
 *  See Zhao and Wachem Acta Mech 224, 2331--2358 (2013) [different!]
 *  The full operation is Eq 22.
 *
 *  The result mI_ab is the inertia tensor in the rotated frame.
 *
 ****************************************************************************/
__host__ __device__ void inertia_tensor_quaternion(const double q[4],
						   const double moment[3], double mI[3][3]) {

  /*Construting the moment of inertia tensor in the principal coordinates*/
  double mIP[3][3];
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      mIP[i][j] = 0.0;
    }
  }
  /* Set the diagonal elements */
  for (int i = 0; i < 3; i++) {
    mIP[i][i] = moment[i];
  }
  /* Rotating it to the body frame */
  double Mi[3],Mdi[3],Mdd[3][3];
  /*First column transform and fill as first row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][0];}
  util_q4_rotate_vector(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[0][i] = Mdi[i];}
  /*Second column transform and fill as second row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][1];}
  util_q4_rotate_vector(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[1][i] = Mdi[i];}
  /*Third column transform and fill as third row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][2];}
  util_q4_rotate_vector(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[2][i] = Mdi[i];}

  /*Repeat the entire procedure*/
  /*First column transform and fill as first row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][0];}
  util_q4_rotate_vector(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[0][i] = Mdi[i];}
  /*Second column transform and fill as second row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][1];}
  util_q4_rotate_vector(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[1][i] = Mdi[i];}
  /*Third column transform and fill as third row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][2];}
  util_q4_rotate_vector(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[2][i] = Mdi[i];}
  return;
}

/*****************************************************************************
 *
 *  util_q4_rotate_vector
 *
 *  Rotate vector a[3] by the quaternion to give vector b.
 *
 *  E.g., Rapaport Eq 8.2.8 with q = [q_0, q = (q_1, q_2, q_3)]
 *  that is, b = (2q_0^2 - 1)a + 2(q.a)q + 2q_0 q x a
 *
 ****************************************************************************/

void util_q4_rotate_vector(const double q[4], const double a[3], double b[3]) {

  double q0    = q[0];
  double qdota = dot_product(q + 1, a);
  double qxb[3] = {0};

  cross_product(q + 1, a, qxb);

  for (int i = 0; i < 3; i++) {
    b[i] = (2.0*q0*q0 - 1.0)*a[i] + 2.0*qdota*q[i+1] + 2.0*q0*qxb[i];
  }

  return;
}

/*****************************************************************************
 *
 *  util_q4_from_euler_angles
 *
 *  The Euler angles are:
 *    phi   a rotation around z-axis
 *    theta a rotation around new x' axis
 *    psi   a rotation around new z'' axis
 *
 *  Returns a unit quaternion describing the equivalent rotation.
 *  See e.g., Rapaport "The Art of Molecular Dynamics" Chapter 8.2.
 *  Note Rapaport has (q1, q2, q3, q4). We have (q1, q2, q3, q0).
 *
 *****************************************************************************/

int util_q4_from_euler_angles(double phi, double theta, double psi,
			      double q[4]) {

  theta = 0.5*theta;
  q[1] = sin(theta)*cos(0.5*(phi - psi));
  q[2] = sin(theta)*sin(0.5*(phi - psi));
  q[3] = cos(theta)*sin(0.5*(phi + psi));
  q[0] = cos(theta)*cos(0.5*(phi + psi));

  return 0;
}

/*****************************************************************************
 *
 *  util_q4_to_euler_angles
 *
 *  This produces Euler angles from a unit quaternion.
 *  See note above.
 *
 *****************************************************************************/

int util_q4_to_euler_angles(const double q[4], double * phi, double * theta,
			    double * psi) {
  assert(phi);
  assert(theta);
  assert(psi);

  int ifail = 0;
  double q1 = q[1];
  double q2 = q[2];
  double q3 = q[3];
  double q4 = q[0];

  /* Check sign of 1 - q1^2 - q2^2 */

  if (1.0 - q1*q1 - q2*q2 < 0.0) {
    ifail = -1;
  }
  else {

    /* sin(theta) and cos(theta) */
    double st = 2.0*sqrt((q1*q1 + q2*q2)*(1.0 - q1*q1 - q2*q2));
    double ct = 1.0 - 2.0*(q1*q1 + q2*q2);

    if (st > DBL_EPSILON) {
      double sp = 2.0*(q1*q3 + q2*q4)/st;
      double cp = 2.0*(q1*q4 - q2*q3)/st;
      double ss = 2.0*(q1*q3 - q2*q4)/st;
      double cs = 2.0*(q1*q4 + q2*q3)/st;
      *phi   = atan2(sp, cp);
      *theta = atan2(st, ct);
      *psi   = atan2(ss, cs);
    }
    else {
      double st1 = sqrt(q1*q1 + q2*q2);
      double ct1 = sqrt(q3*q3 + q4*q4);
      double sp1 = sqrt(q2*q2 + q3*q3);
      double cp1 = sqrt(q1*q1 + q4*q4);
      *phi   = 2.0*atan2(sp1, cp1);
      *theta = 2.0*atan2(st1, ct1);
      *psi   = 0.0;
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_q4_product
 *
 *  Quaternion product (is not commutative, recall, ...)
 *
 ****************************************************************************/

void util_q4_product(const double a[4], const double b[4], double c[4]) {

  c[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
  c[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
  c[2] = a[0]*b[2] + a[2]*b[0] - a[1]*b[3] + a[3]*b[1];
  c[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1];

  return;
}

/*****************************************************************************
 *
 *  Far field predictions of Mitchell and Spagnolie
 *  PENDING CONFIRMATION
 *
 *****************************************************************************/
__host__ __device__ void ellipsoid_nearwall_predicted(double const elabc[3], double const h, double const quater[4], double Upred[3], double opred[3]) {

  double ecc,ecc2,ecc3,ecc4,K;
  double termn,termd,h2,h3,h4;
  double Xa,Ya,Xad,Yad,XamYa,XaYa;
  double phi, theta, psi;
  double beta = 0.0, phiinspag=0.0;
  double cbeta,sbeta,cphi,sphi,cphi2,sphi2,ctheta,stheta,c2theta,s2theta,ctheta2,stheta2;
  double Ux,Uy,Uz,Ox,Oy,Oz;
  double dfact,dfacto,ela,r;
  double term1,term2,term3,term4,term5,term6;
  PI_DOUBLE(pi);
  ela=elabc[0];
  r = elabc[0]/elabc[1];
  dfact = (6.0*pi*0.1*ela)/0.01;
  dfacto = (6.0*pi*0.1*ela*ela)/0.01;
  cbeta=cos(beta);
  sbeta=sin(beta);
  cphi = cos(phiinspag);
  sphi = sin(phiinspag);
  cphi2 = cphi*cphi;
  sphi2 = sphi*sphi;
  ecc=sqrt(1.0 - 1.0/(r*r));
  ecc2=ecc*ecc;
  ecc3=ecc2*ecc;
  ecc4=ecc2*ecc2;
  K=log((1.0+ecc)/(1.0-ecc));
  Xad = -6.0*ecc + 3.0*(1.0+ecc2)*K;
  Xa = 8.0*ecc3/Xad;
  Yad = 6.0*ecc + (9.0*ecc2-3.0)*K;
  Ya = 16.0*ecc2/Yad;
  XamYa = Xa - Ya;
  XaYa = Xa*Ya;
  util_q4_to_euler_angles(quater, &phi, &theta, &psi);
  ctheta = cos(phi);
  stheta = cos(phi);
  c2theta = cos(2*phi);
  s2theta = sin(2*phi);
  ctheta2 = ctheta*ctheta;
  stheta2 = stheta*stheta;
  h2 = h*h;
  h3 = h2*h;
  h4 = h3*h;
  /*Ux*/
  termn = (2.0*cbeta - (1+c2theta)*cbeta*cphi2+cphi*sbeta*s2theta)*XamYa+2.0*Ya*cbeta;
  termd = 2.0*XaYa;
  Ux = termn/termd;
  termn = 9.0*cbeta;
  termd = 16.0*h;
  Ux = Ux - termn/termd;
  termn = 4.0*ecc2*cphi*sbeta*s2theta+(2.0*ecc2*(c2theta+1.0)*cphi2+18.0*ecc2*ctheta2-24.0*ecc2+16.0)*cbeta;
  termd = 128.0*h3;
  Ux = Ux + termn/termd;
  /*Uy*/
  termn = sphi*(sbeta*s2theta-(1.0+c2theta)*cbeta*cphi)*XamYa;
  termd = 2.0*XaYa;
  Uy = termn/termd;
  termn = ecc2*sphi*(2.0*sbeta*s2theta+(c2theta+1.0)*cbeta*cphi);
  termd = 64.0*h3;
  Uy = Uy + termn/termd;
  /*Uz*/
  termn = 9.0*sbeta;
  termd = 8.0*h;
  Uz = termn/termd;
  termn = 2.0*Ya*sbeta + (cbeta*cphi*s2theta+(c2theta+1.0)*sbeta)*XamYa;
  termd = 2.0*XaYa;
  Uz = Uz - termn/termd;
  termn = ecc2*cbeta*cphi*s2theta - (14.0*ecc2*stheta2+6.0*ecc2-16.0)*sbeta;
  termd = 32.0*h3;
  Uz = Uz - termn/termd;
  /*Ox*/
  termn = 9.0*ecc2*sphi*((2.0-2.0*stheta2)*cbeta*cphi-3.0*sbeta*s2theta);
  termd = 64.0*(2.0-ecc2)*h2;
  Ox = termn/termd;
  termn = 6.0*cbeta*cphi*(6.0*ecc2*stheta2*stheta2-8.0*stheta2+2.0*(4.0-ecc2-2.0*ecc2*s2theta));
  termd = 128.0*(ecc2-2.0)*h4;
  Ox = Ox + termn/termd;
  termn = 3.0*ecc2*sbeta*s2theta*sphi*(12.0*ecc2*stheta2+8.0*ecc2-18.0);
  termd = 128.0*(ecc2 - 2.0)*h4;
  Ox = Ox + termn/termd;
  /*Oy*/
  termn = 27.0*ecc2*sbeta*cphi*s2theta + 9.0*ecc2*cbeta*(2.0-4.0*ctheta2+(2*ctheta2)*sphi2);
  termd = 64.0*(2.0-ecc2)*h2;
  Oy = termn/termd;
  termn = 3.0*ecc2*sbeta*cphi*s2theta*(12.0*ecc2*ctheta2-20.0*ecc2+18.0);
  termd = 128.0*(2.0 - ecc2)*h4;
  Oy = Oy - termn/termd;
  term1 = ecc4*ctheta2*ctheta2*(6.0*sphi2-9.0);
  term2 = 2.0*ecc2*ctheta2*sphi2*(4.0+ecc2-5.0*ecc2);
  term3 = 0;
  term4 = ecc2*ctheta2*(12.0-17.0*ecc2);
  term5 = 8.0*ecc4;
  term6 = 10.0*ecc2 - 4.0;
  termn = 6.0*cbeta*(term1 + term2 + term3 - term4 - term5 + term6);
  termd = 128.0*(2.0-ecc2)*h4;
  Oy = Oy - termn/termd;
  /*Oz*/
  termn = -9.0*ecc2;
  termd = 64.0*(2.0 - ecc2)*h2;
  Oz = termn/termd;
  termn = 3.0*ecc2*(6.0*ecc2*ctheta2-12.0*ecc2+8.0);
  termd = 256.0*(2.0-ecc2)*h4;
  Oz = Oz + termn/termd;
  Oz = cbeta*sphi*s2theta*Oz;
  Upred[0]=Ux/dfact;
  Upred[1]=Uy/dfact;
  Upred[2]=Uz/dfact;
  opred[0]=Ox/dfacto;
  opred[1]=Oy/dfacto;
  opred[2]=Oz/dfacto;
return;
}

/*****************************************************************************
 *
 *  Jeffery's predictions for a spheroid
 *
 *  Originally from Jeffrey (1922).
 *
 *  See, e.g., E. Guazzelli, A Physical Introduction to Suspension Dynamics
 *  Cambridge (2012) Chapter 3.
 *
 *  r is the aspect ratio of the ellipsoid
 *  q is the orientation (quaternion)
 *  gammadot is the shear rate
 *  opred[3]    predicted angular velocity
 *  angpred[2]  (theta_1, phi_1) See Figure 3.13.
 *
 *****************************************************************************/
__host__ __device__ void Jeffery_omega_predicted(double const r, double const quater[4], double const gammadot, double opred[3], double angpred[2]) {

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
  util_q4_rotate_vector(quater,v1,p);

  /*Determine pdot in Guazzeli's convention*/
  pdoty=p[0]*v2[0]+p[1]*v2[1]+p[2]*v2[2];
  phiar=(p[0]-pdoty*v2[0])*v3[0]+
        (p[1]-pdoty*v2[1])*v3[1]+
        (p[2]-pdoty*v2[2])*v3[2];
  the1=acos(-pdoty);
  phi1=acos(phiar);

  angpred[0]=phi1;
  angpred[1]=the1;

  pxj= sin(the1)*sin(phi1);
  pyj= sin(the1)*cos(phi1);
  pzj=-cos(the1);
  pxdotj= gammadot*((beta+1.0)*pyj/2.0-beta*pxj*pxj*pyj);
  pydotj= gammadot*((beta-1.0)*pxj/2.0-beta*pyj*pyj*pxj);
  pzdotj=-gammadot*beta*pxj*pyj*pzj;
  /*Determine pdot in Ludwig's convention*/
  pdot[0]=pxdotj;
  pdot[1]=-pzdotj;
  pdot[2]=pydotj;
  /*Determine the spinning velocity*/
  op[1]= gammadot/2.0;
  omp=dot_product(op,p);
  /*Determining the tumbling velocity*/
  cross_product(p,pdot,pcpdot);

  /*Determining the total angular velocity*/
  for (int i = 0; i < 3; i++) {
    opred[i] = omp*p[i]+pcpdot[i];
  }

  return;
}

/*****************************************************************************
 *
 *  Calculating Euler angles from vectors
 *
 *  If a and b are not at right angles, b is adjusted to be so.
 *
 *****************************************************************************/
__host__ __device__ void euler_from_vectors(double a[3], double b[3], double *euler) {

  double c[3],r[3][3];
  util_vector_normalise(3, a);
  orthonormalise_vector_b_to_a(a, b);
  cross_product(a,b,c);
  dcm_from_vectors(a,b,c,r);
  euler_from_dcm((const double (*)[])r,&euler[0],&euler[1],&euler[2]);

 return;
}
/*****************************************************************************
*
*  Calculating Euler angles from Direction Cosine Matrix
*
*****************************************************************************/
__host__ __device__ void euler_from_dcm(double const r[3][3], double *phi, double *theta, double *psi) {

  *theta=acos(r[2][2]);
  if(fabs(fabs(r[2][2])-1.0)>1e-12) {
    *phi=atan2(r[2][0],-r[2][1]);
    *psi=atan2(r[0][2],r[1][2]);
  }
  else {
    *phi=atan2(r[0][1],r[0][0]);
    *psi = 0.0;
  }
return;
}
/*****************************************************************************
*
*  Calculating Direction Cosine Matrix from a given set of orientation vectors
*
*****************************************************************************/
__host__ __device__ void dcm_from_vectors(double const a[3], double const b[3], double const c[3], double r[3][3]) {

double v1[3]={1.0,0.0,0.0};
double v2[3]={0.0,1.0,0.0};
double v3[3]={0.0,0.0,1.0};

r[0][0] = dot_product(v1,a);
r[1][0] = dot_product(v1,b);
r[2][0] = dot_product(v1,c);
r[0][1] = dot_product(v2,a);
r[1][1] = dot_product(v2,b);
r[2][1] = dot_product(v2,c);
r[0][2] = dot_product(v3,a);
r[1][2] = dot_product(v3,b);
r[2][2] = dot_product(v3,c);

return;
}

/*****************************************************************************
 *
 *  Settling velocity of a prolate spheroid
 *
 *  L.G. Leal Advanced Transport phemomena, Cambridge 2007
 *  See page 559
 *
 *  r is the apect ratio
 *  f is the force (magnitude)
 *  mu is the dynamic viscosity (LB. units)
 *  ela  principle semi-major axis
 *  u[2] velocity in parallel and perpenduclar directions
 *
*****************************************************************************/
__host__ __device__ void settling_velocity_prolate(double const r,
						   double const f,
						   double const mu,
						   double const ela,
						   double U[2]) {

double ecc,logecc;
double cfterm1,cfterm21,cf1br,cf1;
double cfterm2,cfterm22,cf2br,cf2;
double dcoef;
double rinv;
PI_DOUBLE(pi);
rinv = 1.0/r;
ecc = sqrt(1.0 - (rinv*rinv));
logecc = log((1.0+ecc)/(1.0-ecc));
cfterm1 = -2.0*ecc;
cfterm21 = 1.0 + ecc*ecc;
cf1br = cfterm1 + cfterm21*logecc;
cf1 = (8.0/3.0)*(ecc*ecc*ecc)/cf1br;
cfterm2 = 2.0*ecc;
cfterm22 = 3.0*ecc*ecc - 1.0;
cf2br = cfterm2 + cfterm22*logecc;
cf2 = (16.0/3.0)*(ecc*ecc*ecc)/cf2br;
dcoef = 6.0*pi*mu*ela;
U[0] = f/(dcoef*cf1);
U[1] = f/(dcoef*cf2);
return;
}
/*****************************************************************************
 *
 *  Calculate the unsteady term dIij/dt
 *
 *  Compute rate of change of the moment of inertia tensor from
 *  the curent quaternion, I_ab being the moment of inertia
 *  in the principle co-ordinate frame, and the angular velocity
 *  at (t - delta t) the previous time step (emega in the lab frame).
 *
 *  This horrific expression is the analytical result from Mathematica.
 *
 *  Using this avoids any finite time derivative.
 *
 ****************************************************************************/
__host__ __device__ void unsteady_mI(const double q[4],
				     const double I[3],
				     const double omega[3], double F[3][3]){

double Ixx, Iyy, Izz;
double ox, oy, oz;

Ixx=I[0];
Iyy=I[1];
Izz=I[2];
ox=omega[0];
oy=omega[1];
oz=omega[2];

F[0][0] = 4.0*Izz*(q[0]*q[2] + q[1]*q[3])*(oy*q[0]*q[0] + 2.0*oz*q[0]*q[1] - oy*q[1]*q[1] - oy*q[2]*q[2] - 2.0*oz*q[2]*q[3] + oy*q[3]*q[3]) + 4.0*Iyy*(q[1]*q[2] - q[0]*q[3])*(-(oz*q[0]*q[0]) + 2.0*oy*q[0]*q[1] + oz*q[1]*q[1] - oz*q[2]*q[2] + 2.0*oy*q[2]*q[3] + oz*q[3]*q[3]) - 4.0*Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])* (q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3]));

F[0][1] = -(Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(oz*q[0]*q[0] - 2.0*oy*q[0]*q[1] - oz*q[1]*q[1] + oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) + 4.0*Iyy*(q[1]*q[2] - q[0]*q[3])* (q[2]*(oz*q[1] - ox*q[3]) - q[0]*(ox*q[1] + oz*q[3])) - 4.0*Ixx*(q[1]*q[2] + q[0]*q[3])*(q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) + Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])* (oz*q[0]*q[0] + oz*q[1]*q[1] + 2.0*ox*q[0]*q[2] - 2.0*ox*q[1]*q[3] - oz*(q[2]*q[2] + q[3]*q[3])) - 2.0*Izz*(q[0]*q[0]*q[0]*(oy*q[1] + ox*q[2]) + q[0]*q[0]*(2.0*oz*q[1]*q[1] + ox*q[1]*q[3] - q[2]*(2.0*oz*q[2] + oy*q[3])) + q[3]*(-(ox*q[1]*q[1]*q[1]) + q[1]*q[1]*(oy*q[2] - 2.0*oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[2]*(oy*q[2]*q[2] + 2.0*oz*q[2]*q[3] - oy*q[3]*q[3])) - q[0]*(oy*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[2] + ox*q[2]*(q[2]*q[2] - q[3]*q[3]) + q[1]*(oy*q[2]*q[2] + 8*oz*q[2]*q[3] - oy*q[3]*q[3])));

F[0][2] = -4.0*Izz*(q[0]*q[2] + q[1]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) +  Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(oy*q[0]*q[0] + 2.0*oz*q[0]*q[1] - oy*q[1]*q[1] - oy*q[2]*q[2] - 2.0*oz*q[2]*q[3] + oy*q[3]*q[3]) + 4.0*Ixx*(q[0]*q[2] - q[1]*q[3])*(q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) - Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])*(oy*q[0]*q[0] + oy*q[1]*q[1] - 2.0*ox*q[1]*q[2] - 2.0*ox*q[0]*q[3] - oy*(q[2]*q[2] + q[3]*q[3])) - 2.0*Iyy*(q[0]*q[0]*q[0]*(oz*q[1] + ox*q[3]) + q[0]*q[0]*(-2.0*oy*q[1]*q[1] - ox*q[1]*q[2] + q[3]*(oz*q[2] + 2.0*oy*q[3])) + q[2]*(ox*q[1]*q[1]*q[1] + q[1]*q[1]*(2.0*oy*q[2] - oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[3]*(oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) - q[0]*(oz*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[3] + ox*q[3]*(-q[2]*q[2] + q[3]*q[3]) + q[1]*(-(oz*q[2]*q[2]) + 8*oy*q[2]*q[3] + oz*q[3]*q[3])));

F[1][0] = -(Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(oz*q[0]*q[0] - 2.0*oy*q[0]*q[1] - oz*q[1]*q[1] + oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) + 4.0*Iyy*(q[1]*q[2] - q[0]*q[3])*(q[2]*(oz*q[1] - ox*q[3]) - q[0]*(ox*q[1] + oz*q[3])) - 4.0*Ixx*(q[1]*q[2] + q[0]*q[3])*  (q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) + Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])*(oz*q[0]*q[0] + oz*q[1]*q[1] + 2.0*ox*q[0]*q[2] - 2.0*ox*q[1]*q[3] - oz*(q[2]*q[2] + q[3]*q[3])) - 2.0*Izz*(q[0]*q[0]*q[0]*(oy*q[1] + ox*q[2]) + q[0]*q[0]*(2.0*oz*q[1]*q[1] + ox*q[1]*q[3] - q[2]*(2.0*oz*q[2] + oy*q[3])) + q[3]*(-(ox*q[1]*q[1]*q[1]) + q[1]*q[1]*(oy*q[2] - 2.0*oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[2]*(oy*q[2]*q[2] + 2.0*oz*q[2]*q[3] - oy*q[3]*q[3])) -q[0]*(oy*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[2] + ox*q[2]*(q[2]*q[2] - q[3]*q[3]) + q[1]*(oy*q[2]*q[2] + 8*oz*q[2]*q[3] - oy*q[3]*q[3])));

F[1][1] = -4.0*Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(q[2]*(-(oz*q[1]) + ox*q[3]) + q[0]*(ox*q[1] + oz*q[3])) + 4.0*Izz*(q[0]*q[1] - q[2]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oz*q[0]*q[2] - 2.0*oz*q[1]*q[3] + ox*(-q[2]*q[2] + q[3]*q[3])) + 4.0*Ixx*(q[1]*q[2] + q[0]*q[3])*(oz*q[0]*q[0] + oz*q[1]*q[1] + 2.0*ox*q[0]*q[2] - 2.0*ox*q[1]*q[3] - oz*(q[2]*q[2] + q[3]*q[3]));

F[1][2] = 4.0*Izz*(q[0]*q[1] - q[2]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) - 4.0*Iyy*(q[0]*q[1] + q[2]*q[3])*(q[2]*(-(oz*q[1]) + ox*q[3]) + q[0]*(ox*q[1] + oz*q[3])) + Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oy*q[1]*q[2] + 2.0*oy*q[0]*q[3] + ox*(q[2]*q[2] - q[3]*q[3])) - Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oz*q[0]*q[2] - 2.0*oz*q[1]*q[3] + ox*(-q[2]*q[2] + q[3]*q[3])) - 2.0*Ixx*(q[0]*q[0]*q[0]*(oz*q[2] + oy*q[3]) + q[0]*q[0]*(q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*(q[2]*q[2] - q[3]*q[3])) + q[0]*(-8*ox*q[1]*q[2]*q[3] + q[1]*q[1]*(oz*q[2] + oy*q[3]) - (oz*q[2] + oy*q[3])*(q[2]*q[2] + q[3]*q[3])) + q[1]*(q[1]*q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) - (oy*q[2] - oz*q[3])*(q[2]*q[2] + q[3]*q[3])));

F[2][0] = -4.0*Izz*(q[0]*q[2] + q[1]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) + Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(oy*q[0]*q[0] + 2.0*oz*q[0]*q[1] - oy*q[1]*q[1] - oy*q[2]*q[2] - 2.0*oz*q[2]*q[3] + oy*q[3]*q[3]) + 4.0*Ixx*(q[0]*q[2] - q[1]*q[3])*(q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) - Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])*(oy*q[0]*q[0] + oy*q[1]*q[1] - 2.0*ox*q[1]*q[2] - 2.0*ox*q[0]*q[3] - oy*(q[2]*q[2] + q[3]*q[3])) - 2.0*Iyy*(q[0]*q[0]*q[0]*(oz*q[1] + ox*q[3]) + q[0]*q[0]*(-2.0*oy*q[1]*q[1] - ox*q[1]*q[2] + q[3]*(oz*q[2] + 2.0*oy*q[3])) + q[2]*(ox*q[1]*q[1]*q[1] + q[1]*q[1]*(2.0*oy*q[2] - oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[3]*(oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) - q[0]*(oz*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[3] + ox*q[3]*(-q[2]*q[2] + q[3]*q[3]) + q[1]*(-(oz*q[2]*q[2]) + 8*oy*q[2]*q[3] + oz*q[3]*q[3])));

F[2][1] = 4.0*Izz*(q[0]*q[1] - q[2]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) - 4.0*Iyy*(q[0]*q[1] + q[2]*q[3])*(q[2]*(-(oz*q[1]) + ox*q[3]) + q[0]*(ox*q[1] + oz*q[3])) + Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oy*q[1]*q[2] + 2.0*oy*q[0]*q[3] + ox*(q[2]*q[2] - q[3]*q[3])) - Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oz*q[0]*q[2] - 2.0*oz*q[1]*q[3] + ox*(-q[2]*q[2] + q[3]*q[3])) - 2.0*Ixx*(q[0]*q[0]*q[0]*(oz*q[2] + oy*q[3]) + q[0]*q[0]*(q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*(q[2]*q[2] - q[3]*q[3])) + q[0]*(-8*ox*q[1]*q[2]*q[3] + q[1]*q[1]*(oz*q[2] + oy*q[3]) - (oz*q[2] + oy*q[3])*(q[2]*q[2] + q[3]*q[3])) + q[1]*(q[1]*q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) - (oy*q[2] - oz*q[3])*(q[2]*q[2] + q[3]*q[3])));

F[2][2] = -4.0*Izz*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3])*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3]) + 4.0*Iyy*(q[0]*q[1] + q[2]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oy*q[1]*q[2] + 2.0*oy*q[0]*q[3] + ox*(q[2]*q[2] - q[3]*q[3])) + 4.0*Ixx*(q[0]*q[2] - q[1]*q[3])*(oy*q[0]*q[0] + oy*q[1]*q[1] - 2.0*ox*q[1]*q[2] - 2.0*ox*q[0]*q[3] - oy*(q[2]*q[2] + q[3]*q[3]));

return;
}

/*****************************************************************************/
