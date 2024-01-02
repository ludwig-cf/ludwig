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
 *  util_q4_from_omega
 *
 *  Angular velocity update as a quaternion.
 *
 *  See, e.g., Zhao and Wachem Acta Mech 224 3091--3109 (2013)
 *  Eq 49. The factor (1/2)\delta t is replaced by the general
 *  fractional time step dt.
 *
 *  q = [ cos(|w|dt), sin(|w|dt) w/|w| ]
 *
 *****************************************************************************/

void util_q4_from_omega(const double omega[3], double dt, double q[4]) {

  double ww = sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);

  if (ww < DBL_EPSILON) {
    q[0] = 1.0; q[1] = 0.0; q[2] = 0.0; q[3] = 0.0;
  }
  else {
    q[0] = cos(ww*dt);
    for (int i = 0; i < 3; i++) {
      q[i+1] = sin(ww*dt)*omega[i]/ww;
    }
  }

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
  double qxa[3] = {0};

  cross_product(q + 1, a, qxa);

  for (int i = 0; i < 3; i++) {
    b[i] = (2.0*q0*q0 - 1.0)*a[i] + 2.0*qdota*q[i+1] + 2.0*q0*qxa[i];
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
 *  util_q4_is_inside_ellispoid
 *
 *  Is position r inside the ellpsoid (a,b,c) with orientation described by
 *  unit quaternion q4 (in the body frame)?
 *
 *  Position r is relative to the centre of the ellipsoid in the lab
 *  or world frame.
 *
 *  Returns 0 if r is outside.
 *
 *  NEEDS REFERENCE TO PROVIDE MEANING TO COMMENTS
 *
 *****************************************************************************/

int util_q4_is_inside_ellipsoid(const double q[4], const double elabc[3],
				const double r[3]) {
  int inside = 0;

  double elev1[3] = {0};
  double elev2[3] = {0};
  double elev3[3] = {0};

  double worldv1[3] = {1.0, 0.0, 0.0};
  double worldv2[3] = {0.0, 1.0, 0.0};

  double elL[3][3] = {0};
  double elA[3][3] = {0};
  double elQ[3][3] = {0};

  /* Construct Lambda matrix */
  for (int i = 0; i < 3; i++) {
    elL[i][i] = 1.0/(elabc[i]*elabc[i]);
  }

  /* Construct Q matrix */
  util_q4_rotate_vector(q, worldv1, elev1);
  util_q4_rotate_vector(q, worldv2, elev2);

  cross_product(elev1, elev2, elev3);
  util_vector_normalise(3, elev3);

  for (int i = 0; i < 3; i++) {
    elQ[i][0] = elev1[i];
    elQ[i][1] = elev2[i];
    elQ[i][2] = elev3[i];
  }

  /* Construct A matrix */
  {
    double elAp[3][3] = {0};
    double elQT[3][3] = {0};
    matrix_product(elQ, elL, elAp);
    matrix_transpose(elQ, elQT);
    matrix_product(elAp, elQT, elA);
  }

  /* Evaluate quadratic equation */
  {
    double x = elA[0][0]*r[X]*r[X] + elA[1][1]*r[Y]*r[Y] + elA[2][2]*r[Z]*r[Z]
      + (elA[0][1] + elA[1][0])*r[X]*r[Y]
      + (elA[0][2] + elA[2][0])*r[X]*r[Z]
      + (elA[1][2] + elA[2][1])*r[Y]*r[Z];

    inside = (x < 1.0);
  }

  return inside;
}

/*****************************************************************************
 *
 *  util_q4_inertia_tensor
 *
 *  Construct the moment of inertia tensor in the principal coordinates
 *  from the quaternion describing ellipsoid orientation.
 *
 *  The (diagonal) moment of interia tensor in the lab frame is moment[3].
 *  This must be rotated by the quaternion.
 *
 *  Zhao and Wachem Acta Mech 224, 2331--2358 (2013)
 *  describe how to rotate a general tensor by a quaternion.
 *  The full operation is described by Eqns 22 - 29.
 *
 *  This is the special case where the initial tensor is diagonal.
 *
 *  The result mI is the full inertia tensor in the rotated frame.
 *
 ****************************************************************************/

void util_q4_inertia_tensor(const double q[4], const double moment[3],
			    double mI[3][3]) {

  double Mdd[3][3] = {0};

  /* Construct the transpose of the rotated column vectors ... */

  for (int j = 0; j < 3; j++) {
    double Mi[3]  = {0};
    double Mdi[3] = {0};
    Mi[j] = moment[j];
    util_q4_rotate_vector(q, Mi, Mdi);
    for (int i = 0; i < 3; i++) {
      Mdd[j][i] = Mdi[i];
    }
  }

  /* Repeat the entire procedure rotating the rows for the final mI */

  for (int j = 0; j < 3; j++) {
    double Mi[3]  = {0};
    double Mdi[3] = {0};
    for (int i = 0; i < 3; i++) {
      Mi[i] = Mdd[i][j];
    }
    util_q4_rotate_vector(q, Mi, Mdi);
    for (int i = 0; i < 3; i++) {
      mI[j][i] = Mdi[i];
    }
  }

  return;
}

#ifdef PENDING_REFACTOR_FOR_VALIDATIONS
/*****************************************************************************
 *
 *  Far field predictions of Mitchell and Spagnolie
 *  PENDING CONFIRMATION
 *
 *****************************************************************************/
void ellipsoid_nearwall_predicted(double const elabc[3], double const h, double const quat[4], double Upred[3], double opred[3]) {

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
  util_q4_to_euler_angles(quat, &phi, &theta, &psi);
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
void Jeffery_omega_predicted(double const r, double const quat[4], double const gammadot, double opred[3], double angpred[2]) {

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
  util_q4_rotate_vector(quat,v1,p);

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
#endif

/*****************************************************************************
 *
 *  util_ellipsoid_euler_from_vectors
 *
 *  The intent here is to allow the user to specify an initial orientation
 *  (of an ellipsoid) by two vectors (along the semi-major and a semi-minor
 *  axis).
 *
 *  These vectors do not have to be unit vectors, and they do not even have
 *  to be at right angles. However, both should be non-zero, and they must
 *  not be linearly dependent (parallel). If they are not at right angles,
 *  the second is adjusted to be so.
 *
 *  The result is a standard set of Euler angles describing the orientation.
 *
 *  If the return value is not zero, the input vectors did not satisfy
 *  the conditions above.
 *
 *****************************************************************************/

int util_ellipsoid_euler_from_vectors(const double a0[3], const double b0[3],
				      double euler[3]) {
  int ifail = 0;
  double a[3] = {a0[0], a0[1], a0[2]};
  double b[3] = {b0[0], b0[1], b0[2]};

  /* Make sure the inputs are unit vectors at right angles ... */
  util_vector_normalise(3, a);
  ifail = util_vector_orthonormalise(a, b);

  /* Cross product of orthongonal unit vectors is itself a unit vector
   * so we can compute the direction cosine matrix. */

  {
    double c[3] = {0};
    double r[3][3] = {0};
    util_vector_cross_product(c, a, b);
    util_vector_basis_to_dcm(a, b, c, r);
    util_vector_dcm_to_euler(r, euler, euler + 1, euler + 2);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_ellipsoid_prolate_settling_velocity
 *
 *  L.G. Leal Advanced Transport phemomena, Cambridge University Press (2007)
 *  See page 559 for a prolate spheroid...
 *
 *  If e is the eccentricity,  e = sqrt(1 - b^2/a^2), and coefficients
 *  cf1 =  (8/3) e^3 [-2e + (1 +e^2) log(1+e/1-e)]^-1
 *  cf2 = (16/3) e^3 [+2e +(3e^2 - 1) log(1+e/1-e)]^-1
 *
 *  The Stokes' relationship is:
 *  F = 6 pi mu a (U_1 cf1 xhat + U_2 cf2 yhat)
 *
 *  a    is the semi-major axis
 *  b    is the semi-minor axis (b < a)
 *  eta  is the dynamic viscosity (lattice units)
 *  f    is the force (magnitude)
 *  u[2] velocity in parallel and perpenduclar directions (xhat, yhat)
 *
 *****************************************************************************/

int util_ellipsoid_prolate_settling_velocity(double a,
					     double b,
					     double eta,
					     double f,
					     double u[2]) {
  int ifail = 0;
  double pi = 4.0*atan(1.0);

  if (a <= 0.0) {
    ifail = -1;
  }
  else {
    /* Have a little care to manage the limit e -> 0 (a sphere) */
    double e    = sqrt(1.0 - (b*b)/(a*a));
    double loge = log((1.0 + e)/(1.0 - e));
    double cf1  = 1.0;
    double cf2  = 1.0;

    if (e > 0.0) {
      cf1 =  (8.0/3.0)*e*e*e/(-2.0*e + (1.0 + e*e)*loge);
      cf2 = (16.0/3.0)*e*e*e/(+2.0*e + (3.0*e*e - 1.0)*loge);
    }

    u[0] = f/(6.0*pi*eta*a*cf1);
    u[1] = f/(6.0*pi*eta*a*cf2);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_ellipsoid_is_sphere
 *
 *  Return 1 if a = b = c.
 *
 *****************************************************************************/

int util_ellipsoid_is_sphere(const double elabc[3]) {

  double a = elabc[0];
  double b = elabc[1];
  double c = elabc[2];

  return ((fabs(b - a) < DBL_EPSILON) && (fabs(c - b) < DBL_EPSILON));
}

/*****************************************************************************
 *
 *  util_spheroid_normal_tangent
 *
 *  Calculate surface normal or tangent on a spheroid at a point r
 *  which may be at, or near, the surface.
 *
 *  The computation is rather similar: tangent = 1 gives tangent,
 *  otherwise the normal is returned.
 *
 *  Note
 *  "Spheroid" has a != b and b = c (for oblate or prolate case).
 *
 *  See, e.g., S.R. Keller and T.Y.T. Wu, J. Fluid Mech. 80, 259--278 (1977).
 *
 *  CHECK what happens in the oblate case?
 *
 *  The vector returned should be a unit vector if r is exactly at the
 *  surface, but may not be in other cases.
 *
 ****************************************************************************/

int util_spheroid_normal_tangent(const double elabc[3], const double elbz[3],
				 const double r[3], int tangent,
				 double result[3]) {
  int ifail = 0;
  double elc;
  double ele,ele2;
  double ela,ela2;
  double elz,elz2;
  double elr;
  double rmod;
  double elrho[3];
  double diff1, diff2;

  /* elabc[0] must be a, the largest dimension, and b == c */
  assert(elabc[0] >  elabc[1]);
  assert(fabs(elabc[1] - elabc[2]) < DBL_EPSILON);

  ela = elabc[0];
  elc = sqrt(elabc[0]*elabc[0] - elabc[1]*elabc[1]);
  ele = elc/ela;
  elz = dot_product(r, elbz);

  for (int ia = 0; ia < 3; ia++) {
    elrho[ia] = r[ia] - elz*elbz[ia];
  }
  elr = modulus(elrho);
  rmod = 0.0;
  if (elr != 0.0) rmod = 1.0/elr;

  for (int ia = 0; ia < 3; ia++) {
    elrho[ia] = elrho[ia]*rmod;
  }

  ela2  = ela*ela;
  elz2  = elz*elz;
  ele2  = ele*ele;
  diff1 = ela2 - elz2;
  diff2 = ela2 - ele2*elz2;

  /* If r is not exactly at the surface, then elz > ela. An adjustment
   * is made ... */

  if (diff1 < 0.0) {
    double dr[3] = {0};
    double gridin[3] = {0};
    double elzin = 0.0;
    elr = modulus(r);
    rmod = 0.0;
    if (elr != 0.0) rmod = 1.0/elr;
    for (int ia = 0; ia < 3; ia++) {
      dr[ia] = r[ia]*rmod;
    }
    for (int ia = 0; ia < 3; ia++) {
      gridin[ia] = r[ia] - dr[ia];
    }
    elzin = dot_product(gridin, elbz);
    elz2  = elzin*elzin;
    diff1 = ela2 - elz2;

    if (diff2 < 0.0) diff2 = ela2 - ele2*elz2;
  }

  assert(diff1 >= 0.0);
  assert(diff2 >  0.0);

  {
    double denom = sqrt(diff2);
    double term1 = sqrt(diff1)/denom;
    double term2 = sqrt(1.0 - ele*ele)*elz/denom;

    if (tangent) {
      /* Tangent vector */
      for (int ia = 0; ia < 3; ia++) {
	result[ia] = -term1*elbz[ia] + term2*elrho[ia];
      }
      elr = modulus(elrho);
      if (elr <= 0.0) ifail = -999; /* tangent fails */
    }
    else {
      /* Normal vector */
      for (int ia = 0; ia < 3; ia++) {
	result[ia] = +term2*elbz[ia] + term1*elrho[ia];
      }
    }
    /* If r is exactly at the surface, the result will be a unit vector. */
  }

  return ifail;
}

/*****************************************************************************
 *
 *  util_spheroid_surface_normal
 *
 *  Calculate surface normal on an spheroid.
 *  elabc[3]  - a >= b = c for spheroid here
 *  m[3]      - vector aligned along principal axis a
 *  r[3]      - position at surface
 *  v[3]      - normal vector at position r
 *
 ****************************************************************************/

int util_spheroid_surface_normal(const double elabc[3], const double m[3],
				 const double r[3], double v[3]) {

  return util_spheroid_normal_tangent(elabc, m, r, 0, v);
}

/*****************************************************************************
 *
 *  util_spheroid_surface_tangent
 *
 *  Calculate surface tangent on a spheroid.
 *  See comments above for normal version, except the result is the tangent.
 *
 *****************************************************************************/

int util_spheroid_surface_tangent(const double elabc[3], const double m[3],
				  const double r[3], double vt[3]) {

  return util_spheroid_normal_tangent(elabc, m, r, 1, vt);
}
