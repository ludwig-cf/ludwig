/* KS: orginal code for eigenvalues and eigenvectors of 3x3 
 * curvature matrix ultimately from Press et al.
 * Requires overhaul. */

#include <math.h>
#include <stdlib.h>

#include "pe.h"

#define NP 3
#define ND 3

#define SIGN(a,b) ((b)<0 ? -fabs(a) : fabs(a))

double *vector( int , int);
double **matrix( int , int , int , int);
void free_vector();
void free_matrix();

double * vector(int nl, int nh)
{
  double *v;

  v = (double *)malloc((unsigned) (nh-nl+1)*sizeof(double));
  if (!v) fatal("allocation failure in vector()");
  return v-nl;
}


double **matrix(int nrl, int nrh, int ncl, int nch)
{
  int i;
  double **m;

  m=(double **) malloc((unsigned) (nrh-nrl+1)*sizeof(double*));
  if (!m) fatal("allocation failure 1 in matrix()");
  m -= nrl;

  for(i=nrl;i<=nrh;i++) {
    m[i]=(double *) malloc((unsigned) (nch-ncl+1)*sizeof(double));
    if (!m[i]) fatal("allocation failure 2 in matrix()");
    m[i] -= ncl;
  }
  return m;
}

void free_vector(double * v, int nl, int nh) {
  free((char*) (v+nl));
}

void free_matrix(double **  m, int nrl, int nrh, int ncl, int nch) {
  int i;

  for(i=nrh;i>=nrl;i--) free((char*) (m[i]+ncl));
  free((char*) (m+nrl));
}


void tred2(double ** a, int n, double d[], double e[]) {
  int l,k,j,i;
  double scale,hh,h,g,f;

  for (i=n;i>=2;i--) {
    l=i-1;
    h=scale=0.0;
    if (l > 1) {
      for (k=1;k<=l;k++)
	scale += fabs(a[i][k]);
      if (scale == 0.0)
	e[i]=a[i][l];
      else {
	for (k=1;k<=l;k++) {
	  a[i][k] /= scale;
	  h += a[i][k]*a[i][k];
	}
	f=a[i][l];
	g = f>0 ? -sqrt(h) : sqrt(h);
	e[i]=scale*g;
	h -= f*g;
	a[i][l]=f-g;
	f=0.0;
	for (j=1;j<=l;j++) {
	  /* Next statement can be omitted if eigenvectors not wanted */
	  a[j][i]=a[i][j]/h;
	  g=0.0;
	  for (k=1;k<=j;k++)
	    g += a[j][k]*a[i][k];
	  for (k=j+1;k<=l;k++)
	    g += a[k][j]*a[i][k];
	  e[j]=g/h;
	  f += e[j]*a[i][j];
	}
	hh=f/(h+h);
	for (j=1;j<=l;j++) {
	  f=a[i][j];
	  e[j]=g=e[j]-hh*f;
	  for (k=1;k<=j;k++)
	    a[j][k] -= (f*e[k]+g*a[i][k]);
	}
      }
    } else
      e[i]=a[i][l];
    d[i]=h;
  }
  /* Next statement can be omitted if eigenvectors not wanted */
  d[1]=0.0;
  e[1]=0.0;
  /* Contents of this loop can be omitted if eigenvectors not
     wanted except for statement d[i]=a[i][i]; */
  for (i=1;i<=n;i++) {
    l=i-1;
    if (d[i]) {
      for (j=1;j<=l;j++) {
	g=0.0;
	for (k=1;k<=l;k++)
	  g += a[i][k]*a[k][j];
	for (k=1;k<=l;k++)
	  a[k][j] -= g*a[k][i];
      }
    }
    d[i]=a[i][i];
    a[i][i]=1.0;
    for (j=1;j<=l;j++) a[j][i]=a[i][j]=0.0;
  }
}


void tqli(double d[], double e[], int n, double ** z) {
  int m,l,iter,i,k;
  double s,r,p,g,f,dd,c,b;

  for (i=2;i<=n;i++) e[i-1]=e[i];
  e[n]=0.0;
  for (l=1;l<=n;l++) {
    iter=0;
    do {
      for (m=l;m<=n-1;m++) {
	dd=fabs(d[m])+fabs(d[m+1]);
	if (fabs(e[m])+dd == dd) break;
      }
      if (m != l) {

/* [JCD: 03/03/2005]: what a non-sense, crashing the whole simulation
 * because we cannot measure the length scales - modify to return */

	if (iter++ == 50){
	  info("Too many iterations in TQLI (%d)\n",l);
/* Set d to 0.0 and return -> L1 = L2 = L3 = 0.0 */
/* Bad hack... */
	  d[1] = 0.0; d[2] = 0.0; d[3] = 0.0;
	  return;
	}

	g=(d[l+1]-d[l])/(2.0*e[l]);
	r=sqrt((g*g)+1.0);
	g=d[m]-d[l]+e[l]/(g+SIGN(r,g));
	s=c=1.0;
	p=0.0;
	for (i=m-1;i>=l;i--) {
	  f=s*e[i];
	  b=c*e[i];
	  if (fabs(f) >= fabs(g)) {
	    c=g/f;
	    r=sqrt((c*c)+1.0);
	    e[i+1]=f*r;
	    c *= (s=1.0/r);
	  } else {
	    s=f/g;
	    r=sqrt((s*s)+1.0);
	    e[i+1]=g*r;
	    s *= (c=1.0/r);
	  }
	  g=d[i+1]-p;
	  r=(d[i]-g)*s+2.0*c*b;
	  p=s*r;
	  d[i+1]=g+p;
	  g=c*r-b;
	  /* Next loop can be omitted if eigenvectors not wanted */
	  for (k=1;k<=n;k++) {
	    f=z[k][i+1];
	    z[k][i+1]=s*z[k][i]+c*f;
	    z[k][i]=c*z[k][i]-s*f;
	  }
	}
	d[l]=d[l]-p;
	e[l]=g;
	e[m]=0.0;
      }
    } while (m != l);
  }
}


void eigen3(double dxx,double dxy,double dxz,double dyy,double dyz,double dzz,
       double *eva1, double *eve1,
       double *eva2, double *eve2,
       double *eva3, double *eve3)
{
  int i;
  double *d,*e,**a;
  /*	static double c[NP][NP]=
	{ 1.0, 0.1, 0.2,
	0.1, 2.0, 0.3,
	0.2, 0.3, 3.0};*/

  d=vector(1,NP);
  e=vector(1,NP);
  /* f=vector(1,NP);*/
  a=matrix(1,NP,1,NP);
  /*	for (i=1;i<=NP;i++)
	for (j=1;j<=NP;j++) a[i][j]=c[i-1][j-1];*/

  a[1][1]=dxx; a[1][2]=dxy; a[1][3]=dxz;
  a[2][1]=dxy; a[2][2]=dyy; a[2][3]=dyz;
  a[3][1]=dxz; a[3][2]=dyz; a[3][3]=dzz;
  tred2(a,NP,d,e);
  tqli(d,e,NP,a);

  *eva1=d[1];*eva2=d[2];*eva3=d[3];
  for (i=1;i<=NP;i++) {
    eve1[i-1]=a[i][1];
    eve2[i-1]=a[i][2];
    eve3[i-1]=a[i][3];
  }
  free_matrix(a,1,NP,1,NP);
  free_vector(e,1,NP);
  free_vector(d,1,NP);
}


/* KS pending replace by ... */

void eigenvalues_and_vectors(double values[ND], double a[ND][ND]) {

  double e[ND];
  void householder(double [ND][ND], double [ND], double [ND]);
  void ql_factor(double [ND], double [ND], double[ND][ND]);

  /* Call Householder reduction. */

  householder(a, values, e);

  /* Call QL */

  ql_factor(values, e, a);

  return;
}

void householder(double a[ND][ND], double d[3], double e[3]) {

  int l,k,j,i;
  double scale,hh,h,g,f;

  for (i = ND-1; i > 0;i--) {
    l = i-1;
    h = scale = 0.0;

    if (l > 1) {
      for (k = 0; k <= l; k++)
	scale += fabs(a[i][k]);

      if (scale == 0.0)
	e[i]=a[i][l];
      else {
	for (k = 0; k <= l; k++) {
	  a[i][k] /= scale;
	  h += a[i][k]*a[i][k];
	}

	f=a[i][l];
	g = f > 0 ? -sqrt(h) : sqrt(h);
	e[i] = scale*g;
	h -= f*g;
	a[i][l] = f-g;
	f = 0.0;
	for (j = 0; j <= l; j++) {
	  a[j][i]=a[i][j]/h;
	  g=0.0;

	  for (k = 0; k <= j; k++)
	    g += a[j][k]*a[i][k];
	  for (k = j+1; k <=l; k++)
	    g += a[k][j]*a[i][k];

	  e[j] = g/h;
	  f += e[j]*a[i][j];
	}

	hh = f/(h+h);

	for (j = 0; j <= l; j++) {
	  f = a[i][j];
	  e[j] = g = e[j]-hh*f;
	  for (k = 0; k <= j; k++)
	    a[j][k] -= (f*e[k]+g*a[i][k]);
	}
      }
    }
    else {
      e[i]=a[i][l];
    }
    d[i]=h;
  }

  d[1] = 0.0;
  e[1] = 0.0;

  for (i = 0;i < ND; i++) {
    l = i-1;
    if (d[i]) {
      for (j = 0; j <= l; j++) {
	g = 0.0;
	for (k = 0; k <= l; k++)
	  g += a[i][k]*a[k][j];
	for (k = 0; k<= l; k++)
	  a[k][j] -= g*a[k][i];
      }
    }
    d[i] = a[i][i];
    a[i][i] = 1.0;
    for (j = 0; j<= l;j++) a[j][i] = a[i][j] = 0.0;
  }

  return;
}

void ql_factor(double d[ND], double e[ND], double z[ND][ND]) {

  int m,l,iter,i,k;
  double s,r,p,g,f,dd,c,b;

  for (i = 1; i < ND;i++) e[i-1] = e[i];
  e[ND-1] = 0.0;

  for (l = 0; l < ND; l++) {
    iter=0;
    do {
      for (m = l; m < ND-1; m++) {
	dd = fabs(d[m]) + fabs(d[m+1]);
	if (fabs(e[m]) + dd == dd) break;
      }

      if (m != l) {

/* [JCD: 03/03/2005]: what a non-sense, crashing the whole simulation
 * because we cannot measure the length scales - modify to return */

	if (iter++ == 30){
	  info("Too many iterations in TQLI (%d)\n",l);
	  /* Set d to 0.0 and return -> L1 = L2 = L3 = 0.0 */
	  d[0] = 0.0; d[1] = 0.0; d[2] = 0.0;
	  return;
	}

	g = (d[l+1]-d[l])/(2.0*e[l]);
	r = sqrt((g*g)+1.0);
	g = d[m]-d[l]+e[l]/(g+SIGN(r,g));
	s = c = 1.0;
	p = 0.0;
	for (i = m-1; i >= l; i--) {
	  f = s*e[i];
	  b = c*e[i];
	  if (fabs(f) >= fabs(g)) {
	    c=g/f;
	    r=sqrt((c*c)+1.0);
	    e[i+1]=f*r;
	    c *= (s=1.0/r);
	  } else {
	    s=f/g;
	    r=sqrt((s*s)+1.0);
	    e[i+1]=g*r;
	    s *= (c=1.0/r);
	  }
	  g=d[i+1]-p;
	  r=(d[i]-g)*s+2.0*c*b;
	  p=s*r;
	  d[i+1]=g+p;
	  g=c*r-b;

	  for (k = 0; k < ND; k++) {
	    f=z[k][i+1];
	    z[k][i+1]=s*z[k][i]+c*f;
	    z[k][i]=c*z[k][i]-s*f;
	  }
	}
	d[l]=d[l]-p;
	e[l]=g;
	e[m]=0.0;
      }
    } while (m != l);
  }

  return;
}
