/*****************************************************************************
 *
 *  util.c
 *
 *  Utility functions, including vectors.
 *
 *  Little / big endian stuff based on suggestions by Harsha S.
 *  Adiga from IBM.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "util.h"

static void util_swap(int ia, int ib, double a[3], double b[3][3]);

/***************************************************************************
 *
 *  is_bigendian
 *
 *  Byte order for this 4-byte int is 00 00 00 01 for big endian (most
 *  significant byte stored first).
 *
 ***************************************************************************/

__host__ int is_bigendian() {

  const int i = 1;

  return (*(char *) &i == 0);
}

/****************************************************************************
 *
 *  reverse_byte_order_double
 *
 *  Reverse the bytes in the char argument to make a double.
 *
 *****************************************************************************/

__host__ double reverse_byte_order_double(char * c) {

  double result;
  char * p = (char *) &result;
  int b;

  for (b = 0; b < sizeof(double); b++) {
    p[b] = c[sizeof(double) - (b + 1)];
  }

  return result;
}
/*****************************************************************************
 *
 *  util_reverse_byte_order
 *
 *  Converts a scalar big endian value to little endian and vice versa.
 *
 *  The data type is identified via the MPI_Datatype argument.
 *  The input and result arguments may alias.
 *
 *****************************************************************************/

__host__
int util_reverse_byte_order(void * arg, void * result, MPI_Datatype type) {

  char * p = NULL;
  char * carg = NULL;
  size_t b;

  assert(arg);
  assert(result);

  carg = (char *) arg;

  switch (type) {
  case MPI_INT:
    {
      int iresult;
      p = (char *) &iresult;
      
      for (b = 0; b < sizeof(int); b++) {
        p[b] = carg[sizeof(int) - (b + 1)];
      }

      *((int *) result) = iresult;
    }
    break;
  case MPI_DOUBLE:
    {
      double dresult;
      p = (char *) &dresult;

      for (b = 0; b < sizeof(double); b++) {
        p[b] = carg[sizeof(double) - (b + 1)];
      }

      *((double *) result) = dresult;
    }
    break;
  default:
    fatal("Not implemented data type\n");
  }

  return 0;
}

/*****************************************************************************
 *
 *  dot_product
 *
 *****************************************************************************/

__host__ __device__
double dot_product(const double a[3], const double b[3]) {

  return (a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z]);
}

/*****************************************************************************
 *
 *  cross_product
 *
 *****************************************************************************/

__host__ __device__
void cross_product(const double a[3], const double b[3], double result[3]) {

  result[X] = a[Y]*b[Z] - a[Z]*b[Y];
  result[Y] = a[Z]*b[X] - a[X]*b[Z];
  result[Z] = a[X]*b[Y] - a[Y]*b[X];

  return;
}

/*****************************************************************************
 *
 *  modulus
 *
 *****************************************************************************/

__host__ __device__
double modulus(const double a[3]) {

  return sqrt(a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z]);
}

/*****************************************************************************
 *
 *  rotate_vector
 *
 *  Rotate the vector v around the unit axis of rotation \hat{w}
 *  by an angle of \theta, where \theta = |w|. (For example, w
 *  might be an angular velocity.)
 *
 *  The rotated vector is computed via
 *      v' = (1 - cos \theta)(\hat{w}.v) \hat{w} + cos \theta v +
 *           (\hat{w} x v) sin \theta      
 *
 *  For theta positive this gives rotations in the correct sense
 *  in the right-handed coordinate system.
 *
 ****************************************************************************/

__host__ __device__
void rotate_vector(double v[3], const double w[3]) {

  double what[3], vrot[3];
  double theta, ct, st;
  double vdotw;

  theta = sqrt(w[X]*w[X] + w[Y]*w[Y] + w[Z]*w[Z]);

  if (theta == 0.0) {
    /* There is no rotation. */
   }
  else {
    /* Work out the unit axis of rotation */

    what[X] = w[X] / theta;
    what[Y] = w[Y] / theta;
    what[Z] = w[Z] / theta;

    /* Rotation */

    st = sin(theta);
    ct = cos(theta);
    vdotw = v[X]*what[X] + v[Y]*what[Y] + v[Z]*what[Z];

    vrot[X] = ct*v[X] + st*(what[Y]*v[Z] - what[Z]*v[Y]);
    vrot[Y] = ct*v[Y] + st*(what[Z]*v[X] - what[X]*v[Z]);
    vrot[Z] = ct*v[Z] + st*(what[X]*v[Y] - what[Y]*v[X]);
    v[X] = (1.0 - ct)*vdotw*what[X] + vrot[X];
    v[Y] = (1.0 - ct)*vdotw*what[Y] + vrot[Y];
    v[Z] = (1.0 - ct)*vdotw*what[Z] + vrot[Z];
  }

  return;
}

/*****************************************************************************
 *
 *  imin, imax, dmin, dmax
 *
 *  minimax functions
 *
 *****************************************************************************/

__host__ __device__ int imin(const int i, const int j) {
  return ((i < j) ? i : j);
}

__host__ __device__ int imax(const int i, const int j) {
  return ((i > j) ? i : j);
}

__host__ __device__ double dmin(const double a, const double b) {
  return ((a < b) ? a : b);
}

__host__ __device__ double dmax(const double a, const double b) {
  return ((a > b) ? a : b);
}

/*****************************************************************************
 *
 *  util_jacobi_sort
 *
 *  Returns sorted eigenvalues and eigenvectors, highest eigenvalue first.
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

__host__
int util_jacobi_sort(double a[3][3], double vals[3], double vecs[3][3]) {

  int ifail;

  ifail = util_jacobi(a, vals, vecs);

  /* And sort */

  if (vals[X] < vals[Y]) util_swap(X, Y, vals, vecs);
  if (vals[X] < vals[Z]) util_swap(X, Z, vals, vecs);
  if (vals[Y] < vals[Z]) util_swap(Y, Z, vals, vecs);

  return ifail;
}

/*****************************************************************************
 *
 *  util_jacobi
 *
 *  Find the eigenvalues and eigenvectors of a 3x3 symmetric matrix a.
 *  This routine from Press et al. (page 467). The eigenvectors are
 *  returned as the columns of vecs[nrow][ncol].
 *
 *  Returns 0 on success. Garbage out usually means garbage in!
 *
 *****************************************************************************/

__host__ int util_jacobi(double a[3][3], double vals[3], double vecs[3][3]) {

  int iterate, ia, ib, ic;
  double tresh, theta, tau, t, sum, s, h, g, c;
  double b[3], z[3];
  KRONECKER_DELTA_CHAR(d);

  const int maxjacobi = 50;    /* Maximum number of iterations */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      vecs[ia][ib] = d[ia][ib];
    }
    vals[ia] = a[ia][ia];
    b[ia] = a[ia][ia];
    z[ia] = 0.0;
  }

  for (iterate = 1; iterate <= maxjacobi; iterate++) {
    sum = 0.0;

    for (ia = 0; ia < 2; ia++) {
      for (ib = ia + 1; ib < 3; ib++) {
	sum += fabs(a[ia][ib]);
      }
    }

    if (sum < DBL_MIN) return 0;

    if (iterate < 4)
      tresh = 0.2*sum/(3*3);
    else
      tresh = 0.0;

    for (ia = 0; ia < 2; ia++) {
      for (ib = ia + 1; ib < 3; ib++) {

	g = 100.0*fabs(a[ia][ib]);

	if (iterate > 4 && (fabs(vals[ia]) + g) == fabs(vals[ia]) &&
	    (fabs(vals[ib]) + g) == fabs(vals[ib])) {
	  a[ia][ib] = 0.0;
	}
	else if (fabs(a[ia][ib]) > tresh) {
	  h = vals[ib] - vals[ia];
	  if ((fabs(h) + g) == fabs(h)) {
	    t = (a[ia][ib])/h;
	  }
	  else {
	    theta = 0.5*h/a[ia][ib];
	    t = 1.0/(fabs(theta) + sqrt(1.0 + theta*theta));
	    if (theta < 0.0) t = -t;
	  }

	  c = 1.0/sqrt(1 + t*t);
	  s = t*c;
	  tau = s/(1.0 + c);
	  h = t*a[ia][ib];
	  z[ia] -= h;
	  z[ib] += h;
	  vals[ia] -= h;
	  vals[ib] += h;
	  a[ia][ib] = 0.0;

	  for (ic = 0; ic <= ia - 1; ic++) {
	    assert(ic < 3);
	    g = a[ic][ia];
	    h = a[ic][ib];
	    a[ic][ia] = g - s*(h + g*tau);
	    a[ic][ib] = h + s*(g - h*tau);
	  }
	  for (ic = ia + 1; ic <= ib - 1; ic++) {
	    assert(ic < 3);
	    g = a[ia][ic];
	    h = a[ic][ib];
	    a[ia][ic] = g - s*(h + g*tau);
	    a[ic][ib] = h + s*(g - h*tau);
	  }
	  for (ic = ib + 1; ic < 3; ic++) {
	    g = a[ia][ic];
	    h = a[ib][ic];
	    a[ia][ic] = g - s*(h + g*tau);
	    a[ib][ic] = h + s*(g - h*tau);
	  }
	  for (ic = 0; ic < 3; ic++) {
	    g = vecs[ic][ia];
	    h = vecs[ic][ib];
	    vecs[ic][ia] = g - s*(h + g*tau);
	    vecs[ic][ib] = h + s*(g - h*tau);
	  }
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      b[ia] += z[ia];
      vals[ia] = b[ia];
      z[ia] = 0.0;
    }
  }

  return -1;
}

/*****************************************************************************
 *
 *  util_swap
 *
 *  Intended for a[3] eigenvalues and b[nrow][ncol] column eigenvectors.
 *
 *****************************************************************************/

static __host__ void util_swap(int ia, int ib, double a[3], double b[3][3]) {

  int ic;
  double tmp;

  tmp = a[ia];
  a[ia] = a[ib];
  a[ib] = tmp;

  for (ic = 0; ic < 3; ic++) {
    tmp = b[ic][ia];
    b[ic][ia] = b[ic][ib];
    b[ic][ib] = tmp;
  }

  return;
}

/*****************************************************************************
 *
 *  util_discrete_volume_sphere
 *
 *  What is the discrete volume of a sphere radius a0 at position
 *  r0 on the unit lattice?
 *
 *  Lattice sites are assumed to be at integer positions. Points
 *  exactly at a0 from r0 are deemed to be outside. This is
 *  coincides with the criteria for colloid construction.
 *
 *  I don't think there's any way to do this except draw a box
 *  round the outside and count each site.
 *
 *  Result vn returned as a double. Returns zero on success.
 *
 *****************************************************************************/

__host__
int util_discrete_volume_sphere(double r0[3], double a0, double * vn) {

  int ic, jc, kc, nr;
  double x0, y0, z0;    /* Reduced coordinate of argument r0 */
  double rsq;           /* test radius (squared) */
  assert(vn);

  *vn = 0.0;

  /* Reduce the coordinates to 0 <= x < 1 etc */
  x0 = r0[X] - floor(r0[X]);
  y0 = r0[Y] - floor(r0[Y]);
  z0 = r0[Z] - floor(r0[Z]);
  assert(x0 < 1.0);
  assert(0.0 <= x0);

  nr = ceil(a0);

  for (ic = -nr; ic <= nr; ic++) {
    for (jc = -nr; jc <= nr; jc++) {
      for (kc = -nr; kc <= nr; kc++) {
	rsq = pow(1.0*ic - x0, 2) + pow(1.0*jc - y0, 2) + pow(1.0*kc - z0, 2);
	if (rsq < a0*a0) *vn += 1.0;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_gauss_jordan
 *
 *  Solve linear system via Gauss Jordan elimination with full pivoting.
 *  See, e.g., Press et al page 39.
 *
 *  A is the n by n matrix, b is rhs on input and solution on output.
 *  We assume storage of A[i*n + j].
 *  A is column-scrambled inverse on exit. At the moment we don't bother
 *  to recover the inverse.
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

__host__
int util_gauss_jordan(const int n, double * a, double * b) {

  int i, j, k, ia, ib;
  int irow, icol;
  int * ipivot = NULL;

  double rpivot, tmp;

  assert(a);
  assert(b);

  ipivot = (int*) calloc(n, sizeof(int));
  if (ipivot == NULL) return -3;

  icol = -1;
  irow = -1;

  for (j = 0; j < n; j++) {
    ipivot[j] = -1;
  }

  for (i = 0; i < n; i++) {
    tmp = 0.0;
    for (j = 0; j < n; j++) {
      if (ipivot[j] != 0) {
	for (k = 0; k < n; k++) {

	  if (ipivot[k] == -1) {
	    if (fabs(a[j*n + k]) >= tmp) {
	      tmp = fabs(a[j*n + k]);
	      irow = j;
	      icol = k;
	    }
	  }
	}
      }
    }

    assert(icol != -1);
    assert(irow != -1);

    ipivot[icol] += 1;

    if (irow != icol) {
      for (ia = 0; ia < n; ia++) {
	tmp = a[irow*n + ia];
	a[irow*n + ia] = a[icol*n + ia];
	a[icol*n + ia] = tmp;
      }
      tmp = b[irow];
      b[irow] = b[icol];
      b[icol] = tmp;
    }

    if (a[icol*n + icol] == 0.0) {
      free(ipivot);
      return -1;
    }

    rpivot = 1.0/a[icol*n + icol];
    a[icol*n + icol] = 1.0;

    for (ia = 0; ia < n; ia++) {
      a[icol*n + ia] *= rpivot;
    }
    b[icol] *= rpivot;

    for (ia = 0; ia < n; ia++) {
      if (ia != icol) {
	tmp = a[ia*n + icol];
	a[ia*n + icol] = 0.0;
	for (ib = 0; ib < n; ib++) {
	  a[ia*n + ib] -= a[icol*n + ib]*tmp;
	}
	b[ia] -= b[icol]*tmp;
      }
    }
  }

  /* Could recover the inverse here if required. */

  free(ipivot);

  return 0;
}

/*****************************************************************************
 *
 *  util_svd_solve
 *
 *  Solve Ax = b for overdetermined system via SVD.
 *
 *  A is m x n matrix and b[m] is right-hand side on entry.
 *  x[n] is solution on exit.
 *
 *  The x returned minimises |Ax - b| in least squares sense.
 *
 *****************************************************************************/

__host__
int util_svd_solve(int m, int n, double ** a, double * b, double * x) {

  int i, j, k;
  int ifail = 0;
  double sum;
  double wmin, wmax;

  double ** u = NULL;
  double ** v = NULL;
  double * w = NULL;
  double * tmp = NULL;

  ifail += util_matrix_create(m, n, &u);
  ifail += util_matrix_create(n, n, &v);
  ifail += util_vector_create(n, &w);
  ifail += util_vector_create(n, &tmp);

  /* Copy the input a to u and do the SVD */

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      u[i][j] = a[i][j];
    }
  }

  ifail += util_svd(m, n, u, w, v);

  /* What is the maximum singular value? Set the minimum. */

  wmax = 0.0;
  for (j = 0; j < n; j++) {
    wmax = dmax(wmax, w[j]);
  }
  wmin = DBL_EPSILON*wmax;

  /* Do backsubstitution; k counts 'non-zero' singular values */

  k = 0;

  for (j = 0; j < n; j++) {
    sum = 0.0;
    if (w[j] > wmin) {
      for (i = 0; i < m; i++) {
	sum += u[i][j]*b[i];
      }
      sum /= w[j];
      k += 1;
    }
    tmp[j] = sum;
  }

  if (k != n) ifail += 1;

  for (j = 0; j < n; j++) {
    sum = 0.0;
    for (k = 0; k < n; k++) {
      sum += v[j][k]*tmp[k];
    }
    x[j] = sum;
  }

  util_vector_free(&tmp);
  util_vector_free(&w);
  util_matrix_free(n, &v);
  util_matrix_free(m, &u);

  return ifail;
}

/*****************************************************************************
 *
 *  util_vector_create
 *
 *****************************************************************************/

__host__
int util_vector_create(int n, double ** p) {

  int ifail = 0;
  double * v = NULL;

  v = (double*) calloc(n, sizeof(double));
  if (v == NULL) ifail = 1;

  *p = v;

  return ifail;
}

/*****************************************************************************
 *
 *  util_vector_free
 *
 *****************************************************************************/

__host__
int util_vector_free(double ** p) {

  free(*p);
  *p = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  util_matrix_create
 *
 *****************************************************************************/

__host__
int util_matrix_create(int m, int n, double *** p) {

  int ifail = 0;
  int i;
  double ** matrix = NULL;

  matrix = (double**) calloc(m, sizeof(double *));
  if (matrix == NULL) return -1;

  for (i = 0; i < m; i++) {
    matrix[i] = (double*) calloc(n, sizeof(double));
    if (matrix[i] == NULL) ifail += 1;
  }

  *p = matrix;

  return ifail;
}

/*****************************************************************************
 *
 *  util_matrix_free
 *
 *****************************************************************************/

__host__
int util_matrix_free(int m, double ***p) {

  int i;
  assert(p);

  for (i = 0; i < m; i++) {
    free((*p)[i]);
  }
  free(*p);
  *p = NULL;
 
  return 0;
}

/*****************************************************************************
 *
 *  util_singular_value_decomposition
 *
 *  For matrix a, with m rows and n columns, compute the singular
 *  value decompsition a = u w v^t
 *  where u[m][n] replaces a on output, the singular values are w[n]
 *  and v[n][n] is returned (not its transpose).
 * 
 *****************************************************************************/

#define MAX_SVD_ITERATION 30

__host__
int util_svd(int m, int n, double ** a, double * w, double ** v) {

  int i, j, k, ell;
  int jj, nm;
  int iteration, flag;

  double anorm, scale;
  double g, f, h;
  double c, s;
  double x, y, z;
  double * rv1 = NULL;

  assert(m >= n); /* Number of rows is >= number of columns */

  rv1 = (double*) calloc(n, sizeof(double));
  if (rv1 == NULL) return -1;

  g = scale = anorm = 0.0;

  for (i = 0; i < n; i++) {
    ell = i + 1;
    rv1[i] = scale*g;
    g = s = scale = 0.0;
    if (i < m) {
      for (k = i; k < m; k++) {
	scale += fabs(a[k][i]);
      }
      if (scale) {
	for (k = i; k < m; k++) {
	  a[k][i] /= scale;
	  s += a[k][i]*a[k][i];
	}
	f = a[i][i];
	g = -copysign(sqrt(s), f);
	h = f*g - s;
	a[i][i] = f - g;
	for (j = ell; j < n; j++) {
	  for (s = 0.0, k = i; k < m; k++) s += a[k][i]*a[k][j];
	  f = s/h;
	  for (k = i; k < m; k++) a[k][j] += f*a[k][i];
	}
	for (k = i; k < m; k++) a[k][i] *= scale;
      }
    }

    w[i] = scale*g;
    g = s = scale = 0.0;

    if (i < m && i != n - 1) {

      for (k = ell; k < n; k++) scale += fabs(a[i][k]);

      if (scale) {
	for (k = ell; k < n; k++) {
	  a[i][k] /= scale;
	  s += a[i][k]*a[i][k];
	}
	f = a[i][ell];
	g = - copysign(sqrt(s), f);
	h = f*g - s;
	a[i][ell] = f - g;

	for (k = ell; k < n; k++) rv1[k] = a[i][k]/h;
	for (j = ell; j < m; j++) {
	  for (s = 0.0, k = ell; k < n; k++) s += a[j][k]*a[i][k];
	  for (k = ell; k < n; k++) a[j][k] += s*rv1[k];
	}
	for (k = ell; k < n; k++) a[i][k] *= scale;
      }
    }

    anorm = dmax(anorm, fabs(w[i]) + fabs(rv1[i]));
  }

  /* Accumulation of right-hand transformations */

  for (i = n - 1; i >= 0; i--) {
    if (i < n - 1) {
      if (g) {
	for (j = ell; j < n; j++) {
	  /* Double division here avoids possible underflow */
	  v[j][i] = (a[i][j]/a[i][ell]) / g;
	}
	for (j = ell; j < n; j++) {
	  for (s = 0.0, k = ell; k < n; k++) s += a[i][k]*v[k][j];
	  for (k = ell; k < n; k++) v[k][j] += s*v[k][i];
	}
      }
      for (j = ell; j < n; j++) v[i][j] = v[j][i] = 0.0;
    }
    v[i][i] = 1.0;
    g = rv1[i];
    ell = i;
  }

  /* Accumulation of left-hand transforms */

  for (i = n - 1; i >= 0; i--) {
    assert(0 <= i); assert(i < n);
    ell = i + 1;
    g = w[i];
    for (j = ell; j < n; j++) a[i][j] = 0.0;

    if (g) {
      g = 1.0 / g;
      for (j = ell; j < n; j++) {
	for (s = 0.0, k = ell; k < m; k++) s += a[k][i]*a[k][j];
	f = (s / a[i][i])*g;
	for (k = i; k < m; k++) a[k][j] += f*a[k][i];
      }
      for (j = i; j < m; j++) a[j][i] *= g;
    }
    else {
      for (j = i; j < m; j++) a[j][i] = 0.0;
    }
    a[i][i] += 1.0;
  } 


  /* Diagonalisation of the bidiagonal form */

  for (k = n - 1; k >= 0; k--) {

    for (iteration = 1; iteration <= MAX_SVD_ITERATION; iteration++) {

      flag = 1;
      for (ell = k; ell >= 0; ell--) {
	nm = ell - 1;
	/* Note we should always have rv1[0] = 0 to prevent nm = -1 below */
	if (fabs(rv1[ell]) + anorm == anorm) {
	  flag = 0;
	  break;
	}
	assert(0 <= nm); assert(nm < n);
	if (fabs(w[nm]) + anorm == anorm) break;
      }

      if (flag) {
	c = 0.0;
	s = 1.0;
	for (i = ell; i <= k; i++) {
	  assert(0 <= i); assert(i < n);
	  f = s*rv1[i];
	  rv1[i] = c*rv1[i];
	  if (fabs(f) + anorm == anorm) break;
	  g = w[i];
	  util_dpythag(f, g, &h);
	  w[i] = h;
	  h = 1.0 / h;
	  c = g*h;
	  s = -f*h;
	  for (j = 0; j < m; j++) {
	    y = a[j][nm];
	    z = a[j][i];
	    a[j][nm] = y*c + z*s;
	    a[j][i] = z*c - y*s;
	  }
	}
      }

      /* Convergence */

      z = w[k];
      if (ell == k) {
	if (z < 0.0) {
	  /* Singular value is non-negative */
	  w[k] = -z;
	  for (j = 0; j < n; j++) v[j][k] = -v[j][k];
	}
	break;
      }

      if (iteration >= MAX_SVD_ITERATION) {
	free(rv1);
	return -2;
      }

      x = w[ell];
      nm = k - 1;
      assert(0 <= nm); assert(nm < n);
      y = w[nm];
      g = rv1[nm];
      h = rv1[k];
      f = ((y - z)*(y + z) + (g - h)*(g + h)) / (2.0*h*y);
      util_dpythag(f, 1.0, &g);
      f = ((x - z)*(x + z) + h*((y / (f + copysign(g, f))) - h)) / x;

      /* Next QR transformation */

      c = s = 1.0;

      for (j = ell; j <= nm; j++) {
	assert(0 <= j); assert(j < n);
	i = j + 1;
	assert(0 <= i); assert(i < n);
	g = rv1[i];
	y = w[i];
	h = s*g;
	g = c*g;
	util_dpythag(f, h, &z);
	rv1[j] = z;
	c = f/z;
	s = h/z;
	f = x*c + g*s;
	g = g*c - x*s;
	h = y*s;
	y *= c;

	for (jj = 0; jj < n; jj++) {
	  x = v[jj][j];
	  z = v[jj][i];
	  v[jj][j] = x*c + z*s;
	  v[jj][i] = z*c - x*s;
	}

	util_dpythag(f, h, &z);
	w[j] = z;

	if (z) {
	  z = 1.0/z;
	  c = f*z;
	  s = h*z;
	}

	f = c*g + s*y;
	x = c*y - s*g;

	for (jj = 0; jj < m; jj++) {
	  y = a[jj][j];
	  z = a[jj][i];
	  a[jj][j] = y*c + z*s;
	  a[jj][i] = z*c - y*s;
	}
      }

      rv1[ell] = 0.0;
      rv1[k] = f;
      w[k] = x;

      /* End iteration */
    }
  }

  free(rv1);

  return 0;
}

/*****************************************************************************
 *
 *  util_matrix_invert
 *
 *  For n x n matrix, compute and return inverse. This is the same
 *  as the Gauss Jordan routine, but we don't bother with a RHS.
 *
 *  This is done in place.
 *
 *****************************************************************************/

__host__ int util_matrix_invert(int n, double ** a) {

  int i, j, k, ia, ib;
  int irow, icol;

  int * indexcol = NULL;
  int * indexrow = NULL;
  int * ipivot = NULL;

  double rpivot, tmp;

  assert(a);

  indexcol = (int*) calloc(n, sizeof(int));
  indexrow = (int*) calloc(n, sizeof(int));
  ipivot = (int*) calloc(n, sizeof(int));

  if (indexcol == NULL) return -3;
  if (indexrow == NULL) return -3;
  if (ipivot == NULL) return -3;

  icol = -1;
  irow = -1;

  for (j = 0; j < n; j++) {
    ipivot[j] = -1;
  }

  for (i = 0; i < n; i++) {
    tmp = 0.0;
    for (j = 0; j < n; j++) {
      if (ipivot[j] != 0) {
	for (k = 0; k < n; k++) {

	  if (ipivot[k] == -1) {
	    if (fabs(a[j][k]) >= tmp) {
	      tmp = fabs(a[j][k]);
	      irow = j;
	      icol = k;
	    }
	  }
	}
      }
    }

    assert(icol != -1);
    assert(irow != -1);

    ipivot[icol] += 1;

    if (irow != icol) {
      for (ia = 0; ia < n; ia++) {
	tmp = a[irow][ia];
	a[irow][ia] = a[icol][ia];
	a[icol][ia] = tmp;
      }
    }

    indexrow[i] = irow;
    indexcol[i] = icol;

    if (a[icol][icol] == 0.0) {
      free(ipivot);
      free(indexrow);
      free(indexcol);
      return -1;
    }

    rpivot = 1.0/a[icol][icol];
    a[icol][icol] = 1.0;

    for (ia = 0; ia < n; ia++) {
      a[icol][ia] *= rpivot;
    }

    for (ia = 0; ia < n; ia++) {
      if (ia != icol) {
	tmp = a[ia][icol];
	a[ia][icol] = 0.0;
	for (ib = 0; ib < n; ib++) {
	  a[ia][ib] -= a[icol][ib]*tmp;
	}
      }
    }
  }

  /* Recover the inverse. */

  for (i = n - 1; i >= 0; i--) {
    if (indexrow[i] != indexcol[i]) {
      for (j = 0; j < n; j++) {
	tmp = a[j][indexrow[i]];
	a[j][indexrow[i]] = a[j][indexcol[i]];
	a[j][indexcol[i]] = tmp;
      }
    }
  }

  free(ipivot);
  free(indexrow);
  free(indexcol);

  return 0;
}

/*****************************************************************************
 *
 *  util_dpythag
 *
 *  Compute sqrt(a^2 + b^2) with care to avoid underflow or overflow
 *  following Press et al Numerical recipes in C (2nd Ed.) p. 70.
 *
 *****************************************************************************/

__host__ __device__
int util_dpythag(double a, double b, double * p) {

  double absa, absb, tmp;

  absa = fabs(a);
  absb = fabs(b);

  if (absa > absb) {
    tmp = absb/absa;
    *p = absa*sqrt(1.0 + tmp*tmp);
  }
  else {
    if (absb == 0.0) {
      *p = 0.0;
    }
    else {
      tmp = absa/absb;
      *p = absb*sqrt(1.0 + tmp*tmp);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  Linear congruential generator for uniform random numbers based
 *  on one by L'Ecuyer and Simard. See, for example the testu01
 *  packages (v1.2.2)
 *  http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
 *
 *  No state here.
 *
 ****************************************************************************/

#include <inttypes.h>

static __host__ long int util_ranlcg_multiply(long a, long s, long c, long m);

#define RANLCG_A 1389796
#define RANLCG_C 0
#define RANLCG_M 2147483647

#if LONG_MAX == 2147483647
#define RANLCG_HLIMIT   32768
#else
#define RANLCG_HLIMIT   2147483648
#endif

/*****************************************************************************
 *
 *  util_ranlcg_reap_gaussian
 *
 *  Box-Mueller. Caller responisble for maintaining state.
 *
 *  Returns two Gaussian deviates per call.
 *
 *****************************************************************************/

__host__
int util_ranlcg_reap_gaussian(int * state, double r[2]) {

  double ranu[2];
  double f, rsq;

  assert(state);
  assert(*state > 0);

  do {
    util_ranlcg_reap_uniform(state, ranu);
    util_ranlcg_reap_uniform(state, ranu + 1);
    ranu[0] = 2.0*ranu[0] - 1.0;
    ranu[1] = 2.0*ranu[1] - 1.0;
    rsq = ranu[0]*ranu[0] + ranu[1]*ranu[1];
  } while (rsq >= 1.0 || rsq <= 0.0);

  f = sqrt(-2.0*log(rsq)/rsq);
  r[0] = f*ranu[0];
  r[1] = f*ranu[1];

  return 0;
}

/*****************************************************************************
 *
 *  util_ranlcg_reap_uniform
 *
 *  Return one uniform on [0,1). The state is updated and returned
 *  to caller.
 *
 *  Returns zero.
 *
 *****************************************************************************/

__host__
int util_ranlcg_reap_uniform(int * state, double * r) {

  long int sl;

  assert(state);
  assert(*state > 0);

  sl = *state;
  sl = util_ranlcg_multiply(RANLCG_A, sl, RANLCG_C, RANLCG_M);
  *r = sl*(1.0/RANLCG_M);

  *state = sl;

  return 0;
}

/*****************************************************************************
 *
 *  util_ranlcg_multiply
 *
 *  A safe multplication: returned value is (a*s + c) % m
 *
 *****************************************************************************/

static long int util_ranlcg_multiply(long a, long s, long c, long m) {

  long a0, a1, q, qh, rh, k, p;

  if (a < RANLCG_HLIMIT) {
    a0 = a;
    p = 0;
  }
  else {
    a1 = a / RANLCG_HLIMIT;
    a0 = a - RANLCG_HLIMIT * a1;
    qh = m / RANLCG_HLIMIT;
    rh = m - RANLCG_HLIMIT * qh;

    if (a1 >= RANLCG_HLIMIT) {
      a1 = a1 - RANLCG_HLIMIT;
      k = s / qh;
      p = RANLCG_HLIMIT * (s - k * qh) - k * rh;
      if (p < 0) p = (p + 1) % m + m - 1;
    }
    else {
      p = 0;
    }

    if (a1 != 0) {
      q = m / a1;
      k = s / q;
      p -= k * (m - a1 * q);
      if (p > 0) p -= m;
      p += a1 * (s - k * q);
      if (p < 0) p = (p + 1) % m + m - 1;
    }

    k = p / qh;
    p = RANLCG_HLIMIT * (p - k * qh) - k * rh;
    if (p < 0) p = (p + 1) % m + m - 1;
  }

  if (a0 != 0) {
    q = m / a0;
    k = s / q;
    p -= k * (m - a0 * q);
    if (p > 0) p -= m;
    p += a0 * (s - k * q);
    if (p < 0) p = (p + 1) % m + m - 1;
  }

  p = (p - m) + c;
  if (p < 0) p += m;

  return p;
}
