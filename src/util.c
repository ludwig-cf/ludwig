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
 *  (c) 2010-2024 The University of Edinburgh
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
  unsigned int b;

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
 *  Comparison of MPI_Datatype is formally dubious, but should be ok
 *  for intrinsic types.
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

  if (type == MPI_INT) {
    
    int iresult;
    p = (char *) &iresult;
      
    for (b = 0; b < sizeof(int); b++) {
      p[b] = carg[sizeof(int) - (b + 1)];
    }

    *((int *) result) = iresult;
  }
  else if (type == MPI_DOUBLE) {

    double dresult;
    p = (char *) &dresult;

    for (b = 0; b < sizeof(double); b++) {
      p[b] = carg[sizeof(double) - (b + 1)];
    }

    *((double *) result) = dresult;
  }
  else {
    printf("Not implemented data type\n");
    assert(0);
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
 *  util_random_unit_vector
 *
 *  Returns a vector randomly on the surface of a unit sphere.
 *  Method of Marsaglia [1972]. See Allen and Tildesley.
 *
 *****************************************************************************/

int util_random_unit_vector(int * state, double rhat[3]) {

  double r[2];
  double zeta1, zeta2, zsq;  

  do {
    util_ranlcg_reap_uniform(state, r);
    util_ranlcg_reap_uniform(state, r + 1);
    zeta1 = 1.0 - 2.0*r[0];
    zeta2 = 1.0 - 2.0*r[1];
    zsq   = zeta1*zeta1 + zeta2*zeta2;
  } while (zsq > 1.0);

  rhat[0] = 2.0*zeta1*sqrt(1.0 - zsq);
  rhat[1] = 2.0*zeta2*sqrt(1.0 - zsq);
  rhat[2] = 1.0 - 2.0*zsq;

  return 0;
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

    /* Successful return is zero. */
    if (sum < DBL_MIN) return 0;

    if (iterate < 4)
      tresh = 0.2*sum/(3*3);
    else
      tresh = 0.0;

    for (ia = 0; ia < 2; ia++) {
      for (ib = ia + 1; ib < 3; ib++) {

	g = 100.0*fabs(a[ia][ib]);

	if (iterate > 4 && (((fabs(vals[ia]) + g) - fabs(vals[ia])) == 0.0) &&
	    (((fabs(vals[ib]) + g) - fabs(vals[ib])) == 0.0)) {
	  a[ia][ib] = 0.0;
	}
	else if (fabs(a[ia][ib]) > tresh) {
	  h = vals[ib] - vals[ia];
	  if (((fabs(h) + g) - fabs(h)) == 0.0) {
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

  /* Exceded maximum iterations: a fail ... */

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
 *  util_discrete_area_disk
 *
 *  For a disk of radius a0 and position r0 in two dimensions, what is
 *  the discrete area?
 *
 *****************************************************************************/

int util_discrete_area_disk(double a0, const double r0[2], double * vn) {

  int ifail = 0;

  if (vn == NULL) {
    ifail = -1;
  }
  else {

    /* Reduce the coordinates to 0 <= x < 1 etc */
    double x0 = r0[X] - floor(r0[X]);
    double y0 = r0[Y] - floor(r0[Y]);

    int nr = ceil(a0);

    assert(0.0 <= x0 && x0 < 1.0);
    assert(0.0 <= y0 && y0 < 1.0);

    *vn = 0.0;

    for (int ic = -nr; ic <= nr; ic++) {
      for (int jc = -nr; jc <= nr; jc++) {
	double rsq = pow(1.0*ic - x0, 2) + pow(1.0*jc - y0, 2);
	if (rsq < a0*a0) *vn += 1.0;
      }
    }
  }

  return ifail;
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
int util_discrete_volume_sphere(const double r0[3], double a0, double * vn) {

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
 *  util_matrix_create
 *
 *****************************************************************************/

__host__
int util_matrix_create(int m, int n, double *** p) {

  int ifail = 0;
  int i;
  double ** matrix = NULL;

  matrix = (double**) calloc(m, sizeof(double *));
  assert(matrix);
  if (matrix == NULL) return -1;

  for (i = 0; i < m; i++) {
    matrix[i] = (double*) calloc(n, sizeof(double));
    assert(matrix[i]);
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
  assert(*p);

  for (i = 0; i < m; i++) {
    free((*p)[i]);
  }
  free(*p);
  *p = NULL;
 
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

  int irow = -1;
  int icol = -1;

  int * indexcol = NULL;
  int * indexrow = NULL;
  int * ipivot = NULL;

  assert(a);

  indexcol = (int*) calloc(n, sizeof(int));
  assert(indexcol);
  if (indexcol == NULL) return -3;

  indexrow = (int*) calloc(n, sizeof(int));
  assert(indexrow);
  if (indexrow == NULL) {
    free(indexcol);
    return -3;
  }

  ipivot = (int*) calloc(n, sizeof(int));
  assert(ipivot);
  if (ipivot == NULL) {
    free(indexrow);
    free(indexcol);
    return -3;
  }

  /* Begin the Gaussian elimination */
  for (int j = 0; j < n; j++) {
    ipivot[j] = -1;
  }

  for (int i = 0; i < n; i++) {
    double tmp = 0.0;
    for (int j = 0; j < n; j++) {
      if (ipivot[j] != 0) {
	for (int k = 0; k < n; k++) {

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
      for (int ia = 0; ia < n; ia++) {
	tmp = a[irow][ia];
	a[irow][ia] = a[icol][ia];
	a[icol][ia] = tmp;
      }
    }

    /* Check the pivot element is not zero ... */
    indexrow[i] = irow;
    indexcol[i] = icol;

    if (a[icol][icol] == 0.0) {
      free(ipivot);
      free(indexrow);
      free(indexcol);
      return -1;
    }

    {
      double rpivot = 1.0/a[icol][icol];
      a[icol][icol] = 1.0;

      for (int ia = 0; ia < n; ia++) {
	a[icol][ia] *= rpivot;
      }
    }

    for (int ia = 0; ia < n; ia++) {
      if (ia != icol) {
	tmp = a[ia][icol];
	a[ia][icol] = 0.0;
	for (int ib = 0; ib < n; ib++) {
	  a[ia][ib] -= a[icol][ib]*tmp;
	}
      }
    }
    /* .. outer loop .. */
  }

  /* Recover the inverse. */

  for (int i = n - 1; i >= 0; i--) {
    if (indexrow[i] != indexcol[i]) {
      for (int j = 0; j < n; j++) {
	double tmp = a[j][indexrow[i]];
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

__host__ int util_dpythag(double a, double b, double * p) {

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

/*****************************************************************************
 *
 *  util_str_tolower
 *
 *  Force first maxlen characters of str to be lower case.
 *
 *****************************************************************************/

__host__ int util_str_tolower(char * str, size_t maxlen) {

  size_t n, nlen;

  assert(str);

  nlen = strlen(str);
  if (maxlen < nlen) nlen = maxlen;

  for (n = 0; n < nlen; n++) {
    str[n] = tolower(str[n]);
  }

  return 0;
}

/*****************************************************************************
 *
 *  util_rectangle_conductance
 *
 *  The steady volume flux (volume flow rate) Q in a rectangular
 *  capillary of cross section width x height 2b x 2c (with 2b > 2c)
 *  we write:
 *
 *    Q = -C (dp/dx) / eta
 *
 *  with dp/dx the pressure gradient and eta the dynamic viscosity.
 *  One can define a viscosity-independent conductance C
 *
 *    C = (4/3) b c^3 [ 1 - 6(c/b) \sum_k tanh (a_k b/c)/a_k^5 ]
 *
 *  where a_k = (2k - 1) pi/2 and the sum is k = 1, ..., \inf.
 *
 *  This function returns the value of conductance for (w, h) w > h.
 *  "q" in the argument list is "conductance".
 *
 *  [1] E.g. T. Papanastasiou, G. Georiou, and A. Alexandrou,
 *  "Viscous Fluid Flow" CRC Press, Boca Raton, Florida (2000).
 *
 *****************************************************************************/

__host__ int util_rectangle_conductance(double w, double h, double * q) {

  int ierr = 0;
  PI_DOUBLE(pi);

  assert(q);

  if (w < h || w <= 0.0 || h <= 0.0) {
    ierr = 1;
  }
  else {
    double b = 0.5*w;
    double c = 0.5*h;
    double asum = 0;
    /* As the terms in the sum decrease with k, run loop in reverse */
    /* Two thousand terms should converge to DBL_EPISLON*sum */
    for (int k = 2000; k > 0; k--) {
      double ak = 0.5*(2*k - 1)*pi;
      asum += tanh(ak*b/c)/pow(ak, 5);
    }

    *q = (4.0/3.0)*b*c*c*c*(1.0 - 6.0*(c/b)*asum);
  }

  return ierr;
}
