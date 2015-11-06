/*****************************************************************************
 *
 *  test_coords_field.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#include "coords.h"
#include "coords_field.h"
#include "test_coords_field.h"

/*****************************************************************************
 *
 *  test_coords_field_set
 *
 *  Set the values of a field in the domain proper (not halos). There
 *  are nf components, and the field data (buf) are of type mpidata.
 *
 *  As we need a complete type for address arithmetic, the input
 *  buf is cast to char, and the sizeof the MPI data type used to
 *  address it.
 *
 *  The supplied function bufset provides the value.
 *
 *****************************************************************************/

int test_coords_field_set(int nf, void * buf, MPI_Datatype mpidata,
			  halo_ft bufset) {
  int n;
  int nhalo;
  int nlocal[3];
  int noffst[3];
  int ic, jc, kc, index, indexf;

  size_t sz;
  unsigned char * fc = (unsigned char *) buf;

  assert (nf >= 0);
  assert(fc);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);
  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffst);

  /* Set values in the domain proper (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);

        for (n = 0; n < nf; n++) {
          coords_field_index(index, n, nf, &indexf);
          bufset(noffst[X] + ic, noffst[Y] + jc, noffst[Z] + kc, n,
                 fc + sz*indexf);
        }

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  test_coords_field_check
 *
 *  All points are checked, i.e., interior points should not have changed,
 *  and halo points (up to extent nhcomm) should be correctly updated.
 *
 *  Some of the differences coming from periodic boundaries are slightly
 *  larger than DBL_EPSILON for double datatype,  but FLT_EPSILON should
 *  catch gross errors.
 *
 *  As we need to deference the incoming buffer buf, it is cast to
 *  char, and the sizeof the MPI_Datatype is used.
 *
 *****************************************************************************/

int test_coords_field_check(int nhcomm, int nf, void * buf,
			    MPI_Datatype mpidata, halo_ft bufref) {
  int n;
  int nlocal[3];
  int noffst[3];
  int ic, jc, kc, index, indexf;
  size_t sz;

  char   cref, cact;            /* Reference, actual, function value */
  double dref, dact;            /* Reference, actual, function value */
  unsigned char * bufc = (unsigned char *) buf;

  assert(nhcomm <= coords_nhalo());
  assert(nf >= 0);
  assert(bufc);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);
  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  coords_nlocal(nlocal);
  coords_nlocal_offset(noffst);

  for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
    for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
      for (kc = 1 - nhcomm; kc <= nlocal[Z] + nhcomm; kc++) {

        index = coords_index(ic, jc, kc);

        if (mpidata == MPI_CHAR) {
          for (n = 0; n < nf; n++) {
            coords_field_index(index, n, nf, &indexf);
            bufref(noffst[X] + ic, noffst[Y] + jc, noffst[Z] + kc, n, &cref);
            cact =  bufc[sz*indexf];
            assert(cref == cact);
          }
        }

        if (mpidata == MPI_DOUBLE) {
          for (n = 0; n < nf; n++) {
            coords_field_index(index, n, nf, &indexf);
            bufref(noffst[X] + ic, noffst[Y] + jc, noffst[Z] + kc, n, &dref);
            dact = *((double *) (bufc + sz*indexf));
            assert(fabs(dact - dref) < FLT_EPSILON);
          }
        }

      }
    }
  }

  return 0;
}


/*****************************************************************************
 *
 *  test_ref_char1
 *
 *  Reference function of signature halo_ft for char.
 *  The function is a periodic function of ic, ie, x only.
 *
 *****************************************************************************/

int test_ref_char1(int ic, int jc, int kc, int n, void * ref) {

  int ntotal[3];
  int iref;
  char * c = (char *) ref;

  assert(c);
  coords_ntotal(ntotal);

  iref = ic;
  if (iref <= 0) iref += ntotal[X];
  if (iref > ntotal[X]) iref -= ntotal[X];

  iref += n;
  assert(iref >= 0);
  assert(iref < CHAR_MAX);

  *c = iref;

  return 0;
}

/*****************************************************************************
 *
 *  test_ref_double1
 *
 *  Periodic function of type halo_ft for data type double.
 *
 *****************************************************************************/

int test_ref_double1(int ic, int jc, int kc, int n, void * ref) {

  double * d = (double *) ref;

  assert(d);

  *d = cos(2.0*M_PI*ic/L(X)) + cos(2.0*M_PI*jc/L(Y)) + cos(2.0*M_PI*kc/L(Z));
  *d += 1.0*n;

  return 0;
}
