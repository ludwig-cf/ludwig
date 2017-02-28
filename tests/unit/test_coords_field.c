/*****************************************************************************
 *
 *  test_coords_field.c
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>

#include "util.h"
#include "coords.h"
#include "memory.h"
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

int test_coords_field_set(cs_t * cs, int nf, void * buf, MPI_Datatype mpidata,
			  halo_ft bufset) {
  int n;
  int nsites;
  int nlocal[3];
  int noffst[3];
  int ic, jc, kc, index, indexf;

  size_t sz;
  unsigned char * fc = (unsigned char *) buf;

  assert(cs);
  assert (nf >= 0);
  assert(fc);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);
  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffst);
  cs_nsites(cs, &nsites);

  /* Set values in the domain proper (not halo regions) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);

        for (n = 0; n < nf; n++) {
	  indexf = mem_addr_rank1(nsites, nf, index, n); 
          bufset(cs, noffst[X] + ic, noffst[Y] + jc, noffst[Z] + kc, n,
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

int test_coords_field_check(cs_t * cs, int nhcomm, int nf, void * buf,
			    MPI_Datatype mpidata, halo_ft bufref) {
  int n;
  int nsites;
  int nlocal[3];
  int noffst[3];
  int ic, jc, kc, index, indexf;
  size_t sz;

  char   cref, cact;            /* Reference, actual, function value */
  double dref, dact;            /* Reference, actual, function value */
  unsigned char * bufc = (unsigned char *) buf;

  assert(cs);
  assert(nf >= 0);
  assert(bufc);
  assert(mpidata == MPI_CHAR || mpidata == MPI_DOUBLE);
  if (mpidata == MPI_CHAR) sz = sizeof(char);
  if (mpidata == MPI_DOUBLE) sz = sizeof(double);

  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffst);
  cs_nsites(cs, &nsites);

  for (ic = 1 - nhcomm; ic <= nlocal[X] + nhcomm; ic++) {
    for (jc = 1 - nhcomm; jc <= nlocal[Y] + nhcomm; jc++) {
      for (kc = 1 - nhcomm; kc <= nlocal[Z] + nhcomm; kc++) {

        index = cs_index(cs, ic, jc, kc);

        if (mpidata == MPI_CHAR) {
          for (n = 0; n < nf; n++) {
	    indexf = mem_addr_rank1(nsites, nf, index, n);
            bufref(cs, noffst[X] + ic, noffst[Y] + jc, noffst[Z] + kc, n, &cref);
            cact =  bufc[sz*indexf];
            assert(cref == cact);
          }
        }

        if (mpidata == MPI_DOUBLE) {
          for (n = 0; n < nf; n++) {
	    indexf = mem_addr_rank1(nsites, nf, index, n);
            bufref(cs, noffst[X] + ic, noffst[Y] + jc, noffst[Z] + kc, n, &dref);
            dact = *((double *) (bufc + sz*indexf));
	    /*printf("%2d %2d %2d %14.7e %14.7e\n", ic, jc, kc, dref, dact);*/
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

int test_ref_char1(cs_t * cs, int ic, int jc, int kc, int n, void * ref) {

  int ntotal[3];
  int iref;
  char * c = (char *) ref;

  assert(cs);
  assert(c);

  cs_ntotal(cs, ntotal);

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

int test_ref_double1(cs_t * cs, int ic, int jc, int kc, int n, void * ref) {

  double * d = (double *) ref;
  double ltot[3];
  PI_DOUBLE(pi);

  assert(d);

  cs_ltot(cs, ltot);

  *d = cos(2.0*pi*ic/ltot[X]) + cos(2.0*pi*jc/ltot[Y]) + cos(2.0*pi*kc/ltot[Z]);
  *d += 1.0*n;

  return 0;
}
