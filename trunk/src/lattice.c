/***************************************************************************
 *
 *  lattice.c
 *
 *  Deals with the allocation, etc, of the large arrays holding the
 *  fluid distributions, force, etc.
 *
 *  $Id: lattice.c,v 1.6 2006-12-20 16:56:57 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ***************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "lattice.h"

double * phi_site;

struct vector * fl_force; /* Force on fluid nodes at current time step. */
struct vector * fl_u;     /* The fluid velocity field (at lattice sites). */


static double total_bytes;     /* bytes currently allocated */

/****************************************************************************
 *
 *  LATT_allocate_phi
 *
 *  Allocate memory for the order parameter arra. If MPI2 is used
 *  this must use MPI_Alloc_mem() to allow use of Windows in the
 *  LE code.
 *
 ****************************************************************************/

void LATT_allocate_phi(const int nsites) {

  info("Requesting %d bytes for phi_site\n", nsites*sizeof(double));

#ifdef _MPI_2_
 {
   int ifail;
   ifail = MPI_Alloc_mem(nsites*sizeof(double), MPI_INFO_NULL, &phi_site);
   if (ifail == MPI_ERR_NO_MEM) fatal("MPI_Alloc_mem(phi) failed\n");
 }
#else

  phi_site = (double *) calloc(nsites, sizeof(double));
  if (phi_site == NULL) fatal("calloc(phi) failed\n");

#endif

  return;
}

/***************************************************************************
 *
 *  LATT_allocate_force
 *
 *  Allocate memory for the force array.
 *
 ***************************************************************************/

void LATT_allocate_force(const int nsite) {

  double bytes;

  bytes = nsite*sizeof(struct vector);

  info("Requesting %.0f bytes for force array\n", bytes);

  fl_force = (struct vector *) calloc(nsite, sizeof(struct vector));
  if (fl_force == (struct vector *) NULL) fatal("calloc(fl_force) failed\n");

  total_bytes += bytes;

  return;
}

/****************************************************************************
 *
 *  set_force_at_site
 *
 *  Set the fluid force at site index
 *
 ****************************************************************************/

void set_force_at_site(const int index, double force[3]) {

  int n;

  assert(fl_force[index].c);
  for (n = 0; n < 3; n++) fl_force[index].c[n] = force[n];

  return;
}

/****************************************************************************
 *
 *  add_force_at_site
 *
 *  Accumulate the fluid force at site index
 *
 ****************************************************************************/

void add_force_at_site(const int index, double force[3]) {

  int n;

  assert(fl_force[index].c);
  for (n = 0; n < 3; n++) fl_force[index].c[n] += force[n];

  return;
}


/***************************************************************************
 *
 *  latt_allocate_velocity
 *
 *  Allocate memory for the fluid velocity field.
 *
 ***************************************************************************/

void latt_allocate_velocity(const int nsite) {

  double bytes;

  bytes = nsite*sizeof(struct vector);

  info("Requesting %.0f bytes for velocity array\n", bytes);

  fl_u = (struct vector *) calloc(nsite, sizeof(struct vector));
  if (fl_u == (struct vector *) NULL) fatal("calloc(fl_u) failed\n");

  total_bytes += bytes;

  return;
}

/****************************************************************************
 *
 *  get_velocity_at_lattice
 *
 *  Return the velcoity at site index.
 *
 ****************************************************************************/

void get_velocity_at_lattice(const int index, double u[3]) {

  int n;

  assert(fl_u[index].c);
  for (n = 0; n < 3; n++) u[n] = fl_u[index].c[n];

  return;
}
