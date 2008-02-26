/***************************************************************************
 *
 *  hydrodynamics.c
 *
 *  Deals with the hydrodynamic sector quantities one would expect
 *  in Navier Stokes, rho, u, ...
 *
 *  $Id: lattice.c,v 1.7.2.2 2008-02-26 09:41:08 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 ***************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "lattice.h"

struct vector * fl_force; /* Force on fluid nodes at current time step. */
struct vector * fl_u;     /* The fluid velocity field (at lattice sites). */


static double total_bytes;     /* bytes currently allocated */

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

  assert(fl_force[index].c != NULL);
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

  assert(fl_force[index].c != NULL);
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

  assert(fl_u[index].c != NULL);
  for (n = 0; n < 3; n++) u[n] = fl_u[index].c[n];

  return;
}
