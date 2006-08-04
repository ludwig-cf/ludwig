/***************************************************************************
 *
 *  lattice.c
 *
 *  Deals with the allocation, etc, of the large arrays holding the
 *  fluid distributions, force, etc.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ***************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "lattice.h"


double * phi_site;

struct vector * fl_force; /* Force on fluid nodes at current time step. */
struct vector * fu;       /* The fluid velocity field (at lattice sites). */


static double total_bytes;     /* bytes currently allocated */


/***************************************************************************
 *
 *  LATT_allocate_sites
 *
 *  Allocate memory for the distributions. If MPI2 is used, then
 *  this must use the appropriate utility to accomodate LE planes.
 *
 ***************************************************************************/
 
void LATT_allocate_sites(const int nsites) {

  info("Requesting %d bytes for site data\n", nsites*sizeof(Site));

#ifdef _MPI_2_
 {
   int ifail;

   ifail = MPI_Alloc_mem(nsites*sizeof(Site), MPI_INFO_NULL, &site);
   if (ifail == MPI_ERR_NO_MEM) fatal("MPI_Alloc_mem(site) failed\n");
 }
#else

  /* Use calloc. */

  site = (Site  *) calloc(nsites, sizeof(Site));
  if (site == NULL) fatal("calloc(site) failed\n");

#endif

  return;
}

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

  fu = (struct vector *) calloc(nsite, sizeof(struct vector));
  if (fu == (struct vector *) NULL) fatal("calloc(fu) failed\n");

  total_bytes += bytes;

  return;
}
