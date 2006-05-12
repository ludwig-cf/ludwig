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

#include "pe.h"

#include "utilities.h"
#include "model.h"
#include "lattice.h"


FVector *  _force;        /* Force on fluid nodes at current time step. */
double  * phi_site;


static int _nalloc_mb;    /* Mbytes currently allocated */


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

  double mbyte;

  mbyte = nsite*sizeof(FVector)/1.0e+6;

  info("Requesting %f Mb for force array\n", mbyte);

  _force = (FVector *) calloc(nsite, sizeof(FVector));
  if (_force == (FVector *) NULL) fatal("calloc(_force) failed\n");

  _nalloc_mb += mbyte;

  return;
}


