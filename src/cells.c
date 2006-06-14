/*****************************************************************************
 *
 *  cells.c
 *
 *  Colloid cell list.
 *
 *  Colloids are stored in a standard cell list (common in molecular
 *  dynamics) which is simply an array of pointers to Colloid which
 *  identify the Colloid at the head of the current list (or NULL).
 *
 *  The cell list serves two important purposes:
 *    (1) it is used to locate colloid-colloid interactions a la
 *        molecular dynamics in O(N) time,
 *    (2) it is used as the basis of the parallelisation for the
 *        colloidal particles.
 *
 *  The list array is indexed in one dimension, but it is more
 *  useful to think of a three-dimensional array indexed
 *  _colloids_cl [ic,jc,kc] when performing loops, where
 *  the indices refer to [x, y, z] directions.
 *
 *  Direct manipulation of the pointer list is discouraged; use
 *  the access functions CELL_get_head_of_list() and
 *  CELL_insert_at_head_of_list() instead.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "utilities.h"
#include "colloids.h"
#include "cells.h"
#include "cmem.h"

#include "pe.h"
#include "coords.h"
#include "cartesian.h"

#define    NMAX_LIST 16     /* Maximum number of particles in a list */

static Colloid ** _colloids_cl;    /* Cell list for colloids */

/*****************************************************************************
 *
 *  CELL_init_cells
 *
 *  Initialise the main cell list. The domain must be split into
 *  a whole number of cells which have a minimum width (lcellmin).
 *  The minimum width is determined by the largest particle size
 *  and any interaction ranges that must be captured. In turn,
 *  this determines the overall number of cells (Ncell).
 *
 *  There are a number of constraints:
 *
 *   1. there must be a whole number of cells in each dimension,
 *      (and at least 3 to get correct pairwise interactions);
 *   2. there must be at least 3 cells in each dimension for
 *      the parallel halo regions to work;
 *   3. there are two cells added in each dimension to act as halo regions.
 *   4. Otherwise, try to maximise the number of cells.
 *
 *   Issues
 *   - Could be adjusted to accommodate solid walls.
 *   - Could take the number of cells from user input.
 *
 *****************************************************************************/

void CELL_init_cells() {

  IVector ncl;               /* Number of cells in each dimension */
  double  lcellmin;          /* Minimum width of cell */
  int     ncells;            /* Working number of cells */

  /* Compute the number of cells in each dimension and their
   * eventual width. The minimum width is ... */

  lcellmin = 2.0*Global_Colloid.ah + Global_Colloid.r_lu_n;

  /* Check there are at least three cells in each dimension. */

  ncl.x = L(X) / (cart_size(X)*lcellmin);
  ncl.y = L(Y) / (cart_size(Y)*lcellmin);
  ncl.z = L(Z) / (cart_size(Z)*lcellmin);

  if (ncl.x < 3) fatal("Ncells.x = %d (3 minimum)\n", ncl.x);
  if (ncl.y < 3) fatal("Ncells.y = %d (3 minumum)\n", ncl.y);
  if (ncl.z < 3) fatal("Ncells.z = %d (3 minumum)\n", ncl.z);

  Global_Colloid.Ncell.x = ncl.x;
  Global_Colloid.Ncell.y = ncl.y;
  Global_Colloid.Ncell.z = ncl.z;

  Global_Colloid.Lcell.x = L(X) / (cart_size(X)*ncl.x);
  Global_Colloid.Lcell.y = L(Y) / (cart_size(Y)*ncl.y);
  Global_Colloid.Lcell.z = L(Z) / (cart_size(Z)*ncl.z);
 
  info("\nCELL_init_cells:\n");
  info("Colloid default radius a0:        %f\n", Global_Colloid.a0);
  info("Default hydrodynamic radius ah:   %f\n", Global_Colloid.ah);
  info("Normal lubrication breakdown:     %f\n", Global_Colloid.r_lu_n);
  info("Tangential lubrication breakdown: %f\n", Global_Colloid.r_lu_t);
  info("Implicit scheme cut off:          %f\n", Global_Colloid.r_clus);
  info("Minimum cell width:               %f\n\n", lcellmin);
  info("Local number of cells             [%d,%d,%d]\n", ncl.x, ncl.y, ncl.z);
  info("Local cell width                  [%.2f,%.2f,%.2f]\n",
       Global_Colloid.Lcell.x, Global_Colloid.Lcell.y, Global_Colloid.Lcell.z);

  /* Set the total number of cells and allocate the cell list */

  ncells = (ncl.x + 2)*(ncl.y + 2)*(ncl.z + 2);

  info("Requesting %d bytes for the cell list\n", ncells*sizeof(Colloid *));
  _colloids_cl = (Colloid **) calloc(ncells, sizeof(Colloid *));

  if (_colloids_cl == (Colloid **) NULL) fatal("calloc(_colloids_cl)");

  return;
}


/*****************************************************************************
 *
 *  CELL_cell_coords
 *
 *  For position vector r in the global coordinate system
 *  return the coordinates (ic,jc,kc) of the corresponding
 *  cell in the local cell list.
 *
 *  If the position is not in the local domain, then neither
 *  is the cell. The caller must handle this.
 *
 *****************************************************************************/

IVector CELL_cell_coords(FVector r) {

  IVector cell;
  FVector rc;

  rc.x = r.x - Lmin(X) + Global_Colloid.Lcell.x;
  rc.y = r.y - Lmin(Y) + Global_Colloid.Lcell.y;
  rc.z = r.z - Lmin(Z) + Global_Colloid.Lcell.z;

  cell.x = (int) floor(rc.x / Global_Colloid.Lcell.x);
  cell.y = (int) floor(rc.y / Global_Colloid.Lcell.y);
  cell.z = (int) floor(rc.z / Global_Colloid.Lcell.z);

#ifdef _MPI_
  cell.x -= cart_coords(X)*Global_Colloid.Ncell.x;
  cell.y -= cart_coords(Y)*Global_Colloid.Ncell.y;
  cell.z -= cart_coords(Z)*Global_Colloid.Ncell.z;
#endif

  return cell;
}


/*****************************************************************************
 *
 *  CCOM_cell_index
 *
 *  Return the local index of local position (ic,jc,kc)
 *
 *****************************************************************************/

int CELL_cell_index(int ic, int jc, int kc) {

  int cifac, cjfac;

  cjfac = (Global_Colloid.Ncell.z + 2);
  cifac = (Global_Colloid.Ncell.y + 2)*cjfac;

  return (kc + cjfac*jc + cifac*ic);
}


/*****************************************************************************
 *
 *  CELL_get_head_of_list
 *
 *  Return a pointer to the colloid (may be NULL) at the head of
 *  the list at (ic, jc, kc).
 *
 *****************************************************************************/

Colloid * CELL_get_head_of_list(int ic, int jc, int kc) {

  return _colloids_cl[CELL_cell_index(ic, jc, kc)];
}


/*****************************************************************************
 *
 *  CELL_insert_at_head_of_list
 *
 *  Add a new colloid (p_colloid->next should be NULL) to the head of
 *  the appropriate list depending on its position.
 *
 *****************************************************************************/

void CELL_insert_at_head_of_list(Colloid * p_colloid) {

  IVector cell;
  int     cindex;

  cell   = CELL_cell_coords(p_colloid->r);
  cindex = CELL_cell_index(cell.x, cell.y, cell.z);

  p_colloid->next = _colloids_cl[cindex];
  _colloids_cl[cindex] = p_colloid;

  return;
}


/*****************************************************************************
 *
 *  CELL_update_cell_lists
 *
 *  Look for particles which have changed cell lists following the
 *  last position update. Move as necessary, or remove if the
 *  particle has left the domain completely.
 *
 *****************************************************************************/

void CELL_update_cell_lists() {

  Colloid * p_colloid;
  Colloid * p_previous;
  Colloid * tmp;

  IVector   ncell;
  IVector   cell;
  int       cl_old, cl_new;
  int       ic, jc, kc;
  int       cifac, cjfac;
  int       ncells;
  int       destroy;

  ncell = Global_Colloid.Ncell;

  cjfac  = (ncell.z + 2);
  cifac  = (ncell.y + 2)*cjfac;
  ncells = (ncell.x + 2)*cifac;

  for (ic = 0; ic <= ncell.x + 1; ic++) {
    for (jc = 0; jc <= ncell.y + 1; jc++) {
      for (kc = 0; kc <= ncell.z + 1; kc++) {

	cl_old = kc + cjfac*jc + cifac*ic;

	p_colloid = _colloids_cl[cl_old];
	p_previous = p_colloid;

	while (p_colloid) {

	  cell = CELL_cell_coords(p_colloid->r);

	  destroy = (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
		     cell.x > ncell.x + 1 ||
		     cell.y > ncell.y + 1 || cell.z > ncell.z + 1);
	  
	  if (destroy) {
	    /* This particle should be unlinked and removed. */

	    tmp = p_colloid->next;
	    if (p_colloid == _colloids_cl[cl_old]) {
	      _colloids_cl[cl_old] = tmp;
	      p_previous = tmp;
	    }
	    else {
	      p_previous->next = tmp;
	    }

	    CMEM_free_colloid(p_colloid);
	    p_colloid = tmp;
	  }
	  else {

	    cl_new = cell.z + cjfac*cell.y + cifac*cell.x;

	    if (cl_new == cl_old) {
	      /* No movement so next colloid */
	      p_previous = p_colloid;
	      p_colloid = p_colloid->next;
	    }
	    else {
	      /* Unlink colloid from old cell list and attach it to
	       * new one. Careful with:
	       *   1. if moving a colloid from head of a list, must
	       *      reset the head of that list;
	       *   2. remember where next colloid in old list is. */

	      tmp = p_colloid->next;

	      if (p_colloid == _colloids_cl[cl_old]) {
		_colloids_cl[cl_old] = tmp;
		p_previous = tmp;
	      }
	      else {
		p_previous->next = tmp;
	      }

	      p_colloid->next = _colloids_cl[cl_new];
	      _colloids_cl[cl_new] = p_colloid;

	      p_colloid = tmp;
	    }
	  }

	  /* Next colloid */
	}

	/* Next cell */
      }
    }
  }

  return;
}

/*****************************************************************************
 *
 *  CELL_sort_list
 *
 *  Sort the colloids in the given list into order based
 *  on increasing index (i.e., smallest index at head of the
 *  list).
 *
 *  When running in parallel, this ensures that all instances of the
 *  list are the same. In this way, data for a given colloid can be
 *  located by its order in the list on all processes.
 *
 *****************************************************************************/

void CELL_sort_list(int ic, int jc, int kc) {

  int        n, m, npart;
  int        cl;
  Colloid *  p_colloid;
  Colloid *  tmp;
  Colloid ** sortlist;

  sortlist = (Colloid **) calloc(NMAX_LIST, sizeof(Colloid *));
  if (sortlist == NULL) fatal("calloc(sortlist) failed\n");

  /* Perform an insertion sort of the particles into a temporary
   * list, then relink the actual list to reflect the sorted one. */

  npart = 0;
  cl = CELL_cell_index(ic, jc, kc);
  p_colloid = _colloids_cl[cl];

  while (p_colloid) {
    sortlist[npart] = p_colloid;
    npart++;
    p_colloid = p_colloid->next;
  }

  if (npart >= NMAX_LIST) fatal("NMAX_LIST exceeded\n");

  for (n = 0; n < npart - 1; n++) {
    for (m = 0; m < npart - n - 1; m++) {
      if (sortlist[m]->index > sortlist[m + 1]->index) {
	tmp = sortlist[m];
	sortlist[m] = sortlist[m + 1];
	sortlist[m + 1] = tmp;
      }
    }
  }

  /* Relink the original list */

  for (n = 0; n < npart; n++) {
    sortlist[n]->next = sortlist[n + 1];
  }

  if (npart > 0) {
    _colloids_cl[cl] = sortlist[0];

    VERBOSE(("CELL_sort_list(%d,%d,%d) sorted %d particle(s)",
	     ic, jc, kc, npart));
    for (n = 0; n < npart; n++) VERBOSE((" [i%d]", sortlist[n]->index));

    VERBOSE(("\n"));
  }

  free(sortlist);

  return;
}


/*****************************************************************************
 *
 *  CELL_destroy_list
 *
 *  Remove any and all particles in the given list.
 *
 *****************************************************************************/

void CELL_destroy_list(int ic, int jc, int kc) {

  int        cl;
  Colloid *  p_colloid;
  Colloid *  tmp;

  cl = CELL_cell_index(ic, jc, kc);
  p_colloid = _colloids_cl[cl];

  while (p_colloid) {
    tmp = p_colloid->next;
    CMEM_free_colloid(p_colloid);
    p_colloid = tmp;
  }

  _colloids_cl[cl] = NULL;

  return;
}


/*****************************************************************************
 *
 *  CMEM_free_all_colloids
 *
 *  Deallocate memory associated with colloid structures.
 *
 *****************************************************************************/

void CMEM_free_all_colloids() {

  Colloid * p_colloid;
  Colloid * tmp;

  int       n, ncells;

  ncells = (Global_Colloid.Ncell.x + 2)*(Global_Colloid.Ncell.y + 2)
          *(Global_Colloid.Ncell.z + 2);

  for (n = 0; n < ncells; n++) {

    p_colloid = _colloids_cl[n];

    while (p_colloid != NULL) {
      tmp = p_colloid->next;
      CMEM_free_colloid(p_colloid);
      p_colloid = tmp;
    }

    /* Nullify head of the list */
    _colloids_cl[n] = NULL;
  }

  return;
}

