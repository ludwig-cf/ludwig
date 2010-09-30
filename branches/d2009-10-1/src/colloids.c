/*****************************************************************************
 *
 *  colloids.c
 *
 *  Basic memory management and cell list routines for particle code.
 *
 *  $Id: colloids.c,v 1.9.4.13 2010-09-30 18:04:55 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk).
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "colloids.h"

static const int nhalo_ = 1;      /* Number of halo cells (one at each end) */
static const double rho0_ = 1.0;  /* Colloid density */

static int ncell_[3] = {2, 2, 2}; /* Width of cell list (minus halos) */
static int ntotal_ = 0;           /* Total (physical) number of colloids */
static int nalloc_ = 0;           /* No. colloids currently allocated */

static colloid_t ** cell_list_;   /* Cell list for colloids */

/*****************************************************************************
 *
 *  colloids_init
 *
 *  Initialise the cell list.
 *
 *****************************************************************************/

void colloids_init() {

  int n;

  /* Set the total number of cells and allocate the cell list */

  n = (ncell_[X] + 2*nhalo_)*(ncell_[Y] + 2*nhalo_)*(ncell_[Z] + 2*nhalo_);

  cell_list_ = (colloid_t **) calloc(n, sizeof(colloid_t *));
  if (cell_list_ == (colloid_t **) NULL) fatal("calloc(cell_list)");

  return;
}

/*****************************************************************************
 *
 *  colloids_finish
 *
 *  Free all the memory used and report.
 *
 *****************************************************************************/

void colloids_finish() {

  int ic, jc, kc;
  colloid_t * p_colloid;
  colloid_t * p_tmp;

  for (ic = 0; ic <= ncell_[X]+1; ic++) {
    for (jc = 0; jc <= ncell_[Y]+1; jc++) {
      for (kc = 0; kc <= ncell_[Z]+1; kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  p_tmp = p_colloid->next;
	  colloid_free(p_colloid);
	  p_colloid = p_tmp;
	}
	/* Next cell */
      }
    }
  }

  /* Finally, free the cell list */
  if (cell_list_) free(cell_list_);

  ncell_[X] = 2;
  ncell_[Y] = 2;
  ncell_[Z] = 2;
  ntotal_ = 0;

  return;
}

/*****************************************************************************
 *
 *  colloids_info
 *
 *****************************************************************************/

void colloids_info(void) {

  info("colloids_info:\n");
  info("Cells:             %d %d %d\n", ncell_[X], ncell_[Y], ncell_[Z]);
  info("Halo:              %d\n", nhalo_);
  info("Total (global):    %d\n", ntotal_);
  info("Allocated (local): %d\n", nalloc_);
  info("\n");

  return;
}

/*****************************************************************************
 *
 *  colloids_cell_ncell_set
 *
 *  There must be at least 2 cells locally for the purposes of
 *  communications.
 *
 *****************************************************************************/

void colloids_cell_ncell_set(const int n[3]) {

  int ia;

  for (ia = 0; ia < 3; ia++) {
    if (n[ia] < 2) fatal("Trying to set ncell_[%d] < 2\n", n[ia]);
    ncell_[ia] = n[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_ntotal
 *
 *  Return the global number of colloids.
 *
 *****************************************************************************/

int colloid_ntotal(void) {
  return ntotal_;
}

/*****************************************************************************
 *
 *  colloids_ntotal_set
 *
 *  Set the global number of colloids from the current list.
 *
 *****************************************************************************/

void colloids_ntotal_set(void) {

  int nlocal;

  nlocal = colloid_nlocal();
  MPI_Allreduce(&nlocal, &ntotal_, 1, MPI_INT, MPI_SUM, cart_comm());

  return;
}

/*****************************************************************************
 *
 *  colloid_rho0
 *
 *  Return the colloid density (generally equal to that of fluid).
 *
 *****************************************************************************/

double colloid_rho0(void) {
  return rho0_;
}

/*****************************************************************************
 *
 *  colloids_cell_list
 *
 *  Return a pointer to the colloid (may be NULL) at the head of
 *  the list at (ic, jc, kc).
 *
 *****************************************************************************/

colloid_t * colloids_cell_list(const int ic, const int jc, const int kc) {

  int xstride;
  int ystride;

  assert(ic >= 0);
  assert(ic < ncell_[X] + 2*nhalo_);
  assert(jc >= 0);
  assert(jc < ncell_[Y] + 2*nhalo_);
  assert(kc >= 0);
  assert(kc < ncell_[Z] + 2*nhalo_);
  assert(cell_list_);

  ystride = (ncell_[Z] + 2*nhalo_);
  xstride = (ncell_[Y] + 2*nhalo_)*ystride;
  return cell_list_[ic*xstride + jc*ystride + kc];
}


/*****************************************************************************
 *
 *  colloids_cell_insert_colloid
 *
 *  Insert a colloid_t into a cell determined by its position.
 *  The list is kept in order of increasing Colloid index.
 *
 *****************************************************************************/

void colloids_cell_insert_colloid(colloid_t * p_new) {

  colloid_t * p_current;
  colloid_t * p_previous;
  int cell[3];
  int       cindex;

  int cifac;
  int cjfac;

  assert(cell_list_);

  cjfac = (ncell_[Z] + 2*nhalo_);
  cifac = (ncell_[Y] + 2*nhalo_)*cjfac;

  colloids_cell_coords(p_new->s.r, cell);
  cindex = cell[X]*cifac + cell[Y]*cjfac + cell[Z];

  p_current = cell_list_[cindex];
  p_previous = p_current;

  while (p_current) {
    if (p_new->s.index < p_current->s.index) break;
    p_previous = p_current;
    p_current = p_current->next;
  }

  /* Insert new colloid at head of list or between existing members. */

  if (p_current == cell_list_[cindex]) {
    p_new->next = cell_list_[cindex];
    cell_list_[cindex] = p_new;
  }
  else {
    p_new->next = p_current;
    p_previous->next = p_new;
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_cell_ncell
 *
 *****************************************************************************/

void colloids_cell_ncell(int n[3]) {

  n[X] = ncell_[X];
  n[Y] = ncell_[Y];
  n[Z] = ncell_[Z];

  return;
}

/*****************************************************************************
 *
 *  Ncell
 *
 *  Return the number of cells in the given coordinate direction
 *  in the current sub-domain.
 *
 *  Pending deletion. Use the above.
 *
 *****************************************************************************/

int Ncell(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return ncell_[dim];
}

/*****************************************************************************
 *
 *  colloids_lcell
 *
 *  Return the cell width in the specified coordinate direction.
 *
 *****************************************************************************/

double colloids_lcell(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return L(dim)/(cart_size(dim)*ncell_[dim]);
}

/*****************************************************************************
 *
 *  colloid_allocate
 *
 *  Allocate space for a colloid structure and return a pointer to
 *  it (or fail gracefully). Use calloc to ensure everything is
 *  zero and pointers are NULL.
 *
 *****************************************************************************/

colloid_t * colloid_allocate(void) {

  colloid_t * p_colloid;

  p_colloid = (colloid_t *) calloc(1, sizeof(colloid_t));
  if (p_colloid == (colloid_t *) NULL) fatal("calloc(colloid_t) failed\n");

  nalloc_++;

  return p_colloid;
}

/*****************************************************************************
 *
 *  colloid_free
 *
 *****************************************************************************/

void colloid_free(colloid_t * p_colloid) {

  assert(p_colloid);

  colloid_link_free_list(p_colloid->lnk);

  free(p_colloid);
  nalloc_--;

  return;
}

/*****************************************************************************
 *
 *  colloids_cell_update
 *
 *  Look for particles which have changed cell lists following the
 *  last position update. Move as necessary, or remove if the
 *  particle has left the domain completely.
 *
 *****************************************************************************/

void colloids_cell_update(void) {

  colloid_t * p_colloid;
  colloid_t * p_previous;
  colloid_t * tmp;

  int cell[3];
  int       cl_old, cl_new;
  int       ic, jc, kc;
  int       destroy;
  int       cifac;
  int       cjfac;

  cjfac = (ncell_[Z] + 2*nhalo_);
  cifac = (ncell_[Y] + 2*nhalo_)*cjfac;

  for (ic = 0; ic <= ncell_[X] + 1; ic++) {
    for (jc = 0; jc <= ncell_[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell_[Z] + 1; kc++) {

	cl_old = kc + cjfac*jc + cifac*ic;

	p_colloid = cell_list_[cl_old];
	p_previous = p_colloid;

	while (p_colloid) {
	  colloids_cell_coords(p_colloid->s.r, cell);
	  destroy = (cell[X] < 0 || cell[Y] < 0 || cell[Z] < 0 ||
		     cell[X] > ncell_[X] + 1 ||
		     cell[Y] > ncell_[Y] + 1 || cell[Z] > ncell_[Z] + 1);

	  if (destroy) {
	    /* This particle should be unlinked and removed. */

	    tmp = p_colloid->next;
	    if (p_colloid == cell_list_[cl_old]) {
	      cell_list_[cl_old] = tmp;
	      p_previous = tmp;
	    }
	    else {
	      p_previous->next = tmp;
	    }

	    colloid_free(p_colloid);
	    p_colloid = tmp;
	  }
	  else {
	    cl_new = cell[Z] + cjfac*cell[Y] + cifac*cell[X];

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

	      if (p_colloid == cell_list_[cl_old]) {
		cell_list_[cl_old] = tmp;
		p_previous = tmp;
	      }
	      else {
		p_previous->next = tmp;
	      }

	      colloids_cell_insert_colloid(p_colloid);

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
 *  colloid_nlocal
 *
 *  Return the local number of colloids. As the colloids move about,
 *  this must be recomputed each time.
 *
 ****************************************************************************/

int colloid_nlocal(void) {

  int       ic, jc, kc;
  int       nlocal;
  colloid_t * p_colloid;

  nlocal = 0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	p_colloid = colloids_cell_list(ic, jc, kc);

	while (p_colloid) {
	  nlocal++;
	  p_colloid = p_colloid->next;
	}
      }
    }
  }

  return nlocal;
}

/*****************************************************************************
 *
 *  colloid_add
 *
 *  The colloid must have an index, and it must have a position.
 *
 *****************************************************************************/

colloid_t * colloid_add(const int index, const double r[3]) {

  int       icell[3];
  colloid_t * p_colloid;

  colloids_cell_coords(r, icell);

  assert(icell[X] >= 0);
  assert(icell[Y] >= 0);
  assert(icell[Z] >= 0);
  assert(icell[X] < Ncell(X) + 2*nhalo_);
  assert(icell[Y] < Ncell(Y) + 2*nhalo_);
  assert(icell[Z] < Ncell(Z) + 2*nhalo_);

  p_colloid = colloid_allocate();
  p_colloid->s.index = index;

  p_colloid->s.r[X] = r[X];
  p_colloid->s.r[Y] = r[Y];
  p_colloid->s.r[Z] = r[Z];

  p_colloid->s.rebuild = 1;
  p_colloid->lnk = NULL;
  p_colloid->next = NULL;

  colloids_cell_insert_colloid(p_colloid);

  return p_colloid;
}

/*****************************************************************************
 *
 *  colloid_add_local
 *
 *  Return a pointer to a new colloid, if r is in the local domain.
 *
 *****************************************************************************/

colloid_t * colloid_add_local(const int index, const double r[3]) {

  int icell[3];

  colloids_cell_coords(r, icell);
  assert(nhalo_ == 1); /* Following would need to be adjusted */
  if (icell[X] < 1 || icell[X] > Ncell(X)) return NULL;
  if (icell[Y] < 1 || icell[Y] > Ncell(Y)) return NULL;
  if (icell[Z] < 1 || icell[Z] > Ncell(Z)) return NULL;

  return colloid_add(index, r);
}

/*****************************************************************************
 *
 *  colloids_cell_coords
 *
 *  For position vector r in the global coordinate system
 *  return the coordinates (ic,jc,kc) of the corresponding
 *  cell in the local cell list.
 *
 *  If the position is not in the local domain, then neither
 *  is the cell. The caller must handle this.
 *
 *  09/01/07 floor() is problematic when colloid exactly on cell
 *           boundary and r is negative, ie., cell goes to -1
 *
 *****************************************************************************/

void colloids_cell_coords(const double r[3], int icell[3]) {

  int ia;
  double lcell;

  for (ia = 0; ia < 3; ia++) {
    lcell = colloids_lcell(ia);
    icell[ia] = (int) floor((r[ia] - Lmin(ia) + lcell) / lcell);
    icell[ia] -= cart_coords(ia)*ncell_[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  colloids_cell_count
 *
 *  Count the number of particles in this cell.
 *
 *****************************************************************************/

int colloids_cell_count(const int ic, const int jc, const int kc) {

  int n;
  colloid_t * pc;

  n = 0;
  pc = colloids_cell_list(ic, jc, kc);

  while (pc) {
    n++;
    pc = pc->next;
  }

  return n;
}

/*****************************************************************************
 *
 *  colloids_nalloc
 *
 *****************************************************************************/

int colloids_nalloc(void) {

  return nalloc_;
}
