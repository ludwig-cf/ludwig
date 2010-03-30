/*****************************************************************************
 *
 *  colloids.c
 *
 *  Basic memory management and cell list routines for particle code.
 *
 *  $Id: colloids.c,v 1.9.4.2 2010-03-30 03:49:40 kevin Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk).
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "colloids.h"

const  int n_halo_       = 2;   /* Number of halo cells (one at each end) */
static int nalloc_colls_ = 0;   /* Number of colloids currently alloacted */
static int nalloc_links_ = 0;   /* Number of links currently allocated */
static int cifac_;
static int cjfac_;
static int ncell[3];            /* Width of cell list (minus halos) */
static int N_colloid_ = 0;      /* The global number of colloids */

static double lcell[3];
static Colloid ** cell_list_;   /* Cell list for colloids */
static double rho0_ = 1.0;      /* Colloid density */

/*****************************************************************************
 *
 *  colloids_init
 *
 *  Initialise the cell list.
 *
 *****************************************************************************/

void colloids_init() {

  double lcellmin;          /* Minimum width of cell */
  int    n;

  /* Look for minimum cell list width  in the user input */

  n = RUN_get_double_parameter("cell_list_lmin", &lcellmin);

  info("\nColloid cell list\n");

  if (n != 0) {
    info("[User   ] Requested minimum cell width is %f\n", lcellmin);
  }
  else {
    /* Fall back to a default */
    lcellmin = 6.0;
    info("[Default] Requested minimum cell width is %f\n", lcellmin);
  }

  /* Work out the number and width of the cells */

  ncell[X] = L(X) / (cart_size(X)*lcellmin);
  ncell[Y] = L(Y) / (cart_size(Y)*lcellmin);
  ncell[Z] = L(Z) / (cart_size(Z)*lcellmin);

  lcell[X] = L(X) / (cart_size(X)*ncell[X]);
  lcell[Y] = L(Y) / (cart_size(Y)*ncell[Y]);
  lcell[Z] = L(Z) / (cart_size(Z)*ncell[Z]);

  info("[       ] Actual local number of cells [%d,%d,%d]\n",
       ncell[X], ncell[Y], ncell[Z]);
  info("[       ] Actual local cell width      [%.2f,%.2f,%.2f]\n",
       lcell[X], lcell[Y], lcell[Z]);

  /* For particle halo swaps require at least one;
   * for halo sums, require at least two, so two is the minimum. */

  if (ncell[X] < 2 || ncell[Y] < 2 || ncell[Z] < 2) {
    info("[Error  ] Please check the cell width (cell_list_lmin).\n");
    fatal("[Stop] Must be at least two cells in each direction.\n");
  }

  /* Set the total number of cells and allocate the cell list */

  n = (ncell[X] + n_halo_)*(ncell[Y] + n_halo_)*(ncell[Z] + n_halo_);

  cell_list_ = (Colloid **) calloc(n, sizeof(Colloid *));

  if (cell_list_ == (Colloid **) NULL) fatal("calloc(cell_list)");

  cjfac_ = (ncell[Z] + n_halo_);
  cifac_ = (ncell[Y] + n_halo_)*cjfac_;

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

  Colloid * p_colloid;
  Colloid * p_tmp;
  int ic, jc, kc;

  info("\nReleasing colloids...\n");

  for (ic = 0; ic <= ncell[X]+1; ic++) {
    for (jc = 0; jc <= ncell[Y]+1; jc++) {
      for (kc = 0; kc <= ncell[Z]+1; kc++) {

	p_colloid = CELL_get_head_of_list(ic, jc, kc);

	while (p_colloid) {
	  p_tmp = p_colloid->next;
	  free_colloid(p_colloid);
	  p_colloid = p_tmp;
	}
	/* Next cell */
      }
    }
  }

  colloids_memory_report();

  /* Finally, free the cell list */
  if (cell_list_) free(cell_list_);

  return;
}

/*****************************************************************************
 *
 *  get_N_colloid
 *
 *  Return the global number of colloids.
 *
 *****************************************************************************/

int get_N_colloid() {
  return N_colloid_;
}

/*****************************************************************************
 *
 *  set_N_colloid
 *
 *  Set the global number of colloids.
 *
 *****************************************************************************/

void set_N_colloid(const int n) {
  assert(n >= 0);
  N_colloid_ = n;
  return;
}

/*****************************************************************************
 *
 *  get_colloid_rho0
 *
 *  Return the colloid density (generally equal to that of fluid).
 *
 *****************************************************************************/

double get_colloid_rho0() {
  return rho0_;
}

/*****************************************************************************
 *
 *  cell_coords
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
 *****************************************************************************/

IVector cell_coords(FVector r) {

  IVector cell;
  FVector rc;

  rc.x = r.x - Lmin(X) + lcell[X];
  rc.y = r.y - Lmin(Y) + lcell[Y];
  rc.z = r.z - Lmin(Z) + lcell[Z];

  cell.x = (int) floor(rc.x / lcell[X]);
  cell.y = (int) floor(rc.y / lcell[Y]);
  cell.z = (int) floor(rc.z / lcell[Z]);

  cell.x -= cart_coords(X)*ncell[X];
  cell.y -= cart_coords(Y)*ncell[Y];
  cell.z -= cart_coords(Z)*ncell[Z];

  return cell;
}

/*****************************************************************************
 *
 *  cell_get_colloid
 *
 *  Return a pointer to the colloid (may be NULL) at the head of
 *  the list at (ic, jc, kc).
 *
 *****************************************************************************/

Colloid * CELL_get_head_of_list(const int ic, const int jc, const int kc) {

  return cell_list_[ic*cifac_ + jc*cjfac_ + kc];
}


/*****************************************************************************
 *
 *  cell_insert_colloid
 *
 *  Insert a Colloid into a cell determined by its position.
 *  The list is kept in order of increasing Colloid index.
 *
 *****************************************************************************/

void cell_insert_colloid(Colloid * p_new) {

  Colloid * p_current;
  Colloid * p_previous;
  IVector   cell;
  int       cindex;

  cell   = cell_coords(p_new->r);
  cindex = cell.x*cifac_ + cell.y*cjfac_ + cell.z;

  if (cell.x < 0 || cell.x > (ncell[X] + 1) ||
      cell.y < 0 || cell.y > (ncell[Y] + 1) ||
      cell.z < 0 || cell.z > (ncell[Z] + 1)) {
    verbose("*** Dubious cell index %d position %d %d %d\n", cindex,
	   cell.x, cell.y, cell.z);
    verbose("*** Particle %d position %g %g %g\n", p_new->index, p_new->r.x,
	    p_new->r.y, p_new->r.z);
  }

  p_current = cell_list_[cindex];
  p_previous = p_current;

  while (p_current) {
    if (p_new->index < p_current->index) break;
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
 *  Ncell
 *
 *  Return the number of cells in the given coordinate direction
 *  in the current sub-domain.
 *
 *****************************************************************************/

int Ncell(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return ncell[dim];
}

/*****************************************************************************
 *
 *  Lcell
 *
 *  Return the cell width in the specified coordinate direction.
 *
 *****************************************************************************/

double Lcell(const int dim) {
  assert(dim == X || dim == Y || dim == Z);
  return lcell[dim];
}

/*****************************************************************************
 *
 *  allocate_colloid
 *
 *  Allocate space for a colloid structure and return a pointer to
 *  it (or fail gracefully).
 *
 *****************************************************************************/

Colloid * allocate_colloid() {

  Colloid * p_colloid;

  p_colloid = (Colloid *) malloc(sizeof(Colloid));
  if (p_colloid == (Colloid *) NULL) fatal("malloc(Colloid) failed\n");

  nalloc_colls_++;

  return p_colloid;
}

/*****************************************************************************
 *
 *  allocate_boundary_link
 *
 *  Return a pointer to a newly allocated COLL_Link structure
 *  or fail gracefully.
 *
 *****************************************************************************/

COLL_Link * allocate_boundary_link() {

  COLL_Link * p_link;

  p_link = (COLL_Link *) malloc(sizeof(COLL_Link));
  if (p_link == (COLL_Link *) NULL) fatal("malloc(Coll_link) failed\n");

  nalloc_links_++;

  return p_link;
}

/*****************************************************************************
 *
 *  free_colloid
 *
 *  Destroy an unwanted colloid. Note that this is the only way
 *  that boundary links should be freed, so that the current
 *  link count is maintained correctly. 
 *
 *****************************************************************************/

void free_colloid(Colloid * p_colloid) {

  COLL_Link * p_link;
  COLL_Link * tmp;

  if (p_colloid == NULL) fatal("Trying to free NULL colloid\n");

  /* Free any links, then the colloid itself */

  p_link = p_colloid->lnk;
    
  while (p_link != NULL) {
    tmp = p_link->next;
    free(p_link);
    p_link = tmp;
    nalloc_links_--;
  }

  free(p_colloid);
  nalloc_colls_--;

  return;
}

/*****************************************************************************
 *
 *  colloids_report_memory
 *
 *  Show the current total for colloids and links.
 *
 *****************************************************************************/

void colloids_memory_report() {

  info("[colloids: %d (%d bytes) links: %d (%d bytes)]\n",
       nalloc_colls_, nalloc_colls_*sizeof(Colloid),
       nalloc_links_, nalloc_links_*sizeof(COLL_Link));

  return;
}


/*****************************************************************************
 *
 *  update_cells
 *
 *  Look for particles which have changed cell lists following the
 *  last position update. Move as necessary, or remove if the
 *  particle has left the domain completely.
 *
 *****************************************************************************/

void cell_update() {

  Colloid * p_colloid;
  Colloid * p_previous;
  Colloid * tmp;

  IVector   cell;
  int       cl_old, cl_new;
  int       ic, jc, kc;
  int       destroy;

  for (ic = 0; ic <= ncell[X] + 1; ic++) {
    for (jc = 0; jc <= ncell[Y] + 1; jc++) {
      for (kc = 0; kc <= ncell[Z] + 1; kc++) {

	cl_old = kc + cjfac_*jc + cifac_*ic;

	p_colloid = cell_list_[cl_old];
	p_previous = p_colloid;

	while (p_colloid) {

	  cell = cell_coords(p_colloid->r);

	  destroy = (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
		     cell.x > ncell[X] + 1 ||
		     cell.y > ncell[Y] + 1 || cell.z > ncell[Z] + 1);
	  
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

	    free_colloid(p_colloid);
	    p_colloid = tmp;
	  }
	  else {

	    cl_new = cell.z + cjfac_*cell.y + cifac_*cell.x;

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

	      /*
	      p_colloid->next = cell_list_[cl_new];
	      cell_list_[cl_new] = p_colloid;
	      */
	      cell_insert_colloid(p_colloid);

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


/*****************************************************************************
 *
 *  colloid_add_local
 *
 *  Add a colloid only if the position is in the local domain.
 *  (and not in the halo).
 *
 *  This returns a pointer to a new colloid, or NULL if none
 *  is added.
 *
 *****************************************************************************/

Colloid * colloid_add_local(const int index, const double r[3]) {

  IVector cell;
  Colloid * p_c = NULL;
  FVector rnew;

  rnew.x = r[X];
  rnew.y = r[Y];
  rnew.z = r[Z];

  cell = cell_coords(rnew);

  if (cell.x < 1 || cell.x > Ncell(X)) return p_c;
  if (cell.y < 1 || cell.y > Ncell(Y)) return p_c;
  if (cell.z < 1 || cell.z > Ncell(Z)) return p_c;

  p_c = allocate_colloid();

  /* Put the new colloid at the head of the appropriate cell list */

  p_c->index = index;
  p_c->r     = rnew;
  p_c->lnk   = NULL;

  cell_insert_colloid(p_c);

  return p_c;
}
