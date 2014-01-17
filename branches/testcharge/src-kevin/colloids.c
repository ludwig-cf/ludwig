/*****************************************************************************
 *
 *  colloids.c
 *
 *  Basic memory management and cell list routines for particle code.
 *
 *  $Id: colloids.c,v 1.11 2010-10-15 12:40:02 kevin Exp $
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
#include "util.h"
#include "colloids.h"

static const int nhalo_ = 1;      /* Number of halo cells (one at each end) */
static const double rho0_ = 1.0;  /* Colloid density */

static int ncell_[3] = {2, 2, 2}; /* Width of cell list (minus halos) */
static int ntotal_ = 0;           /* Total (physical) number of colloids */
static int nalloc_ = 0;           /* No. colloids currently allocated */

static colloid_t ** cell_list_;   /* Cell list for colloids */

#ifndef OLD_ONLY

#define RHO_DEFAULT 1.0

struct colloids_info_s {
  int nhalo;                  /* Halo extent in cell list */
  int ntotal;                 /* Total, physical, number of colloids */
  int nallocated;             /* Number colloid_t allocated */
  int ncell[3];               /* Number of cells (excluding  2*halo) */
  int str[3];                 /* Strides for cell list */
  int nsites;                 /* Total number of map sites */
  double rho0;                /* Mean density (usually matches fluid) */

  colloid_t ** clist;         /* Cell list pointers */
  colloid_t ** map_old;       /* Map (previous time step) pointers */
  colloid_t ** map_new;       /* Map (current time step) pointers */
};

int colloid_create(colloids_info_t * cinfo, colloid_t ** pc);
void colloid_free(colloids_info_t * cinfo, colloid_t * pc);

/*****************************************************************************
 *
 *  colloids_info_create
 *
 *****************************************************************************/

int colloids_info_create(int ncell[3], colloids_info_t ** pinfo) {

  int nhalo = 1;                   /* Always exactly one halo cell each side */
  int nlist;
  colloids_info_t * obj = NULL;

  assert(pinfo);

  obj = calloc(1, sizeof(colloids_info_t));
  if (obj == NULL) fatal("calloc(colloids_info_t) failed\n");

  /* Defaults */

  obj->nhalo = nhalo;
  obj->ncell[X] = ncell[X];
  obj->ncell[Y] = ncell[Y];
  obj->ncell[Z] = ncell[Z];

  obj->str[Z] = 1;
  obj->str[Y] = obj->str[Z]*(ncell[Z] + 2*nhalo);
  obj->str[X] = obj->str[Y]*(ncell[Y] + 2*nhalo);

  nlist = (ncell[X] + 2*nhalo)*(ncell[Y] + 2*nhalo)*(ncell[Z] + 2*nhalo);
  obj->clist = calloc(nlist, sizeof(colloid_t *));
  if (obj->clist == NULL) fatal("calloc(nlist, colloid_t *) failed\n");

  obj->rho0 = RHO_DEFAULT;

  *pinfo = obj;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_free
 *
 *****************************************************************************/

void colloids_info_free(colloids_info_t * info) {

  assert(info);

  colloids_info_cell_list_clean(info);

  free(info->clist);
  if (info->map_old) free(info->map_old);
  if (info->map_new) free(info->map_new);
  free(info);

  return;
}

/*****************************************************************************
 *
 *  colloids_info_nallocated
 *
 *  Return number of colloid_t allocated.
 *  Just for book-keeping; there's no physical meaning.
 *
 *****************************************************************************/

int colloids_info_nallocated(colloids_info_t * cinfo, int * nallocated) {

  assert(cinfo);
  assert(nallocated);

  *nallocated = cinfo->nallocated;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_rho0
 *
 *****************************************************************************/

int colloids_info_rho0(colloids_info_t * cinfo, double * rho0) {

  assert(cinfo);
  assert(rho0);

  *rho0 = cinfo->rho0;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_map_init
 *
 *  Allocated separately from the main structure, as not always
 *  required.
 *
 *****************************************************************************/

int colloids_info_map_init(colloids_info_t * info) {

  int nsites;

  assert(info);

  nsites = coords_nsites();

  info->nsites = nsites;
  info->map_old = (colloid_t **) calloc(nsites, sizeof(colloid_t *));
  info->map_new = (colloid_t **) calloc(nsites, sizeof(colloid_t *));

  if (info->map_old == (colloid_t **) NULL) fatal("calloc (map_old) failed");
  if (info->map_new == (colloid_t **) NULL) fatal("calloc (map_new) failed");

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_ntotal
 *
 *****************************************************************************/

int colloids_info_ntotal(colloids_info_t * info, int * ntotal) {

  assert(info);
  assert(ntotal);

  *ntotal = info->ntotal;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_ncell
 *
 *****************************************************************************/

int colloids_info_ncell(colloids_info_t * info, int ncell[3]) {

  assert(info);

  ncell[X] = info->ncell[X];
  ncell[Y] = info->ncell[Y];
  ncell[Z] = info->ncell[Z];

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_lcell
 *
 *  Return the extent of the (uniform) cells in each direction.
 *
 *****************************************************************************/

int colloids_info_lcell(colloids_info_t * cinfo, double lcell[3]) {

  assert(cinfo);
  assert(lcell);

  lcell[X] = L(X)/(cart_size(X)*cinfo->ncell[X]);
  lcell[Y] = L(Y)/(cart_size(Y)*cinfo->ncell[Y]);
  lcell[Z] = L(Z)/(cart_size(Z)*cinfo->ncell[Z]);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_cell_index
 *
 *****************************************************************************/

int colloids_info_cell_index(colloids_info_t * cinfo, int ic, int jc, int kc) {

  int index;

  assert(cinfo);
  assert(ic >= 0); assert(ic < cinfo->ncell[X] + 2*cinfo->nhalo);
  assert(jc >= 0); assert(jc < cinfo->ncell[Y] + 2*cinfo->nhalo);
  assert(kc >= 0); assert(kc < cinfo->ncell[Z] + 2*cinfo->nhalo);

  index = cinfo->str[X]*ic + cinfo->str[Y]*jc + cinfo->str[Z]*kc;

  return index;
}

/*****************************************************************************
 *
 *  colloids_info_map
 *
 *  Look at the pointer map for site index (current time step).
 *
 *****************************************************************************/

int colloids_info_map(colloids_info_t * info, int index, colloid_t ** pc) {

  assert(info);
  assert(pc);

  *pc = NULL;
  if (info->map_new) *pc = info->map_new[index];

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_map_old
 *
 *  Look at the pointer map for site index (old time step).
 *
 *****************************************************************************/

int colloids_info_map_old(colloids_info_t * info, int index, colloid_t ** pc) {

  assert(info);
  assert(pc);

  *pc = NULL;
  if (info->map_old) *pc = info->map_old[index];

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_map_set
 *
 *  Colloid pc may be NULL.
 *
 *****************************************************************************/

int colloids_info_map_set(colloids_info_t * cinfo, int index, colloid_t * pc) {

  assert(cinfo);
  assert(cinfo->map_new);
  assert(index >= 0);
  assert(index < cinfo->nsites);

  cinfo->map_new[index] = pc;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_map_update
 *
 *****************************************************************************/

int colloids_info_map_update(colloids_info_t * cinfo) {

  int n;
  colloid_t ** maptmp;

  assert(cinfo);

  for (n = 0; n < cinfo->nsites; n++) {
    cinfo->map_old[n] = NULL;
  }

  maptmp = cinfo->map_old;
  cinfo->map_old = cinfo->map_new;
  cinfo->map_new = maptmp;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_nhalo
 *
 *  The halo extent of the cell list.
 *
 *****************************************************************************/

int colloids_info_nhalo(colloids_info_t * cinfo, int * nhalo) {

  assert(cinfo);
  assert(nhalo);

  *nhalo = cinfo->nhalo;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_nlocal
 *
 *  Return the local number of colloids. As the colloids move about,
 *  this must be recomputed each time.
 *
 ****************************************************************************/

int colloids_info_nlocal(colloids_info_t * cinfo, int * nlocal) {

  int ic, jc, kc;
  colloid_t * pc = NULL;

  assert(cinfo);
  assert(nlocal);

  *nlocal = 0;

  for (ic = 1; ic <= cinfo->ncell[X]; ic++) {
    for (jc = 1; jc <= cinfo->ncell[Y]; jc++) {
      for (kc = 1; kc <= cinfo->ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);
	for (; pc; pc = pc->next) *nlocal += 1;

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_ntotal_set
 *
 *  Set the global number of colloids from the current list.
 *
 *****************************************************************************/

int colloids_info_ntotal_set(colloids_info_t * cinfo) {

  int nlocal;

  assert(cinfo);

  colloids_info_nlocal(cinfo, &nlocal);
  MPI_Allreduce(&nlocal, &cinfo->ntotal, 1, MPI_INT, MPI_SUM, cart_comm());

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_cell_list_head
 *
 *****************************************************************************/

int colloids_info_cell_list_head(colloids_info_t * cinfo,
				 int ic, int jc, int kc, colloid_t ** pc) {
  int index;

  assert(cinfo);
  assert(pc);

  index = colloids_info_cell_index(cinfo, ic, jc, kc);
  *pc = cinfo->clist[index];

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_cell_coords
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

int colloids_info_cell_coords(colloids_info_t * cinfo, const double r[3],
			      int icell[3]) {
  int ia;
  double lcell;

  assert(cinfo);

  for (ia = 0; ia < 3; ia++) {
    lcell = L(ia) / (cart_size(ia)*cinfo->ncell[ia]);
    icell[ia] = (int) floor((r[ia] - Lmin(ia) + lcell) / lcell);
    icell[ia] -= cart_coords(ia)*cinfo->ncell[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_insert_colloid
 *
 *  Insert a colloid_t into a cell determined by its position.
 *  The list is kept in order of increasing Colloid index.
 *
 *****************************************************************************/

int colloids_info_insert_colloid(colloids_info_t * cinfo, colloid_t * coll) {

  int index;
  int newcell[3];
  colloid_t * p_current;
  colloid_t * p_previous;

  assert(cinfo);
  assert(coll);

  colloids_info_cell_coords(cinfo, coll->s.r, newcell);
  index = colloids_info_cell_index(cinfo, newcell[X], newcell[Y], newcell[Z]);

  p_current = cinfo->clist[index];
  p_previous = p_current;

  while (p_current) {
    if (coll->s.index < p_current->s.index) break;
    p_previous = p_current;
    p_current = p_current->next;
  }

  /* Insert new colloid at head of list or between existing members. */

  if (p_current == cinfo->clist[index]) {
    coll->next = cinfo->clist[index];
    cinfo->clist[index] = coll;
  }
  else {
    coll->next = p_current;
    p_previous->next = coll;
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_cell_list_clean
 *
 *****************************************************************************/

int colloids_info_cell_list_clean(colloids_info_t * cinfo) {

  int ic, jc, kc;
  colloid_t * pc;
  colloid_t * ptmp;

  assert(cinfo);

  for (ic = 1 - cinfo->nhalo; ic <= cinfo->ncell[X] + cinfo->nhalo; ic++) {
    for (jc = 1 - cinfo->nhalo; jc <= cinfo->ncell[Y] + cinfo->nhalo; jc++) {
      for (kc = 1 - cinfo->nhalo; kc <= cinfo->ncell[Z] + cinfo->nhalo; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  ptmp = pc->next;
	  colloid_free(cinfo, pc);
	  pc = ptmp;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_update_cell_list
 *
 *  Look for particles which have changed cell lists following the
 *  last position update. Move as necessary, or remove if the
 *  particle has left the domain completely.
 *
 *****************************************************************************/

int colloids_info_update_cell_list(colloids_info_t * cinfo) {

  int ic, jc, kc;
  int cell[3];
  int cl_old, cl_new;
  int destroy;

  colloid_t * p_colloid;
  colloid_t * p_previous;
  colloid_t * tmp;

  assert(cinfo);

  for (ic = 1 - cinfo->nhalo; ic <= cinfo->ncell[X] + cinfo->nhalo; ic++) {
    for (jc = 1 - cinfo->nhalo; jc <= cinfo->ncell[Y] + cinfo->nhalo; jc++) {
      for (kc = 1 - cinfo->nhalo; kc <= cinfo->ncell[Z] + cinfo->nhalo; kc++) {

	cl_old = colloids_info_cell_index(cinfo, ic, jc, kc);

	p_colloid = cinfo->clist[cl_old];
	p_previous = p_colloid;

	while (p_colloid) {
	  colloids_info_cell_coords(cinfo, p_colloid->s.r, cell);
	  destroy = (cell[X] < 1 - cinfo->nhalo ||
		     cell[Y] < 1 - cinfo->nhalo ||
		     cell[Z] < 1 - cinfo->nhalo ||
		     cell[X] > cinfo->ncell[X] + cinfo->nhalo ||
		     cell[Y] > cinfo->ncell[Y] + cinfo->nhalo ||
		     cell[Z] > cinfo->ncell[Z] + cinfo->nhalo);

	  if (destroy) {
	    /* This particle should be unlinked and removed. */

	    tmp = p_colloid->next;
	    if (p_colloid == cinfo->clist[cl_old]) {
	      cinfo->clist[cl_old] = tmp;
	      p_previous = tmp;
	    }
	    else {
	      p_previous->next = tmp;
	    }

	    colloid_free(cinfo, p_colloid);
	    p_colloid = tmp;
	  }
	  else {
	    cl_new = cinfo->str[Z]*cell[Z]
	      + cinfo->str[Y]*cell[Y]
	      + cinfo->str[X]*cell[X];

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

	      if (p_colloid == cinfo->clist[cl_old]) {
		cinfo->clist[cl_old] = tmp;
		p_previous = tmp;
	      }
	      else {
		p_previous->next = tmp;
	      }

	      colloids_info_insert_colloid(cinfo, p_colloid);

	      p_colloid = tmp;
	    }
	  }

	  /* Next colloid */
	}

	/* Next cell */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_add_local
 *
 *  Return a pointer to a new colloid, if r is in the local domain.
 *  Index is the (unique) id for the new colloid.
 *
 *  If r[3] is not in the local domain, no colloid is added, and
 *  *pc is returned unchanged (NULL).
 *
 *****************************************************************************/

int colloids_info_add_local(colloids_info_t * cinfo, int index,
			    const double r[3], colloid_t ** pc) {
  int is_local = 1;
  int icell[3];

  assert(cinfo);
  assert(pc);

  colloids_info_cell_coords(cinfo, r, icell);

  assert(cinfo->nhalo == 1); /* Following would need to be adjusted */

  if (icell[X] < 1 || icell[X] > cinfo->ncell[X]) is_local = 0;
  if (icell[Y] < 1 || icell[Y] > cinfo->ncell[Y]) is_local = 0;
  if (icell[Z] < 1 || icell[Z] > cinfo->ncell[Z]) is_local = 0;

  *pc = NULL;
  if (is_local) colloids_info_add(cinfo, index, r, pc);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_add
 *
 *  The colloid must have an index, and it must have a position.
 *
 *****************************************************************************/

int colloids_info_add(colloids_info_t * cinfo, int index, const double r[3],
		      colloid_t ** pc) {

  int icell[3];

  assert(cinfo);
  assert(pc);

  colloids_info_cell_coords(cinfo, r, icell);

  assert(icell[X] >= 1 - cinfo->nhalo);
  assert(icell[Y] >= 1 - cinfo->nhalo);
  assert(icell[Z] >= 1 - cinfo->nhalo);
  assert(icell[X] < cinfo->ncell[X] + 2*cinfo->nhalo);
  assert(icell[Y] < cinfo->ncell[Y] + 2*cinfo->nhalo);
  assert(icell[Z] < cinfo->ncell[Z] + 2*cinfo->nhalo);

  colloid_create(cinfo, pc);
  (*pc)->s.index = index;

  (*pc)->s.r[X] = r[X];
  (*pc)->s.r[Y] = r[Y];
  (*pc)->s.r[Z] = r[Z];

  (*pc)->s.rebuild = 1;

  colloids_info_insert_colloid(cinfo, *pc);

  return 0;
}

/*****************************************************************************
 *
 *  colloid_create
 *
 *  Allocate space for a colloid structure and return a pointer to
 *  it (or fail gracefully). Use calloc to ensure everything is
 *  zero and pointers are NULL.
 *
 *****************************************************************************/

int colloid_create(colloids_info_t * cinfo, colloid_t ** pc) {

  colloid_t * obj = NULL;

  obj = (colloid_t *) calloc(1, sizeof(colloid_t));
  if (obj == (colloid_t *) NULL) fatal("calloc(colloid_t) failed\n");

  cinfo->nallocated += 1;
  *pc = obj;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_free
 *
 *****************************************************************************/

void colloid_free(colloids_info_t * cinfo, colloid_t * pc) {

  assert(cinfo);
  assert(pc);

  colloid_link_free_list(pc->lnk);
  free(pc);

  cinfo->nallocated -= 1;

  return;
}
/*****************************************************************************
 *
 *  colloids_info_q_local
 *
 *  Add up the local charge for exactly two valencies q[2].
 *  That is, charge associated with local colloids.
 *
 *****************************************************************************/

int colloids_info_q_local(colloids_info_t * cinfo, double q[2]) {

  int ic, jc, kc;
  colloid_t * pc = NULL;

  assert(cinfo);
  assert(q);

  q[0] = 0.0;
  q[1] = 0.0;

  for (ic = 1; ic <= cinfo->ncell[X]; ic++) {
    for (jc = 1; jc <= cinfo->ncell[Y]; jc++) {
      for (kc = 1; kc <= cinfo->ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  q[0] += pc->s.q0;
	  q[1] += pc->s.q1;
	  pc = pc->next;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_v_local
 *
 *  Add up and return the net volume of the colloids in the local domain.
 *  Note this is not the volume occupied on the local lattice, it is
 *  just the sum of the discrete volumes for all particles.
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

int colloids_info_v_local(colloids_info_t * cinfo, double * v) {

  int ic, jc, kc;
  double vol;
  colloid_t * pc = NULL;

  assert(cinfo);
  assert(v);
  *v = 0.0;

  for (ic = 1; ic <= cinfo->ncell[X]; ic++) {
    for (jc = 1; jc <= cinfo->ncell[Y]; jc++) {
      for (kc = 1; kc <= cinfo->ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	while (pc) {
	  util_discrete_volume_sphere(pc->s.r, pc->s.a0, &vol);
	  *v += vol;
	  pc = pc->next;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_cell_count
 *
 *  Just count the number of colloids current in cell (ic, jc, kc).
 *
 *****************************************************************************/

int colloids_info_cell_count(colloids_info_t * cinfo, int ic, int jc, int kc,
			     int * ncount) {

  colloid_t * pc = NULL;

  assert(cinfo);
  assert(ncount);

  *ncount = 0;
  colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);
  for (; pc; pc = pc->next) *ncount += 1;

  return 0;
}

#endif




#ifdef OLD_ONLY

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
#ifdef OLD_ONLY
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
#endif
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

/*****************************************************************************
 *
 *  colloids_q_local
 *
 *  Add up the local charge for exactly two valencies.
 *
 *****************************************************************************/

int colloids_q_local(double q[2]) {

  int ic, jc, kc;
  colloid_t * pc;

  q[0] = 0.0;
  q[1] = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	pc = colloids_cell_list(ic, jc, kc);

	while (pc) {
	  q[0] += pc->s.q0;
	  q[1] += pc->s.q1;
	  pc = pc->next;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_v_local
 *
 *  Add up and return the net volume of the colloids in the local domain.
 *  Note this is not the volume occupied on the local lattice, it is
 *  just the sum of the discrete volumes for all particles.
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

int colloids_v_local(double * v) {

  int ic, jc, kc;
  double vol;
  colloid_t * pc;

  assert(v);
  *v = 0.0;

  for (ic = 1; ic <= Ncell(X); ic++) {
    for (jc = 1; jc <= Ncell(Y); jc++) {
      for (kc = 1; kc <= Ncell(Z); kc++) {

	pc = colloids_cell_list(ic, jc, kc);

	while (pc) {
	  util_discrete_volume_sphere(pc->s.r, pc->s.a0, &vol);
	  *v += vol;
	  pc = pc->next;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_map_new
 *
 *  Look at the pointer map for site index (current time step).
 *
 *****************************************************************************/

int colloids_map_new(int index, colloid_t ** pc) {

  assert(pc);
  assert(0);
  *pc = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_map_old
 *
 *  Look at the pointer map for site index (old time step).
 *
 *****************************************************************************/

int colloids_map_old(int index, colloid_t ** pc) {

  assert(pc);
  assert(0);
  *pc = NULL;

  return 0;
}

#endif
