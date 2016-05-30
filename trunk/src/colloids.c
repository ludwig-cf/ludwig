/*****************************************************************************
 *
 *  colloids.c
 *
 *  Basic memory management and cell list routines for particle code.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2015 The University of Edinburgh
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
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
#include "colloids_s.h"
#define RHO_DEFAULT 1.0
#define DRMAX_DEFAULT 0.8


__targetHost__ int colloid_create(colloids_info_t * cinfo, colloid_t ** pc);
__targetHost__ void colloid_free(colloids_info_t * cinfo, colloid_t * pc);

/*****************************************************************************
 *
 *  colloids_info_create
 *
 *****************************************************************************/

__targetHost__ int colloids_info_create(int ncell[3], colloids_info_t ** pinfo) {

  int nhalo = 1;                   /* Always exactly one halo cell each side */
  int nlist;
  colloids_info_t * obj = NULL;

  assert(pinfo);

  obj = (colloids_info_t*) calloc(1, sizeof(colloids_info_t));
  if (obj == NULL) fatal("calloc(colloids_info_t) failed\n");

  /* allocate target copy of structure */

  targetCalloc((void**) &(obj->tcopy), sizeof(colloids_info_t));

  /* Defaults */

  obj->nhalo = nhalo;
  obj->ncell[X] = ncell[X];
  obj->ncell[Y] = ncell[Y];
  obj->ncell[Z] = ncell[Z];

  obj->str[Z] = 1;
  obj->str[Y] = obj->str[Z]*(ncell[Z] + 2*nhalo);
  obj->str[X] = obj->str[Y]*(ncell[Y] + 2*nhalo);

  nlist = (ncell[X] + 2*nhalo)*(ncell[Y] + 2*nhalo)*(ncell[Z] + 2*nhalo);
  obj->clist = (colloid_t**) calloc(nlist, sizeof(colloid_t *));
  if (obj->clist == NULL) fatal("calloc(nlist, colloid_t *) failed\n");

  obj->ncells = nlist;
  obj->rho0 = RHO_DEFAULT;
  obj->drmax = DRMAX_DEFAULT;

  *pinfo = obj;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_free
 *
 *****************************************************************************/

__targetHost__ void colloids_info_free(colloids_info_t * info) {

  assert(info);

  colloids_info_cell_list_clean(info);

  free(info->clist);
  if (info->map_old) free(info->map_old);
  if (info->map_new) free(info->map_new);

  if (info->tcopy) targetFree(info->tcopy);

  free(info);

  return;
}

/*****************************************************************************
 *
 *  colloids_info_recreate
 *
 *  Move the contents of an existing cell list to a new size.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_recreate(int newcell[3], colloids_info_t ** pinfo) {

  colloids_info_t * newinfo = NULL;
  colloid_t * pc;
  colloid_t * pcnew;

  assert(pinfo);

  colloids_info_create(newcell, &newinfo);

  colloids_info_list_local_build(*pinfo);
  colloids_info_local_head(*pinfo, &pc);

  /* Need to copy all colloid state across */

  for ( ; pc; pc = pc->nextlocal) {
    colloids_info_add_local(newinfo, pc->s.index, pc->s.r, &pcnew);
    pcnew->s = pc->s;
  }

  colloids_info_ntotal_set(newinfo);
  assert(newinfo->ntotal == (*pinfo)->ntotal);

  colloids_info_free(*pinfo);
  *pinfo = newinfo;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_nallocated
 *
 *  Return number of colloid_t allocated.
 *  Just for book-keeping; there's no physical meaning.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_nallocated(colloids_info_t * cinfo, int * nallocated) {

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

__targetHost__ int colloids_info_rho0(colloids_info_t * cinfo, double * rho0) {

  assert(cinfo);
  assert(rho0);

  *rho0 = cinfo->rho0;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_rho0_set
 *
 *****************************************************************************/

__targetHost__ int colloids_info_rho0_set(colloids_info_t * cinfo, double rho0) {

  assert(cinfo);

  cinfo->rho0 = rho0;

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

__targetHost__ int colloids_info_map_init(colloids_info_t * info) {

  int nsites;
  void * tmpptr;

  assert(info);
  assert(info->tcopy);

  nsites = coords_nsites();

  info->nsites = nsites;
  info->map_old = (colloid_t **) calloc(nsites, sizeof(colloid_t *));
  info->map_new = (colloid_t **) calloc(nsites, sizeof(colloid_t *));

  if (info->map_old == (colloid_t **) NULL) fatal("calloc (map_old) failed");
  if (info->map_new == (colloid_t **) NULL) fatal("calloc (map_new) failed");

  /* allocate data space on target */

  targetCalloc((void**) &tmpptr, nsites*sizeof(colloid_t*));
  copyToTarget(&(info->tcopy->map_new), &tmpptr, sizeof(colloid_t**)); 

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_ntotal
 *
 *****************************************************************************/

__targetHost__ int colloids_info_ntotal(colloids_info_t * info, int * ntotal) {

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

__targetHost__ int colloids_info_ncell(colloids_info_t * info, int ncell[3]) {

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

__targetHost__ int colloids_info_lcell(colloids_info_t * cinfo, double lcell[3]) {

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

__targetHost__ int colloids_info_cell_index(colloids_info_t * cinfo, int ic, int jc, int kc) {

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

__targetHost__ int colloids_info_map(colloids_info_t * info, int index, colloid_t ** pc) {

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

__targetHost__ int colloids_info_map_old(colloids_info_t * info, int index, colloid_t ** pc) {

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

__targetHost__ int colloids_info_map_set(colloids_info_t * cinfo, int index, colloid_t * pc) {

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

__targetHost__ int colloids_info_map_update(colloids_info_t * cinfo) {

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

__targetHost__ int colloids_info_nhalo(colloids_info_t * cinfo, int * nhalo) {

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

__targetHost__ int colloids_info_nlocal(colloids_info_t * cinfo, int * nlocal) {

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

__targetHost__ int colloids_info_ntotal_set(colloids_info_t * cinfo) {

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

__targetHost__ int colloids_info_cell_list_head(colloids_info_t * cinfo,
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

__targetHost__ int colloids_info_cell_coords(colloids_info_t * cinfo,
					     const double r[3], int icell[3]) {
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

__targetHost__ int colloids_info_insert_colloid(colloids_info_t * cinfo, colloid_t * coll) {

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

__targetHost__ int colloids_info_cell_list_clean(colloids_info_t * cinfo) {

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

__targetHost__ int colloids_info_update_cell_list(colloids_info_t * cinfo) {

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

__targetHost__ int colloids_info_add_local(colloids_info_t * cinfo, int index,
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

__targetHost__ int colloids_info_add(colloids_info_t * cinfo, int index,
				     const double r[3], colloid_t ** pc) {

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

__targetHost__ int colloid_create(colloids_info_t * cinfo, colloid_t ** pc) {

  colloid_t * obj = NULL;

  targetCallocUnified((void**) &obj, sizeof(colloid_t));  

  cinfo->nallocated += 1;
  *pc = obj;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_free
 *
 *****************************************************************************/

__targetHost__ void colloid_free(colloids_info_t * cinfo, colloid_t * pc) {

  assert(cinfo);
  assert(pc);

  colloid_link_free_list(pc->lnk);
  targetFree(pc);

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

__targetHost__ int colloids_info_q_local(colloids_info_t * cinfo, double q[2]) {

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

__targetHost__ int colloids_info_cell_count(colloids_info_t * cinfo, int ic, int jc, int kc,
			     int * ncount) {

  colloid_t * pc = NULL;

  assert(cinfo);
  assert(ncount);

  *ncount = 0;
  colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);
  for (; pc; pc = pc->next) *ncount += 1;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_local_head
 *
 *****************************************************************************/

__targetHost__ int colloids_info_local_head(colloids_info_t * cinfo, colloid_t ** pc) {

  assert(cinfo);
  assert(pc);

  *pc = cinfo->headlocal;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_all_head
 *
 *****************************************************************************/

int colloids_info_all_head(colloids_info_t * cinfo, colloid_t ** pc) {

  assert(cinfo);
  assert(pc);

  *pc = cinfo->headall;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_position_update
 *
 *  Update the colloid positions (all cells).
 *
 *  Moving a particle more than 1 lattice unit in any direction can
 *  cause it to leave the cell list entirely, which ends in
 *  catastrophe. We therefore have a check here against a maximum
 *  velocity (effectively drmax) and stop if the check fails.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_position_update(colloids_info_t * cinfo) {

  int ia;
  int ic, jc, kc;
  int ncell[3];
  int nhalo;
  int ifail;

  colloid_t * coll;

  assert(cinfo);

  colloids_info_ncell(cinfo, ncell);
  colloids_info_nhalo(cinfo, &nhalo);

  for (ic = 1 - nhalo; ic <= ncell[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= ncell[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= ncell[Z] + nhalo; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &coll);

	while (coll) {

	  if (coll->s.isfixedr == 0) {
	    ifail = 0;
	    for (ia = 0; ia < 3; ia++) {
	      if (coll->s.dr[ia] > cinfo->drmax) ifail = 1;
	      coll->s.r[ia] += coll->s.dr[ia];
	      /* This should trap NaNs */
	      if (coll->s.dr[ia] != coll->s.dr[ia]) ifail = 1;
	    }

	    if (ifail == 1) {
	      verbose("Colloid velocity exceeded max %14.7e\n", cinfo->drmax);
	      colloid_state_write_ascii(coll->s, stdout);
	      fatal("Stopping\n");
	    }
	  }

	  coll = coll->next;
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_update_lists
 *
 *****************************************************************************/

__targetHost__ int colloids_info_update_lists(colloids_info_t * cinfo) {

  colloids_info_list_local_build(cinfo);
  colloids_info_list_all_build(cinfo);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_list_all_build
 *
 *  Update the linear 'all' list from the current cell list.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_list_all_build(colloids_info_t * cinfo) {

  int n;
  colloid_t * pc;
  colloid_t * lastcell = NULL;  /* Last colloid, last cell */

  assert(cinfo);

  /* Nullify the entire list and locate the first colloid. */

  cinfo->headall = NULL;

  for (n = 0; n < cinfo->ncells; n++) {
    for (pc = cinfo->clist[n]; pc; pc = pc->next) {
      if (cinfo->headall == NULL) cinfo->headall = pc;
      pc->nextall = NULL;
    }
  }

  /* Now link up the all the individual cell lists via nextall */

  for (n = 0; n < cinfo->ncells; n++) {

    for (pc = cinfo->clist[n]; pc; pc = pc->next) {
      if (lastcell) {
	lastcell->nextall = pc;
	lastcell = NULL;
      }

      if (pc->next) {
	pc->nextall = pc->next;
      }
      else {
	lastcell = pc;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_list_local_build
 *
 *  Update the linear 'local' list from the current cell list.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_list_local_build(colloids_info_t * cinfo) {

  int ic, jc, kc;
  colloid_t * pc;
  colloid_t * lastcell = NULL;

  assert(cinfo);

  /* Find first local colloid; nullify any existing list */

  cinfo->headlocal = NULL;

  for (ic = 1; ic <= cinfo->ncell[X]; ic++) {
    for (jc = 1; jc <= cinfo->ncell[Y]; jc++) {
      for (kc = 1; kc <= cinfo->ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	for (; pc; pc = pc->next) {
	  if (cinfo->headlocal == NULL) cinfo->headlocal = pc;
	  pc->nextlocal = NULL;
	}
      }
    }
  }

  /* Now link up the list via nextlocal */

  lastcell = NULL;

  for (ic = 1; ic <= cinfo->ncell[X]; ic++) {
    for (jc = 1; jc <= cinfo->ncell[Y]; jc++) {
      for (kc = 1; kc <= cinfo->ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);

	for (; pc; pc = pc->next) {
	  if (lastcell) {
	    lastcell->nextlocal = pc;
	    lastcell = NULL;
	  }

	  if (pc->next) {
	    pc->nextlocal = pc->next;
	  }
	  else {
	    lastcell = pc;
	  }
	}

      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_climits
 *
 *****************************************************************************/

__targetHost__ int colloids_info_climits(colloids_info_t * cinfo, int ia, int ic,
			  int * lim) {

  int irange, halo;

  assert(cinfo);
  assert(ia == X || ia == Y || ia == Z);

  irange = 1 + (cinfo->ncell[ia] == 2);
  halo = (cart_size(ia) > 1 || irange == 1);

  lim[0] = imax(1 - halo, ic - irange);
  lim[1] = imin(ic + irange, cinfo->ncell[ia] + halo);

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_a0max
 *
 *  Find the largest input radius a0 currently present and return.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_a0max(colloids_info_t * cinfo, double * a0max) {

  double a0_local = 0.0;
  colloid_t * pc = NULL;

  assert(cinfo);
  assert(a0max);

  /* Make sure lists are up-to-date */
  colloids_info_update_lists(cinfo);

  colloids_info_local_head(cinfo, &pc);
  for (; pc; pc = pc->next) a0_local = dmax(a0_local, pc->s.a0);

  MPI_Allreduce(&a0_local, a0max, 1, MPI_DOUBLE, MPI_MAX, cart_comm());

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_ahmax
 *
 *  Find the largest ah present and return.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_ahmax(colloids_info_t * cinfo, double * ahmax) {

  double ahmax_local;
  colloid_t * pc = NULL;

  assert(cinfo);

  ahmax_local = 0.0;

  /* Make sure lists are up-to-date */
  colloids_info_update_lists(cinfo);

  colloids_info_local_head(cinfo, &pc);
  for (; pc; pc = pc->next) ahmax_local = dmax(ahmax_local, pc->s.ah);

  MPI_Allreduce(&ahmax_local, ahmax, 1, MPI_DOUBLE, MPI_MAX, pe_comm());

  return 0;
}

/*****************************************************************************
 *
 *  colloids_info_count_local
 *
 *  Return number of local colloids of given type.
 *
 *****************************************************************************/

__targetHost__ int colloids_info_count_local(colloids_info_t * cinfo, colloid_type_enum_t it,
			      int * count) {

  int nlocal = 0;
  int ic, jc, kc;
  colloid_t * pc = NULL;

  assert(cinfo);

  for (ic = 1; ic <= cinfo->ncell[X]; ic++) {
    for (jc = 1; jc <= cinfo->ncell[Y]; jc++) {
      for (kc = 1; kc <= cinfo->ncell[Z]; kc++) {

	colloids_info_cell_list_head(cinfo, ic, jc, kc, &pc);
	for (; pc; pc = pc->next) {
	  if (pc->s.type == it) nlocal += 1;
	}
      }
    }
  }

  *count = nlocal;

  return 0;
}

/*****************************************************************************
 *
 *  colloids_number_sites
 *
 *  returns the total number of lattice sites affected by colloids
 *
 *****************************************************************************/

__targetHost__ int colloids_number_sites(colloids_info_t *cinfo)
{
  

  int ia;
  int i, j, ij, ji;

  colloid_t * pc;
  colloid_link_t * p_link;

  /* All colloids, including halo */
  colloids_info_all_head(cinfo, &pc);
 
  int ncolsite=0;

  for ( ; pc; pc = pc->nextall) {

    p_link = pc->lnk;

    for (; p_link; p_link = p_link->next) {

      if (p_link->status == LINK_UNUSED) continue;

      /* increment by 2 (outward and inward sites) */
      ncolsite+=2;

     
    }
  }

  return ncolsite;

}

/*****************************************************************************
 *
 *  colloid_list_sites
 *
 *  provides a list of lattice site indexes affected by colloids 
 *
 *****************************************************************************/

__targetHost__ void colloids_list_sites(int* colloidSiteList, colloids_info_t *cinfo)
{
  

  int ia;
  int i, j, ij, ji;

  colloid_t * pc;
  colloid_link_t * p_link;

  /* All colloids, including halo */  
  colloids_info_all_head(cinfo, &pc);
 
  int ncolsite=0;

  for ( ; pc; pc = pc->nextall) {

    p_link = pc->lnk;

    for (; p_link; p_link = p_link->next) {

      if (p_link->status == LINK_UNUSED) continue;

      colloidSiteList[ncolsite++]= p_link->i;
      colloidSiteList[ncolsite++]= p_link->j;

    }
  }
  return;
}
