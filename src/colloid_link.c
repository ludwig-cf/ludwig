/*****************************************************************************
 *
 *  colloid_link.h
 *
 *  Colloid boundary link structure.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2026 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alexei Borissov
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "colloid_link.h"

static int nlinks_ = 0; /* Total currently allocated (linked list) */

/*****************************************************************************
 *
 *  colloid_link_allocate
 *
 *****************************************************************************/

colloid_link_t * colloid_link_allocate(void) {

  colloid_link_t * p_link;

  p_link = (colloid_link_t *) calloc(1, sizeof(colloid_link_t));
  assert(p_link);
  p_link->status = LINK_UNUSED;
  nlinks_++;

  return p_link;
}

/*****************************************************************************
 *
 *  colloid_link_free_list
 *
 *  Should take the first link in the list as argument.
 *
 *****************************************************************************/

void colloid_link_free_list(colloid_link_t * p) {

  while (p) {
    colloid_link_t * tmp = p->next;
    free(p);
    p = tmp;
    nlinks_--;
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_link_count
 *
 *  Should take the first link in the list and returns the number of
 *  links.
 *
 *****************************************************************************/

int colloid_link_count(colloid_link_t * p) {

  int count = 0;

  assert(p);

  while (p) {
    count++;
    p = p->next;
  }

  return count;
}

/*****************************************************************************
 *
 *  colloid_link_total
 *
 *****************************************************************************/

int colloid_link_total(void) { return nlinks_; }

/*****************************************************************************
 *
 *  colloid_link_max_2d
 *
 *  How many links do we need to allocate for a 2d disk of radius a?
 *  if the model has nvel velocities.
 *
 *  In general, this is a complex function of the radius, and the
 *  position of the centre relative to the lattice. However, we
 *  can make an estimate based on the perimeter length 2 pi a.
 *
 *  The estimate is ticklish in the limit a -> 0, where we need
 *  at least (nvel - 1) links. However, 2d radii should probably
 *  not be less than a ~ 4.0 in real application.
 *
 *  For each unit length of perimeter, we allow (nvel - 1)/2 links
 *  (i.e, half the non-zero links possible).
 *
 *  Everything else is rounded up, as we want to ensure there are
 *  sufficient links in all cases, and don't care too much about
 *  overestimating. In contrast, an underestimate would be fatal.
 *
 *****************************************************************************/

int colloid_link_max_2d(double a, int nvel) {

  int    pi = 4;                  /* This is approximate */
  double ai = fmax(4.0, ceil(a)); /* A minimum reasonable a ~ 4 */

  return 2*pi*ai*(nvel - 1)/2;
}

/*****************************************************************************
 *
 *  colloid_link_max_3d
 *
 *  This is as for the 2d case (see comments above), except that the
 *  estimate is based on the surface area 4 pi a^2.
 *
 *  A minimum reasonable redius in 3d is a ~ 1.0.
 *
 *****************************************************************************/

int colloid_link_max_3d(double a, int nvel) {

  int    pi = 4;                  /* This is approximate */
  double ai = fmax(1.0, ceil(a)); /* A minimum reasonable a ~ 1.0 */

  return 4*pi*ai*ai*(nvel - 1)/2;
}

/*****************************************************************************
 *
 *  colloid_links_array_create
 *
 *  All managed memory.
 *  The boundary vector array is flattened as rb[3][maxlinks].
 *
 *****************************************************************************/

int colloid_links_array_create(int maxlinks, colloid_links_array_t ** array) {

  int                     ifail = 0;
  colloid_links_array_t * links = NULL;

  tdpAssert(tdpMallocManaged((void **) &links, sizeof(colloid_links_array_t),
                             tdpMemAttachGlobal));
  tdpAssert(tdpMemset(links, 0, sizeof(colloid_links_array_t)));

  ifail = colloid_links_array_initialise(maxlinks, links);

  if (ifail == 0) {
    *array = links;
  }
  else {
    tdpAssert(tdpFree(links));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_links_array_initialise
 *
 *****************************************************************************/

int colloid_links_array_initialise(int maxlinks, colloid_links_array_t * a) {

  int ifail = 0;

  if (a == NULL || maxlinks <= 0) {
    ifail = -1;
  }
  else {
    size_t sz = 0;

    a->max_links    = maxlinks;
    a->active_links = 0;

    sz = maxlinks * sizeof(int);
    tdpAssert(tdpMallocManaged((void **) &a->i, sz, tdpMemAttachGlobal));
    tdpAssert(tdpMallocManaged((void **) &a->j, sz, tdpMemAttachGlobal));
    tdpAssert(tdpMallocManaged((void **) &a->p, sz, tdpMemAttachGlobal));
    tdpAssert(tdpMallocManaged((void **) &a->status, sz, tdpMemAttachGlobal));

    for (int i = 0; i < maxlinks; i++) {
      a->i[i] = 0;
    }
    for (int i = 0; i < maxlinks; i++) {
      a->j[i] = 0;
    }
    for (int i = 0; i < maxlinks; i++) {
      a->p[i] = 0;
    }
    for (int i = 0; i < maxlinks; i++) {
      a->status[i] = LINK_UNUSED;
    }

    sz = 3 * sizeof(double *);
    tdpAssert(tdpMallocManaged((void **) &a->rb, sz, tdpMemAttachGlobal));
    sz = 3 * maxlinks * sizeof(double);
    tdpAssert(tdpMallocManaged((void **) &a->rb[0], sz, tdpMemAttachGlobal));

    a->rb[1] = a->rb[0] + maxlinks;
    a->rb[2] = a->rb[1] + maxlinks;

    for (int j = 0; j < 3; j++) {
      tdpAssert(tdpMemset(a->rb[j], 0, maxlinks * sizeof(double)));
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_links_array_finalise
 *
 *****************************************************************************/

int colloid_links_array_finalise(colloid_links_array_t * links) {

  if (links) {
    tdpAssert(tdpFree(links->i));
    tdpAssert(tdpFree(links->j));
    tdpAssert(tdpFree(links->p));
    tdpAssert(tdpFree(links->status));
    tdpAssert(tdpFree(links->rb[0]));
    tdpAssert(tdpFree(links->rb));
    *links = (colloid_links_array_t) {};
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloid_links_array_free
 *
 *****************************************************************************/

int colloid_links_array_free(colloid_links_array_t ** links) {

  assert(links);

  if (links) {
    colloid_links_array_finalise(*links);
    tdpAssert(tdpFree(*links));
    *links = NULL;
  }

  return 0;
}

/***************************************************************************
 *
 *  colloid_link_to_array
 *
 *  Copies single link to array at the specified index.
 *
 ***************************************************************************/

int colloid_link_to_array(const colloid_link_t *  link,
                          colloid_links_array_t * array, int index) {

  assert(link);
  assert(array);
  assert(0 <= index && index < array->max_links);

  array->i[index]      = link->i;
  array->j[index]      = link->j;
  array->p[index]      = link->p;
  array->status[index] = link->status;
  array->rb[X][index]  = link->rb[X];
  array->rb[Y][index]  = link->rb[Y];
  array->rb[Z][index]  = link->rb[Z];

  return 0;
}
