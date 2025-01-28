/*****************************************************************************
 *
 *  colloid_link.h
 *
 *  Colloid boundary link structure.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2025 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "colloid_link.h"

static int nlinks_ = 0;   /* Total currently allocated */

/*****************************************************************************
 *
 *  colloid_link_allocate
 *
 *****************************************************************************/

colloid_link_t * colloid_link_allocate(void) {

  colloid_link_t * p_link;

  p_link = (colloid_link_t *) malloc(sizeof(colloid_link_t));
  assert(p_link);
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

  colloid_link_t * tmp;

  while (p) {
    tmp = p->next;
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

int colloid_link_total(void) {

  return nlinks_;
}

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

  int pi = 4;                            /* This is approximate */
  int ai = fmax(4.0, ceil(a));           /* A minimum reasonable a ~ 4 */

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

  int pi = 4;                             /* This is approximate */
  int ai = fmax(1.0, ceil(a));            /* A minimum reasonable a ~ 1.0 */

  return 4*pi*ai*ai*(nvel - 1)/2;
}
