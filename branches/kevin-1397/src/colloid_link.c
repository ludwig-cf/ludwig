/*****************************************************************************
 *
 *  colloid_link.h
 *
 *  Colloid boundary link structure.
 *
 *  $Id: colloid_link.c,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
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
