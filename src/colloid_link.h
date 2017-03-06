/*****************************************************************************
 *
 *  colloid_link.h
 *
 *  The implementation is exposed for the time being.
 *
 *  $Id: colloid_link.h,v 1.2 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_LINK_H
#define LUDWIG_COLLOID_LINK_H

typedef struct colloid_link_type colloid_link_t;

struct colloid_link_type {

  int    i;               /* Index of lattice site outside colloid */ 
  int    j;               /* Index of lattice site inside */
  int    p;               /* Index of velocity connecting i -> j */
  int    status;          /* What is at site i (fluid, solid, etc) */
  double rb[3];           /* Vector connecting centre of colloid and
			   * centre of the boundary link */

  colloid_link_t * spare; /* Unused */
  colloid_link_t * next;  /* Linked list */
};

enum link_status {LINK_FLUID, LINK_COLLOID, LINK_BOUNDARY, LINK_UNUSED}; 

colloid_link_t * colloid_link_allocate(void);
void             colloid_link_free_list(colloid_link_t * link);
int              colloid_link_count(colloid_link_t * link);
int              colloid_link_total(void);

#endif
