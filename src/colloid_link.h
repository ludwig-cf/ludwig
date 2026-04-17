/*****************************************************************************
 *
 *  colloid_link.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_LINK_H
#define LUDWIG_COLLOID_LINK_H

typedef struct colloid_links_array_s colloid_links_array_t;
typedef struct colloid_link_type colloid_link_t;

#include "colloids.h"

struct colloid_links_array_s {
  int max_links;              /* Max number of links for given colloid */
  int active_links;           /* Actual number of links */
  int * i;                    /* Array of outside (fluid) site indices */
  int * j;                    /* Array of inside (solid) site indices */
  int * p;                    /* Array of LB basis vectors for links */
  int * status;               /* Array of link statuses */
  double ** rb;               /* Array of vectors connecting centre of colloid and centre of the boundary link*/
};

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

int colloid_link_max_2d(double a, int nvel);
int colloid_link_max_3d(double a, int nvel);

void create_links_arrays(colloids_info_t * cinfo, colloid_t * pc);

#endif
