/*****************************************************************************
 *
 *  cmem.c
 *
 *  Colloid memory management.
 *
 *  A running total of the number of colloids, and links, allocated
 *  is kept for information, and to check for memory leaks.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include "pe.h"

#include "utilities.h"
#include "colloids.h"
#include "cmem.h"

static int nalloc_colls = 0;   /* Number of colloids currently alloacted */
static int nalloc_links = 0;   /* Number of links currently allocated */


/*****************************************************************************
 *
 *  CMEM_allocate_colloid
 *
 *  Allocate space for a colloid structure and return a pointer to
 *  it (or fail gracefully).
 *
 *****************************************************************************/

Colloid * CMEM_allocate_colloid() {

  Colloid * p_colloid;

  p_colloid = (Colloid *) malloc(sizeof(Colloid));
  if (p_colloid == (Colloid *) NULL) fatal("malloc(Colloid) failed\n");

  nalloc_colls++;

  return p_colloid;
}


/*****************************************************************************
 *
 *  CMEM_allocate_boundary_link
 *
 *  Return a pointer to a newly allocated COLL_Link structure
 *  or fail gracefully.
 *
 *****************************************************************************/

COLL_Link * CMEM_allocate_boundary_link() {

  COLL_Link * p_link;

  p_link = (COLL_Link *) malloc(sizeof(COLL_Link));
  if (p_link == (COLL_Link *) NULL) fatal("malloc(Coll_link) failed\n");

  nalloc_links++;

  return p_link;
}


/*****************************************************************************
 *
 *  CMEM_free_colloid
 *
 *  Destroy an unwanted colloid. Note that this is the only way
 *  that boundary links should be freed, so that the current
 *  link count is maintained correctly. 
 *
 *****************************************************************************/

void CMEM_free_colloid(Colloid * p_colloid) {

  COLL_Link * p_link;
  COLL_Link * tmp;

  if (p_colloid == NULL) {
    /* Trying to free a NULL pointer is fatal. */
    fatal("Trying to free NULL colloid\n");
  }
  else {

    /* Free any links, then the colloid itself */

    p_link = p_colloid->lnk;
    
    while (p_link != NULL) {
      tmp = p_link->next;
      free(p_link);
      p_link = tmp;
      nalloc_links--;
    }

    free(p_colloid);

    nalloc_colls--;
  }

  return;
}

/*****************************************************************************
 *
 *  CMEM_report_memory
 *
 *  Show the current total for colloids and links.
 *
 *****************************************************************************/

void CMEM_report_memory() {

  info("\nCMEM_report_memory\n");
  info("[colloids: %d (%f Mb) links: %d (%f Mb)]\n",
       nalloc_colls, nalloc_colls*sizeof(Colloid)/1.e6,
       nalloc_links, nalloc_links*sizeof(COLL_Link)/1.e6);

  /* If verbose may want all processes */

  return;
}
