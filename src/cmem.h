/*****************************************************************************
 *
 *  cmem.h
 *
 *****************************************************************************/

#ifndef _CMEM_H_
#define _CMEM_H_

Colloid   * CMEM_allocate_colloid(void);
COLL_Link * CMEM_allocate_boundary_link(void);
void        CMEM_free_colloid(Colloid *);
void        CMEM_report_memory(void);

#endif
