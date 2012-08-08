/*****************************************************************************
 *
 *  build.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _BUILD_H
#define _BUILD_H

void COLL_update_links(void);
void COLL_update_map(void);
void COLL_init_coordinates(void);

#ifdef OLD_PHI
void COLL_remove_or_replace_fluid(void);
#else
#include "field.h"
int build_remove_or_replace_fluid(field_t * fphi, field_t * fp, field_t * fq);
#endif

#endif
