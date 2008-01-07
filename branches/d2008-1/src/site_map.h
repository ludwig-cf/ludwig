/*****************************************************************************
 *
 *  site_map.h
 *
 *  $Id: site_map.h,v 1.1.2.1 2008-01-07 17:32:29 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef SITEMAP_H
#define SITEMAP_H

void init_site_map(void);
void finish_site_map(void);
void site_map_set_all(char);
double site_map_volume(char);

char site_map_get_status(int, int, int);
void site_map_set_status(int, int, int, char);

enum lattice_type { FLUID, SOLID, COLLOID, BOUNDARY };

#endif
