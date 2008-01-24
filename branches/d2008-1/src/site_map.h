/*****************************************************************************
 *
 *  site_map.h
 *
 *  $Id: site_map.h,v 1.1.2.2 2008-01-24 18:25:33 kevin Exp $
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

void site_map_init(void);
void site_map_finish(void);
void site_map_set_all(char);
void site_map_halo(void);
double site_map_volume(char);


char site_map_get_status(int, int, int);
void site_map_set_status(int, int, int, char);

enum lattice_type { FLUID, SOLID, COLLOID, BOUNDARY };
extern struct io_info_t * io_info_site_map;
#endif
