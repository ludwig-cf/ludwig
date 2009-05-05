/*****************************************************************************
 *
 *  site_map.h
 *
 *  $Id: site_map.h,v 1.2 2008-08-24 16:58:10 kevin Exp $
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
double site_map_get_C(int);
double site_map_get_H(int);

char site_map_get_status(int, int, int);
char site_map_get_status_index(int);
void site_map_set_status(int, int, int, char);
void site_map_io_status_with_h(void);

enum lattice_type { FLUID, BOUNDARY, SOLID, COLLOID };
extern struct io_info_t * io_info_site_map;
#endif
