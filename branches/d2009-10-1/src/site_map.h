/*****************************************************************************
 *
 *  site_map.h
 *
 *  $Id: site_map.h,v 1.2.16.1 2010-04-05 03:36:26 kevin Exp $
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

void   site_map_init(void);
void   site_map_finish(void);
void   site_map_set_all(char);
void   site_map_halo(void);
double site_map_volume(char);
double site_map_C(const int index);
double site_map_H(const int index);
void   site_map_C_set(const int index, const double c);
void   site_map_H_set(const int index, const double h);
void   site_map_set(const int index, const char status, const double c,
		    const double h);

char site_map_get_status(int, int, int);
char site_map_get_status_index(int);
void site_map_set_status(int, int, int, char);
void site_map_io_status_with_h(void);

enum lattice_type { FLUID, BOUNDARY, SOLID, COLLOID };
extern struct io_info_t * io_info_site_map;
#endif
