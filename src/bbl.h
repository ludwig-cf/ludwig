/*****************************************************************************
 *
 *  bbl.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2016 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_BBL_H
#define LUDWIG_BBL_H

#include "coords.h"
#include "colloids.h"
#include "lb_data.h"
#include "wall.h"

typedef struct bbl_s bbl_t;

int bbl_create(pe_t * pe, cs_t * cs, lb_t * lb, bbl_t ** pobj);
int bbl_free(bbl_t * obj);

int bounce_back_on_links(bbl_t * bbl, lb_t * lb, wall_t * wall,
			 colloids_info_t * cinfo);
int bbl_pass0(bbl_t * bbl, lb_t * lb, colloids_info_t * cinfo);

int bbl_active_set(bbl_t * bbl, colloids_info_t * cinfo);
int bbl_update_colloids(bbl_t * bbl, wall_t * wall, colloids_info_t * cinfo);
int bbl_update_colloid_default(bbl_t * bbl, wall_t * wall, colloid_t * pc, double rho0, double xb[6]);
int bbl_update_ellipsoid(bbl_t * bbl, wall_t * wall, colloid_t * pc, double rho0, double xb[6]);

void setter_ladd_ellipsoid(colloid_t *pc, wall_t * wall,double rho0,  double a[6][6], double xb[3]);
int solver_Gausselimination(double a[6][6], double xb[6]);
__host__ void record_force_torque(colloid_t * pc);

int bbl_surface_stress(bbl_t * bbl, double slocal[3][3]);
int bbl_order_parameter_deficit(bbl_t * bbl, double * delta);

#endif
