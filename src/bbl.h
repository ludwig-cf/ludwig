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
int bbl_update_ellipsoids(bbl_t * bbl, wall_t * wall, colloids_info_t * cinfo);
void setter_ladd_Gausselmn_LHS(const double half, const double *zeta, const double *dwall, const double mass, const double mI[3][3],  double a[6][6]);
void setter_ladd_Gausselmn_RHS(const double half, const double mass, const double mI[3][3], const double *v, const double *w, const double *f0, const double *t0, const double *force, const double *torque, const double *fc0, const double *tc0, double xb[6]);
void setter_ladd_LHS_ellipsoid(const double *quaterold, const double *moment, const double mI[3][3],  double a[6][6]);
void solver_ladd_Gausselmn(bbl_t * bbl, double a[6][6], double xb[6]);
__host__ void record_force_torque(colloid_t * pc);

int bbl_surface_stress(bbl_t * bbl, double slocal[3][3]);
int bbl_order_parameter_deficit(bbl_t * bbl, double * delta);

#endif
