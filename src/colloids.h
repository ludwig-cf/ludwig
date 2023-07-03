/*****************************************************************************
 *
 *  colloids.h
 *
 *  Data structures holding linked list of colloids.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOIDS_H
#define LUDWIG_COLLOIDS_H

#include "coords.h"
#include "colloid.h"
#include "colloid_link.h"

/* Auxiliary for diagnostic quantities (for output) */

typedef struct colloid_diagnostic_s colloid_diagnostic_t;

struct colloid_diagnostic_s {

  int index;          /* Copy of particle index for identification. */
  double ftotal[3];   /* Net force on particle (all contributions) */
  double fhydro[3];   /* Hydrodynamic force on particle */
  double Thydro[3];   /* Hydrodynamic torque on particle */
  double fsbulk[3];   /* Bulk stress (divergence) contribution */
  double fsgrad[3];   /* Gradient stress (divergence) contribution */
  double fschem[3];   /* Total "chemical" stress contribution (is fd2 + fd3) */
  double finter[3];   /* External field/interaction contribution */
  double fbuild[3];   /* Force corrections from changes in discrete shape */
  double fnonhy[3];   /* Total non-hydrodynamic (interpendent of fd0-fd1) */

  /* Might want to split external field / conservative interaction
   * contributions. A separate wall contribution for lubrication
   * corrections is missing; it should currently appear in the
   * hydrodynamic contribution. */
};

/* Colloid structure */

typedef struct colloid colloid_t;

struct colloid {

  colloid_state_t s;

  /* AUXILARY */

  double random[6];     /* Random numbers for MC/Brownian dynamics */
  double force[3];      /* Total force on colloid */
  double torque[3];     /* Total torque on colloid */
  double f0[3];         /* Velocity independent force */
  double t0[3];         /* Velocity independent torque */
  double cbar[3];       /* Mean boundary link vector */
  double rxcbar[3];     /* Mean r_b x c_b */
  double deltam;        /* Mass difference owing to change in shape */
  double sumw;          /* Sum of weights over links */
  double zeta[21];      /* Upper triangle of 6x6 drag matrix zeta */
  double stats[3];      /* Particle statisitics */
  double fc0[3];        /* total force on squirmer for mass conservation */
  double tc0[3];        /* total torque on squirmer for mass conservation */
  double sump;          /* flux through squirmer surface */ 
  double dq[2];         /* charge remove/replace mismatch for 2 charges */

  double fsub[3];       /* Subgrid particle force from fluid */
  double fex[3];        /* External forces (non-fluid) on particle */
  double tex[3];        /* External torques on particle */

  /* Diagnostic utilities (all for current time step) */

  colloid_diagnostic_t diagnostic;

  /* Pointers */

  colloid_link_t * lnk; /* Pointer to the list of links defining surface */
  colloid_t * next;     /* colloid is a linked list */

  colloid_t * nextall;  /* List of all local colloids (incl. halo) */
  colloid_t * nextlocal;/* List of local colloids (excl. halo) */

  /* Bonded neighbours cf. colloid.h */

  colloid_t * bonded[NBOND_MAX];
};

typedef struct colloids_info_s colloids_info_t;

struct colloids_info_s {

  int nhalo;                  /* Halo extent in cell list */
  int ntotal;                 /* Total, physical, number of colloids */
  int nallocated;             /* Number colloid_t allocated */
  int ncell[3];               /* Number of cells (excluding  2*halo) */
  int str[3];                 /* Strides for cell list */
  int nsites;                 /* Total number of map sites */
  int ncells;                 /* Total number of cells */

  int nsubgrid;               /* Total number of subgrid particles */
  int rebuild_freq;           /* Rebuild shape every so many steps */

  double rho0;                /* Mean density (usually matches fluid) */
  double drmax;               /* Maximum movement per time step */

  colloid_t ** clist;         /* Cell list pointers */
  colloid_t ** map_old;       /* Map (previous time step) pointers */
  colloid_t ** map_new;       /* Map (current time step) pointers */
  colloid_t * headall;        /* All colloid list (incl. halo) head */
  colloid_t * headlocal;      /* Local list (excl. halo) head */

  pe_t * pe;                  /* Parallel environment */
  cs_t * cs;                  /* Coordinate system */
  colloids_info_t * target;   /* Copy of this structure on target */ 
};

__host__ int colloids_info_create(pe_t * pe, cs_t * cs, int ncell[3],
				  colloids_info_t ** pinfo);
__host__ void colloids_info_free(colloids_info_t * info);
__host__ int colloids_info_recreate(int newcell[3], colloids_info_t ** pinfo);
__host__ int colloids_memcpy(colloids_info_t * info, int flag);
__host__ int colloids_info_nallocated(colloids_info_t * cinfo, int * nallocated);
__host__ int colloids_info_rho0(colloids_info_t * cinfo, double * rho0);
__host__ int colloids_info_rho0_set(colloids_info_t * cinfo, double rho0);
__host__ int colloids_info_map_init(colloids_info_t * info);
__host__ int colloids_info_ncell(colloids_info_t * info, int ncell[3]);
__host__ int colloids_info_nhalo(colloids_info_t * info, int * nhalo);
__host__ int colloids_info_ntotal(colloids_info_t * info, int * ntotal);
__host__ int colloids_info_nlocal(colloids_info_t * cinfo, int * nlocal);
__host__ int colloids_info_ntotal_set(colloids_info_t * cinfo);
__host__ int colloids_info_rebuild_freq(colloids_info_t * cinfo, int * nf);
__host__ int colloids_info_rebuild_freq_set(colloids_info_t * cinfo, int nf);
__host__ int colloids_info_map(colloids_info_t * info, int index, colloid_t ** pc);
__host__ int colloids_info_map_old(colloids_info_t * info, int index, colloid_t ** pc);
__host__ int colloids_info_cell_index(colloids_info_t * cinfo, int ic, int jc, int kc);
__host__ int colloids_info_insert_colloid(colloids_info_t * cinfo, colloid_t * coll);
__host__ int colloids_info_cell_list_clean(colloids_info_t * cinfo);
__host__ int colloids_info_all_head(colloids_info_t * cinfo, colloid_t ** pc);
__host__ int colloids_info_local_head(colloids_info_t * cinfo, colloid_t ** pc);
__host__ int colloids_info_cell_list_head(colloids_info_t * info,
				 int ic, int jc, int kc, colloid_t ** pc);
__host__ int colloids_info_cell_coords(colloids_info_t * cinfo, const double r[3],
			      int icell[3]);
__host__ int colloids_info_add_local(colloids_info_t * cinfo, int index,
			    const double r[3], colloid_t ** pc);
__host__ int colloids_info_add(colloids_info_t * confo, int index, const double r[3],
		      colloid_t ** pc);
__host__ int colloids_info_update_cell_list(colloids_info_t * cinfo);
__host__ int colloids_info_q_local(colloids_info_t * cinfo, double q[2]);
__host__ int colloids_info_v_local(colloids_info_t * cinfo, double * v);
__host__ int colloids_info_lcell(colloids_info_t * cinfo, double lcell[3]);
__host__ int colloids_info_cell_count(colloids_info_t * cinfo, int ic, int jc, int kc,
			     int * ncount);
__host__ int colloids_info_map_update(colloids_info_t * cinfo);
__host__ int colloids_info_position_update(colloids_info_t * cinfo);
__host__ int colloids_info_map_set(colloids_info_t * cinfo, int index,
			      colloid_t * pc);
__host__ int colloids_info_update_lists(colloids_info_t * cinfo);
__host__ int colloids_info_list_all_build(colloids_info_t * cinfo);
__host__ int colloids_info_list_local_build(colloids_info_t * cinfo);
__host__ int colloids_info_climits(colloids_info_t * cinfo, int ia, int ic, int * lim);
__host__ int colloids_info_a0max(colloids_info_t * cinfo, double * a0max);
__host__ int colloids_info_ahmax(colloids_info_t * cinfo, double * ahmax);
__host__ int colloids_info_count_local(colloids_info_t * cinfo, colloid_type_enum_t it,
			      int * count);
__host__ int colloids_number_sites(colloids_info_t *cinfo);
__host__ void colloids_list_sites(int* colloidSiteList, colloids_info_t *cinfo);
__host__ void colloids_q_boundary_normal(colloids_info_t * cinfo,
					       const int index,
					       const int di[3],
					       double dn[3]);
__host__ int colloid_rb(colloids_info_t * info, colloid_t * pc, int index,
			double rb[3]);
__host__ int colloid_rb_ub(colloids_info_t * info, colloid_t * pc, int index,
			   double rb[3], double ub[3]);
__host__ double colloids_largest_dimension(colloid_t * pc);
__host__ int is_site_inside_colloid(colloid_t * pc, double rsep[3]);

#endif
