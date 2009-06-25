
/*****************************************************************************
 *
 *  colloids.h
 *
 *  Data structures and global list pointers for colloids.
 *
 *  See Ludwig Technical Notes for a complete description
 *  of the colloid implementation.
 *
 *  $Id: colloids.h,v 1.7.14.1 2009-06-25 14:49:13 ricardmn Exp $
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _COLLOIDS_H
#define _COLLOIDS_H

/* Colloid structure
 * contains state information for individual colloid.
 * This involves both physical properties (position, velocity, etc)
 * and structural information (linked lists, parallel information). */

#include "utilities.h"

typedef struct colloid Colloid;
typedef struct coll_link COLL_Link;

struct colloid {

  int       index;         /* Unique global index for colloid */
  int       rebuild;       /* Rebuild flag */
  double    a0;            /* Input radius (lattice units) */
  double    ah;            /* Hydrodynamic radius (from calibration) */
  FVector   r;             /* Position vector of centre of mass */
  double    dr[3];         /* position update */
  FVector   v;             /* Linear velocity */
  FVector   omega;         /* Angular velocity */
  double    random[6];     /* Random numbers for MC/Brownian dynamics */
  FVector   force;         /* Total force on colloid */
  FVector   torque;        /* Total torque on colloid */
  FVector   f0;            /* Velocity independent force */
  FVector   t0;            /* Velocity independent torque */
  FVector   cbar;          /* Mean boundary link vector */
  FVector   rxcbar;        /* Mean r_b x c_b */
  double    deltam;        /* Mass difference owing to change in shape */
  double    deltaphi;      /* Order parameter deficit on bounce-back */
  double    sumw;          /* Sum of weights over links */
  double    zeta[21];      /* Upper triangle of 6x6 drag matrix zeta */
  FVector   stats;         /* Particle statisitics */

  /* Magnetic dipole */

  double   s[3];   

  /* Active particle stuff */

  FVector  dir;            /* Currect direction of motion vector */
  FVector  tot_va;         /* Active particle 2 link sum */
  double   dp;             /* Momentum exchange parameter */
  double   cosine_ca;      /* Cosine of the cone angle */
  int      n1_nodes;       /* Number of fluid nodes cone 1 ('front'?) */
  int      n2_nodes;       /* Number of fluid nodes cone 2 ('back'?) */

  FVector   fc0;           /* total force on squirmer for mass conservation */          
  FVector   tc0;           /* total torque on squirmer for mass conservation */
  double sump;             /* flux through squirmer surface*/ 
  double b1;		   /* squirmer active parameter B_1 */
  double b2;               /* squirmer active parameter B_2 */


  /* Pointers */

  COLL_Link * lnk;         /* Pointer to the list of links defining surface */
  Colloid   * next;        /* colloid is a linked list */

};

/* Colloid boundary link structure
 * A linked list for the boundary links for a given colloid. */

enum link_status {LINK_FLUID, LINK_COLLOID, LINK_BOUNDARY, LINK_UNUSED}; 

struct coll_link {

  int       i;             /* Index of lattice site outside colloid */ 
  int       j;             /* Index of lattice site inside */
  int       v;             /* Index of velocity connecting i -> j */
  int       status;        /* What is at site i (fluid, solid, etc) */

  FVector   rb;            /* Vector connecting centre of colloid and
			    * centre of the boundary link */

  COLL_Link * next;        /* COLL_Link is a linked list */
};

void      colloids_init(void);
void      colloids_finish(void);
IVector   cell_coords(FVector);
Colloid * CELL_get_head_of_list(const int, const int, const int);
void      cell_insert_colloid(Colloid *);
int       Ncell(const int);
double    Lcell(const int);
Colloid * allocate_colloid(void);
void      free_colloid(Colloid *);
COLL_Link * allocate_boundary_link(void);
void      cell_update(void);
void      colloids_memory_report(void);
void      set_N_colloid(const int);
int       get_N_colloid(void);
double    get_colloid_rho0(void);

#endif /* _COLLOIDS_H */
