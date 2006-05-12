
#ifndef _COLLOIDS_H
#define _COLLOIDS_H

/*****************************************************************************
 *
 *  colloids.h
 *
 *  Data structures and global list pointers for colloids.
 *
 *  See Ludwig Technical Notes for a complete description
 *  of the colloid implementation.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

/* Global_Colloid
 * contains fixed parameters for colloid physics */

struct {

  int       N_colloid;     /* Total number of colloids in model */
  int       nlocal;        /* Local number of colloids */
  int       fr;            /* Rebuild frequency */
  Float     a0;            /* Default input radius */
  Float     ah;            /* Hydrodynamic radius for a0 */
  Float     vf;            /* volume fraction (input) */
  Float     rho;           /* Colloid density */
  Float     deltaf;        /* Current mass deficit */
  Float     deltag;        /* Current order parameter deficit */
  Float     r_lu_n;        /* Cutoff for normal lubrication */
  Float     r_lu_t;        /* Cutoff for tangential lubrication */
  Float     r_lu_r;        /* Cutoff for rotational lubrication */
  Float     r_ssph;        /* Cutoff distance for soft-sphere potential */
  Float     r_clus;        /* Cutoff distance for cluster-implicit */
  IVector   Ncell;         /* Number of cells in each dimension (local) */
  FVector   Lcell;         /* Cell width in each dimension */
  FVector   F;             /* Force on all colloids, e.g., gravity */
  int       pid;           /* Identifier for drop-in potential */
  Float     drop_in_p1;    /* Parameter for "drop-in" potential */
  Float     drop_in_p2;    /* Parameter for "drop-in" potential */
  Float     drop_in_p3;    /* Parameter for "drop-in" potential */

} Global_Colloid;


/* Colloid structure
 * contains state information for individual colloid.
 * This involves both physical properties (position, velocity, etc)
 * and structural information (linked lists, parallel information). */

typedef struct colloid Colloid;
typedef struct coll_link COLL_Link;

struct colloid {

  int       index;         /* Unique global index for colloid */
  int       rebuild;       /* Rebuild flag */
  Float     a0;            /* Input radius (lattice units) */
  Float     ah;            /* Hydrodynamic radius (from calibration) */
  FVector   r;             /* Position vector of centre of mass */
  FVector   v;             /* Linear velocity */
  FVector   omega;         /* Angular velocity */
  FVector   force;         /* Total force on colloid */
  FVector   torque;        /* Total torque on colloid */
  FVector   f0;            /* Velocity independent force */
  FVector   t0;            /* Velocity independent torque */
  FVector   cbar;          /* Mean boundary link vector */
  FVector   rxcbar;        /* Mean r_b x c_b */
  Float     deltam;        /* Mass difference owing to change in shape */
  Float     deltaphi;      /* Order parameter deficit on bounce-back */
  Float     sumw;          /* Sum of weights over links */
  Float     zeta[21];      /* Upper triangle of 6x6 drag matrix zeta */
  FVector   stats;         /* Particle statisitics */

  /* Active particle stuff */

  FVector  dir;            /* Currect direction of motion vector */
  Float    dp;             /* Momentum exchange parameter */
  Float    angle;          /* Cone angle */
  int      n1_nodes;       /* Number of fluid nodes cone 1 ('front'?) */
  int      n2_nodes;       /* Number of fluid nodes cone 2 ('back'?) */

  /* Pointers */

  COLL_Link * lnk;         /* Pointer to the list of links defining surface */
  int       export;        /* Communication flag */
  Colloid   * next;        /* colloid is a linked list */

};

/* Colloid boundary link structure
 * A linked list for the boundary links for a given colloid. */


struct coll_link {

  int       i;             /* Index of lattice site inside/outside colloid */ 
  int       j;             /* Index of lattice site outside/inside */
  int       v;             /* Index of velocity connecting i -> j */
  int       solid;         /* colloid -> colloid (or solid?) link */

  FVector   rb;            /* Vector connecting centre of colloid and
			    * centre of the boundary link */

  COLL_Link * next;        /* COLL_Link is a linked list */
};

Colloid * COLL_add_colloid(int, Float, Float, FVector, FVector, FVector);
void      COLL_add_colloid_no_halo(int, Float, Float, FVector, FVector,
				   FVector);
FVector   COLL_fvector_separation(FVector, FVector);
FVector   COLL_fcoords_from_ijk(int, int, int);
void      COLL_set_colloid_gravity(void);
void      COLL_zero_forces(void);
void      COLL_init(void);
void      COLL_finish(void);

void      COLL_bounce_back(void);
Float     COLL_interactions(void);
void      COLL_update(void);
void      COLL_forces(void);

#endif /* _COLLOIDS_H */
