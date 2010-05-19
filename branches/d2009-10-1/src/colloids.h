/*****************************************************************************
 *
 *  colloids.h
 *
 *  Data structures and global list pointers for colloids.
 *
 *  $Id: colloids.h,v 1.9.2.8 2010-05-19 19:16:50 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_H
#define COLLOIDS_H

/* Colloid structure
 * contains state information for individual colloid.
 * This involves both physical properties (position, velocity, etc)
 * and structural information (linked lists, parallel information). */

typedef struct colloid_state_type colloid_state_t;
typedef struct colloid Colloid;
typedef struct coll_link COLL_Link;

struct colloid_state_type {

  int    index;         /* Unique global index for colloid */
  int    rebuild;       /* Rebuild flag */
  double a0;            /* Input radius (lattice units) */
  double ah;            /* Hydrodynamic radius (from calibration) */
  double r0[3];         /* Initial position */
  double r[3];          /* Position */
  double v[3];          /* Velocity */
  double w[3];          /* Angular velocity */
  double s[3];          /* Magnetic dipole, or spin */
  double direction[3];  /* Currect direction of motion vector (squirmer) */
  double b1;	        /* squirmer active parameter b1 */
  double b2;            /* squirmer active parameter b2 */
  double c;             /* Wetting free energy parameter C */
  double h;             /* Wetting free energy parameter H */

};

struct colloid {

  /* STATE */

  int    index;         /* Unique global index for colloid */
  int    rebuild;       /* Rebuild flag */
  double a0;            /* Input radius (lattice units) */
  double ah;            /* Hydrodynamic radius (from calibration) */
  double r[3];
  double v[3];
  double omega[3];      /* Angular velocity */
  double dr[3];         /* position update */

  /* Magnetic dipole */

  double s[3];   

  /* Wetting free energy parameters E.g., for binary fluid. */

  double c_wetting;
  double h_wetting;

  /* Active particle stuff */

  double direction[3];   /* Currect direction of motion vector */
  double b1;	         /* squirmer active parameter B_1 */
  double b2;             /* squirmer active parameter B_2 */


  /* AUXILARY */

  double random[6];     /* Random numbers for MC/Brownian dynamics */
  double force[3];      /* Total force on colloid */
  double torque[3];     /* Total torque on colloid */
  double f0[3];         /* Velocity independent force */
  double t0[3];         /* Velocity independent torque */
  double cbar[3];       /* Mean boundary link vector */
  double rxcbar[3];     /* Mean r_b x c_b */
  double deltam;        /* Mass difference owing to change in shape */
  double deltaphi;      /* Order parameter deficit on bounce-back */
  double sumw;          /* Sum of weights over links */
  double zeta[21];      /* Upper triangle of 6x6 drag matrix zeta */
  double stats[3];      /* Particle statisitics */
  double fc0[3];        /* total force on squirmer for mass conservation */
  double tc0[3];        /* total torque on squirmer for mass conservation */
  double sump;          /* flux through squirmer surface */ 

  /* Pointers */

  COLL_Link * lnk;      /* Pointer to the list of links defining surface */
  Colloid   * next;     /* colloid is a linked list */

};

/* Colloid boundary link structure
 * A linked list for the boundary links for a given colloid. */

enum link_status {LINK_FLUID, LINK_COLLOID, LINK_BOUNDARY, LINK_UNUSED}; 

struct coll_link {

  int       i;             /* Index of lattice site outside colloid */ 
  int       j;             /* Index of lattice site inside */
  int       v;             /* Index of velocity connecting i -> j */
  int       status;        /* What is at site i (fluid, solid, etc) */
  double    rb[3];         /* Vector connecting centre of colloid and
			    * centre of the boundary link */

  COLL_Link * next;        /* COLL_Link is a linked list */
};

void      colloids_init(void);
void      colloids_finish(void);
void      colloid_cell_coords(const double r[3], int icell[3]);
Colloid * CELL_get_head_of_list(const int, const int, const int);
void      cell_insert_colloid(Colloid *);
int       Ncell(const int);
double    Lcell(const int);
Colloid * allocate_colloid(void);
void      free_colloid(Colloid *);
COLL_Link * allocate_boundary_link(void);
void      cell_update(void);
void      set_N_colloid(const int);
double    colloid_rho0(void);
Colloid * colloid_add_local(const int index, const double r[3]);
Colloid * colloid_add(const int index, const double r[3]);
int       colloid_ntotal(void);
int       colloid_nlocal(void);

#endif
