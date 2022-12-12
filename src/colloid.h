/*****************************************************************************
 *
 *  colloid.h
 *
 *  The implementation is exposed for the time being.
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

#ifndef LUDWIG_COLLOID_H
#define LUDWIG_COLLOID_H

#include <stdio.h>

/* Tag to describe I/O format version appearing in files */

enum colloid_io_version {COLLOID_IO_VERSION = 0200};
typedef enum colloid_io_version colloid_io_version_t;

/* These describe the padding etc, and are really for internal
 * unit test consumption. The total number of variables is
 * useful to know to check the ASCII read/write. */


/* NPAD_INT 1 = 12 - iscentre - indexcentre - ishole - NBONDMAX2(3) -nbonds2 - NBONDMAX3 (3) - nbonds3*/

/* NPAD_DOUBLE - 6 = 15 - u0 - delta - cutoff - n[3] - fphi[3] - fsprings[3] - fsub[3] - tphi[3] - tsprings[3] - total_force[3] */

#define NTOT_VAR (12 + 3 + 3 + 240 + 642 + 11 + 0              + 60            + 0           + 14)
//                ints nbonds 1 2 3 mesh ints pad	     double	   double pad       tuple


// PADDING MUST BE INCLUDED IN NTOT_VAR !
#define NPAD_INT  0
#define NPAD_DBL  0

#define NBOND_MAX  3
#define NBOND_MAX2  3
#define NBOND_MAX3  240
#define NBOND_MAX_MESH 642 

enum colloid_type_enum {COLLOID_TYPE_DEFAULT = 0,
			COLLOID_TYPE_ACTIVE,
			COLLOID_TYPE_SUBGRID,
			COLLOID_TYPE_JANUS};

typedef enum colloid_type_enum colloid_type_enum_t;
typedef struct colloid_state_type colloid_state_t;

typedef struct {
  int indices[7];
  double r0s[7];
} tTuple;

struct colloid_state_type {

  int index;            /* Unique global index for colloid */
  int rebuild;          /* Rebuild flag */
  int nbonds;           /* Number of bonds e.g. fene (to NBOND_MAX) */
  int nbonds2;           /* Number of bonds e.g. fene (to NBOND_MAX2) */
  int nbonds3;           /* Number of bonds e.g. fene (to NBOND_MAX3) */
  int nbonds_mesh;           /* Number of bonds e.g. fene (to NBOND_MAX) */
  int nangles;          /* Number of angles, e.g., fene (1 at the moment) */

  int isfixedr;         /* Set to 1 for no position update */
  int isfixedv;         /* Set to 1 for no velocity update */
  int isfixedw;         /* Set to 1 for no angular velocity update */
  int isfixeds;         /* Set to zero for no s, m update */

  int type;             /* Particle type */
  int bond[NBOND_MAX];  /* Bonded neighbours ids (index) */
  int bond2[NBOND_MAX2];  /* Bonded neighbours ids (index) */
  int bond3[NBOND_MAX3];  /* Bonded neighbours ids (index) */
  int bond_mesh[NBOND_MAX_MESH];  /* Bonded neighbours ids (index) */

  int rng;              /* Random number state */

  int isfixedrxyz[3];   /* Position update in specific coordinate directions */
  int isfixedvxyz[3];   /* Velocity update in specific coordinate directions */

  int inter_type;         /* Interaction type of a particle */
/* -----> CHEMOVESICLE V2 */
  int iscentre;
  int indexcentre;
  int ishole;
/* <----- */

  /* New integer additions can be immediately before the padding */
  /* This should allow existing binary files to be read correctly */

  int intpad[NPAD_INT]; /* I'm going to pad to 32 ints to allow for future
			 * expansion. Additions should be appended here,
			 * and the padding reduced appropriately. */

  double a0;            /* Input radius (lattice units) */
  double ah;            /* Hydrodynamic radius (from calibration) */
  double r[3];          /* Position */
  double v[3];          /* Velocity */
  double w[3];          /* Angular velocity omega */
  double s[3];          /* Magnetic dipole, or spin */
  double m[3];          /* Vesicle orientation */
  double n[3];          /* Vesicle secondary orientation */
  double b1;	        /* squirmer active parameter b1 */
  double b2;            /* squirmer active parameter b2 */
  double c;             /* Wetting free energy parameter C */
  double h;             /* Wetting free energy parameter H */
  double dr[3];         /* r update (pending refactor of move/build process) */
  double deltaphi;      /* order parameter bbl net; required to restart */

  /* Charges. We allow two charge valencies (cf a general number
   * number in the electrokinetics section). q0 will be associated
   * with psi->rho[0] and q1 to psi->rho[1] in the electrokinetics.
   * The charge will
   * be converted to a density by dividing by the current discrete
   * volume to ensure conservation. */

  double q0;            /* magnitude charge 0 */
  double q1;            /* magnitude charge 1 */
  double epsilon;       /* permittivity */

  double deltaq0;       /* surplus/deficit of charge 0 at change of shape */
  double deltaq1;       /* surplus/deficit of charge 1 at change of shape */
  double sa;            /* surface area (finite difference) */
  double saf;           /* surface area to fluid (finite difference grid) */

  double al;            /* Offset parameter used for subgrid particles */

/* -----> CHEMOVESICLE V2+V3 */
  double u0;
  double delta;
  double cutoff;
  double fsub[3]; /* Used to pass force quantities to the state so that they can be easily extracted with extract_colloids.c */
  double fphi[3]; /* idem */
  double fsprings[3]; /* idem */
  double tphi[3];
  double tsprings[3];
  double total_force[3]; /* Total force acted on vesicle */
  double total_torque[3]; /* Total torque acted on vesicle */
/* <----- */

  double dpad[NPAD_DBL];/* Again, this pads to 512 bytes to allow
			 * for future expansion. */
  tTuple tuple;
};

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp);
int colloid_state_read_binary(colloid_state_t * ps, FILE * fp);
int colloid_state_write_ascii(const colloid_state_t * ps, FILE * fp);
int colloid_state_write_binary(const colloid_state_t * ps, FILE * fp);

#endif
