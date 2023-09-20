/*****************************************************************************
 *
 *  colloid.h
 *
 *  The implementation is exposed for the time being.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Sumesh Thampi added the ellipsoidal particles.
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_H
#define LUDWIG_COLLOID_H

#include <stdio.h>

/* Tag to describe I/O format version appearing in files */

enum colloid_io_version {COLLOID_IO_VERSION = 0210};
typedef enum colloid_io_version colloid_io_version_t;

typedef enum colloid_bc_enum {
  COLLOID_BC_INVALID = 0,
  COLLOID_BC_BBL,
  COLLOID_BC_SUBGRID
} colloid_bc_enum_t;

typedef enum colloid_shape_enum {
  COLLOID_SHAPE_INVALID = 0,
  COLLOID_SHAPE_DISK,
  COLLOID_SHAPE_SPHERE,
  COLLOID_SHAPE_ELLIPSOID
} colloid_shape_enum_t;

/* Additional attributes bitmask */

#define COLLOID_ATTR_JANUS   (1 << 0)

/* These describe the padding etc, and are really for internal
 * unit test consumption. The total number of variables is
 * useful to know to check the ASCII read/write. */

#define NTOT_VAR (32+48)
#define NPAD_INT  7
#define NPAD_DBL  4
#define NBOND_MAX  2


typedef struct colloid_state_type colloid_state_t;

struct colloid_state_type {

  int index;            /* Unique global index for colloid */
  int rebuild;          /* Rebuild flag */
  int nbonds;           /* Number of bonds e.g. fene (to NBOND_MAX) */
  int nangles;          /* Number of angles, e.g., fene (1 at the moment) */

  int isfixedr;         /* Set to 1 for no position update */
  int isfixedv;         /* Set to 1 for no velocity update */
  int isfixedw;         /* Set to 1 for no angular velocity update */
  int isfixeds;         /* Set to zero for no s, m update */

  int type;             /* Particle type NO LONGER USED; see "shape" etc */
  int bond[NBOND_MAX];  /* Bonded neighbours ids (index) */

  int rng;              /* Random number state */

  int isfixedrxyz[3];   /* Position update in specific coordinate directions */
  int isfixedvxyz[3];   /* Velocity update in specific coordinate directions */

  int inter_type;       /* Interaction type of a particle */

  int ioversion;        /* For internal use */
  int bc;               /* Broadly, boundary condition (bbl, subgrid) */
  int shape;            /* Particle shape (2d disk, sphere, ellipsoid) */
  int active;           /* Particle is active */
  int magnetic;         /* Particle is magnetic */
  int attr;             /* Additional attributes bitmask */

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
  double m[3];          /* Current direction of motion vector (squirmer) */
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

  /* parameters describing ellipsoids */

  double elabc[3];	/* Semi principal axes a,b,c */
  double quater[4];	/* Quaternion */
  double quaterold[4];	/* Quaternion at previous time step */

  double dpad[NPAD_DBL];/* Again, this pads to 512 bytes to allow
			 * for future expansion. */
};

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp);
int colloid_state_read_binary(colloid_state_t * ps, FILE * fp);
int colloid_state_write_ascii(const colloid_state_t * ps, FILE * fp);
int colloid_state_write_binary(const colloid_state_t * ps, FILE * fp);
int colloid_state_mass(const colloid_state_t * s, double rho0, double * mass);
int colloid_type_check(colloid_state_t * s);
int colloid_r_inside(const colloid_state_t * s, const double r[3]);

double colloid_principal_radius(const colloid_state_t * s);

#endif
