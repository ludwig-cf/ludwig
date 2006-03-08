/*****************************************************************************
 *
 *  wall.h
 *
 *  Data structures for solid boundary walls.
 *
 *  These structures store information on simple planar walls
 *  which are often used as boundaries at the side of the domain.
 *
 *  The wall structure is also used as a way to inform colloidal
 *  particles about the presence of the wall for the purposes of
 *  lubrication force calculation.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

enum wall_type {WALL_XY, WALL_XZ, WALL_YZ};


/* Wall structure.
 * This type of wall is a single object which represents
 * a solid wall of thickness two lattice sites providing
 * a way to split the system in one dimension. The wall
 * occupies the first and last site in the direction
 * specified by orientation.
 *
 * Note:
 * The wall is assumed to be in the XY plane at the
 * moment (i.e. solid sites at z = 1 and z = N_total.z).
 *
 * This type of wall is typically used to apply (time dependent)
 * sheer. The maximum velocity of the of wall (usually parallel
 * to the plane of the wall) is umax. One side of the wall is
 * allowed (magically) to move in the opposite direction to the
 * other via the value of sheer_phase (e.g. +1 for two sides of
 * the system moving in same direction or -1 for opposite direction).
 *
 * Lubrication at the wall is determined by the displacement
 * delta, which must be determined by calibration for given
 * viscosity. The apparent position of the wall from the boundary
 * nodes is displaced by a distance delta into the fluid.
 *
 * These walls are assumed to have uniform wetting properties.
 *
 * This could be a list, but maximum of 1 wall at the moment. */


typedef struct wall_struct      Wall;
typedef struct wall_link_struct Wall_link;

struct wall_struct{

  int         present;         /* Switch on at run time */
  int         orientation;     /* In the XY, YZ, or XZ plane */
  Float       rlower;          /* Position of solid sites at one end */
  Float       rupper;          /* Position of nsolid sites at other end */
  Float       r_lu_n;          /* Displacement of lubricating surface */
  Float       sheer_uxmax;     /* Maximum sheer velocity (lattice units) */
  Float       sheer_uymax;     /* ditto */
  Float       sheer_diff;      /* Velocity of one side relative to other */
  Float       sheer_period;    /* Period for sin(2\pi t/T) varying sheer */
  FVector     sheer_u;         /* Current sheer velocity */ 

  Float       C;               /* Wetting property */
  Float       H;               /* Wetting property */

  Float       duct_delta;      /* Duct boundary layer width */
  Float       duct_force;      /* Body force associated with duct */

  Wall_link * lnklower;        /* Head of list of links on the 'upper'... */
  Wall_link * lnkupper;        /* ... side and on the lower side */
};


/* Wall link
 * List of boundary links associated with the wall.
 * This link follows the convention that the link starts in the
 * fluid and ends in the solid. */

struct wall_link_struct {

  int         i;               /* Outside (fluid) lattice node index */
  int         j;               /* Inside (solid) lattice node index */
  int         p;               /* Basis vector for this link */

  Float       phi_b;           /* Order parameter at boundary node */
                               /* (experimental ... use site value) */

  Wall_link * next;            /* This is a linked list */
};


extern Wall    _wall;          /* Global wall structure */

FVector     WALL_lubrication(Colloid *);
void        WALL_init(void);
void        WALL_bounce_back(void);
void        WALL_update(int step);

