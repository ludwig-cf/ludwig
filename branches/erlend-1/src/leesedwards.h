#ifndef _LEESEDWARDS_H
#define _LEESEDWARDS_H

enum { PHI_ONLY, SITE_AND_PHI };

enum { INTERP_SITE_F, INTERP_SITE_G, INTERP_PHI };


/* struct for Lees-Edwards */
typedef struct{
  int rank;       /* Rank in global system */
  int peX;        /* Value of pe[X] of local PEs */
  int loc;        /* Location of Lees-Edwards plane */
  double vel;     /* Velocity of Lees-Edwards plane */
  double disp;    /* Displacement (in lattice site) at current timestep */
  double frac;    /* Weight for linear interpolation = 1-(disp-floor(disp)) */
} LE_Plane;


void LE_init( void );
void LE_apply_LEBC( void );
void LE_print_params( void );
void LE_update_buffers( int );

#endif /* _LEESEDWARDS_H */
