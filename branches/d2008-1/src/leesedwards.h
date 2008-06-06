#ifndef _LEESEDWARDS_H
#define _LEESEDWARDS_H

enum { PHI_ONLY, SITE_AND_PHI };


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

int le_get_nplane(void);
int le_get_nxbuffer(void);
int le_index_real_to_buffer(const int, const int);
int le_index_buffer_to_real(const int);

double    le_buffer_displacement(const int);
MPI_Comm  le_communicator(void);
void      le_displacement_ranks(const double, int[2]);

#endif /* _LEESEDWARDS_H */
