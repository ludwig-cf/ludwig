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
int le_site_index(const int, const int, const int);
int le_plane_location(const int);
int le_get_nplane_total(void);

double    le_buffer_displacement(const int);
double    le_get_block_uy(int);
double    le_get_plane_uy();
MPI_Comm  le_communicator(void);
void      le_displacement_ranks(const double, int[2], int[2]);
void      le_init_shear_profile(void);


/* Address macro. For performance purposes, -DNDEBUG replaces
 * calls to ADDR, ie., le_site_index() with a macro, which requires that
 * the local system size be available as "nlocal". nhalo_ is
 * const, and available from coords.h, as are {X,Y,Z}  */

#ifdef NDEBUG
#define ADDR(ic,jc,kc) \
((nlocal[Y]+2*nhalo_)*(nlocal[Z]+2*nhalo_)*(nhalo_+(ic)-1) + \
                      (nlocal[Z]+2*nhalo_)*(nhalo_+(jc)-1) + \
                                           (nhalo_+(kc)-1))
#else
#define ADDR le_site_index
#endif

#endif /* _LEESEDWARDS_H */
