#ifndef _COM_H
#define _COM_H

#ifdef _MPI_                     /* Parallel implementation using MPI */

#include<mpi.h>                  /* Generic MPI include file */
#include<limits.h>

/* MPI tags */
enum{                            
  TAG_HALO_SWP_X_BWD = 100,
  TAG_HALO_SWP_X_FWD,
  TAG_HALO_SWP_Y_BWD,
  TAG_HALO_SWP_Y_FWD,
  TAG_HALO_SWP_Z_BWD,
  TAG_HALO_SWP_Z_FWD,
  TAG_IO
};
#endif /* _MPI_ */


void    COM_init( int, char ** );
void    COM_finish( void );
void    COM_halo( void );
void    COM_halo_phi( void );
IVector COM_index2coord( int index );
int     COM_local_index( int );
void    COM_read_site( char *, void (*func)( FILE * ));
void    COM_write_site( char *, void (*func)( FILE *, int, int ));

#endif /* _COM_H */








