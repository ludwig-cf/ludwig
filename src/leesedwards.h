/*****************************************************************************
 *
 *  leesedwards.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2014 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LEESEDWARDS_H
#define LEESEDWARDS_H

HOST void le_init(void);
HOST void le_finish(void);

HOST int le_info(void);
HOST int le_get_nxbuffer(void);
HOST int le_index_real_to_buffer(const int, const int);
HOST int le_index_buffer_to_real(const int);
HOST int le_site_index(const int, const int, const int);
HOST int le_plane_location(const int);
HOST int le_get_nplane_total(void);
HOST int le_get_nplane_local(void);
HOST int le_nsites(void);

HOST double    le_buffer_displacement(const int, const double);
HOST double    le_get_block_uy(int);
HOST double    le_get_steady_uy(const int); 
HOST double    le_plane_uy(const double);
HOST double    le_plane_uy_max(void);
HOST double    le_shear_rate(void);
HOST MPI_Comm  le_communicator(void);
HOST MPI_Comm  le_plane_comm(void);
HOST void      le_jstart_to_ranks(const int, int send[2], int recv[2]);
HOST void      le_set_oscillatory(const double);
HOST void      le_set_nplane_total(const int nplane);
HOST void      le_set_plane_uymax(const double uy);

#endif
