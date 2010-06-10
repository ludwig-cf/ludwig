/*****************************************************************************
 *
 *  colloids_halo.h
 *
 *  $Id: colloids_halo.h,v 1.1.2.1 2010-06-10 15:05:36 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLOIDS_HALO_H
#define COLLOIDS_HALO_H

void colloids_halo_state(void);
void colloids_halo_dim(int dim);
void colloids_halo_send_count(int dim, int nsend[2]);

#endif
 
