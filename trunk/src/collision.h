/*****************************************************************************
 *
 *  collision.h
 *
 *  $Id: collision.h,v 1.4 2008-08-24 17:34:26 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2007 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef _COLLISION_H
#define _COLLISION_H

void    MODEL_init( void );
void    MODEL_finish( void );
void    get_fluctuations_stress(double shat[3][3]);
void   RAND_init_fluctuations(void);
void    collide(void);
void     MISC_curvature(void);

#endif
