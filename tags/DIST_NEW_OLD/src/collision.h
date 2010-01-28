/*****************************************************************************
 *
 *  collision.h
 *
 *  $Id: collision.h,v 1.4.14.2 2010-01-15 16:59:15 kevin Exp $
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
void    get_fluctuations_stress(double shat[3][3]);
void   RAND_init_fluctuations(void);
void    collide(void);
void    test_isothermal_fluctuations(void);

#endif
