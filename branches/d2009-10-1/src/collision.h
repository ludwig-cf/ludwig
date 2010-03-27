/*****************************************************************************
 *
 *  collision.h
 *
 *  $Id: collision.h,v 1.4.14.3 2010-03-27 11:22:31 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef COLLISION_H
#define COLLISION_H

void    get_fluctuations_stress(double shat[3][3]);
void   RAND_init_fluctuations(void);
void    collide(void);
void    test_isothermal_fluctuations(void);

#endif
