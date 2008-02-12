/*****************************************************************************
 *
 *  collision.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _COLLISION_H
#define _COLLISION_H

void    MODEL_init( void );
void    MODEL_finish( void );
void    MODEL_get_gradients( void );
void    get_fluctuations_stress(double shat[3][3]);
void   RAND_init_fluctuations(void);
void    collide(void);
void     MISC_curvature(void);
void     latt_zero_force(void);

#endif
