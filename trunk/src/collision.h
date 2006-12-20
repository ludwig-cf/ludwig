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
void    MODEL_calc_phi( void );
void    get_fluctuations_stress(double shat[3][3]);
void   RAND_init_fluctuations(void);
void    MODEL_collide_multirelaxation(void);
void     MISC_curvature(void);
void     MISC_set_mean_phi(double);
void     latt_zero_force(void);
double    MISC_fluid_volume(void);

#endif
