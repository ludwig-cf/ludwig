/*****************************************************************************
 *
 *  This is a wrapper for the LB model.
 *
 *  The choice is currently _D3Q15_ (default) or _D3Q19_. See the files
 *  d3qNVEL.c for the actual definitions.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _MODEL_H
#define _MODEL_H

double  MODEL_get_rho_at_site(const int);
double  MODEL_get_phi_at_site(const int);
FVector MODEL_get_momentum_at_site(const int);
void    MODEL_init( void );
void    MODEL_finish( void );
void    MODEL_get_gradients( void );
void    MODEL_calc_rho( void );
void    MODEL_calc_phi( void );
void    get_fluctuations_stress(double shat[3][3]);
void   RAND_init_fluctuations(void);
void    MODEL_collide_multirelaxation(void);
void     MISC_curvature(void);
void     MISC_set_mean_phi(double);
double    MISC_fluid_volume(void);
double    get_eta_shear(void);
double    get_kT(void);
double    get_rho0(void);
double    get_phi0(void);

#endif /* _MODEL_H_ */
