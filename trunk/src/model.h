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

#ifdef _D3Q19_
  #include "d3q19.h"
#else
  /* This is the default. */
  #include "d3q15.h"
#endif


/*------------ Definition of Global structure -----------------------------*/ 

typedef struct {         /* Global variables live here */

  int     input_format,  /* Binary or ASCII i/o        */
          output_format; /* Binary or ASCII i/o        */

  double  rho,           /* Average simulation density */
          phi,           /* Average order parameter    */
          mobility,      /* Mobility: unnormalised     */
          noise;         /* Initial noise amplitude    */

  char    input_config[256],
          output_config[256];

} Global;

extern Global     gbl;        /* Most global variables live here */
extern char       *site_map;  /* Map of full and empty sites */

/*--------------------- Definition of MODEL functions --------------------*/

double  MODEL_get_rho_at_site(const int);
double  MODEL_get_phi_at_site(const int);
FVector MODEL_get_momentum_at_site(const int);
void    MODEL_init( void );
void    MODEL_finish( void );
void    MODEL_process_options( Input_Data * );
void    MODEL_get_gradients( void );
void    MODEL_set_distributions_atrest( FVector, FTensor, Site * );
void    MODEL_calc_rho( void );
void    MODEL_calc_phi( void );
void    get_fluctuations_stress(double shat[3][3]);
void   RAND_init_fluctuations(void);
void    MODEL_limited_propagate(void);
void     MISC_set_mean_phi(double);
double    MISC_fluid_volume(void);
double    get_eta_shear(void);

extern void (*MODEL_write_site)( FILE *, int, int );
extern void (*MODEL_write_phi)( FILE *, int, int );

#endif /* _MODEL_H_ */
