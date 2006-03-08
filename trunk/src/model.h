#ifndef _MODEL_H
#define _MODEL_H


#ifdef _D3Q19_
/* Temporary include situation for d3q19... */

#include "d3q19.h"

/* Undefined at the moment */
static int BC_Map[NVEL];


#else

/*!
 * Define the number of velocities the model should have 
 */
enum { NVEL = 15 };

/* Other model-specific constants for D3Q15 */
/*!
 * Number of velocities crossing a LE plane (i.e., X=+/-1) 
 */
#define LE_N_VEL_XING 5

/* 
 * The Site type is the storage for the distribution function 
 * It must have the size of the array f declared at compile time
 * so it can be copied as an object
 */

/*! 
 * These are the vectors of the D3Q15 model we shall use
 *
 *     c(0) =( 0, 0, 0)         
 *     c(1) =( 1,-1,-1)     
 *     c(2) =( 1,-1, 1)     
 *     c(3) =( 1, 1,-1)     
 *     c(4) =( 1, 1, 1)     
 *     c(5) =( 0, 1, 0)   
 *     c(6) =( 1, 0, 0)     
 *     c(7) =( 0, 0, 1)     
 *     c(8) =(-1, 0, 0)    
 *     c(9) =( 0,-1, 0)   
 *     c(10)=( 0, 0,-1)     
 *     c(11)=(-1,-1,-1)     
 *     c(12)=(-1,-1, 1)       
 *     c(13)=(-1, 1,-1)     
 *     c(14)=(-1, 1, 1)     
 */ 

/*
 * Here are the lattice vectors, defined as above. They are not much used
 * only to check things 
 */

static int cv[NVEL][3] = {{ 0, 0, 0},
			  { 1,-1,-1},
			  { 1,-1, 1},
			  { 1, 1,-1},
			  { 1, 1, 1},
			  { 0, 1, 0},
			  { 1, 0, 0},
			  { 0, 0, 1},
			  {-1, 0, 0},
			  { 0,-1, 0},
			  { 0, 0,-1},
			  {-1,-1,-1},
			  {-1,-1, 1},
			  {-1, 1,-1},
                          {-1, 1, 1}};

/*! 
 * The weight vector -- again -- not used much
 */

static double wv[NVEL] = {2.0/9.0,
			 1.0/72.0,
			 1.0/72.0,
			 1.0/72.0,
			 1.0/72.0,
			 1.0/9.0,
			 1.0/9.0,
			 1.0/9.0,
			 1.0/9.0,
			 1.0/9.0,
			 1.0/9.0,
			 1.0/72.0,
			 1.0/72.0,
			 1.0/72.0,
			 1.0/72.0};

/*! 
 * Map between velocity vectors and their images 
 * Required for BBL bounce back code to work
 */

static int BC_Map[NVEL] = {  0,     /* 0th  element - mirror is 0  */
			    14,     /* 1st  element - mirror is 14 */
			    13,     /* 2nd  element - mirror is 13 */
			    12,     /* 3rd  element - mirror is 12 */
			    11,     /* 4th  element - mirror is 11 */
			     9,     /* 5th  element - mirror is 9  */
			     8,     /* 6th  element - mirror is 8  */
			     10,    /* 7th  element - mirror is 10 */
			     6,     /* 8th  element - mirror is 6  */
			     5,     /* 9th  element - mirror is 5  */
			     7,     /* 10th element - mirror is 7  */
			     4,     /* 11th element - mirror is 11 */
			     3,     /* 12th element - mirror is 3  */
			     2,     /* 13th element - mirror is 2  */
			     1 };   /* 14th element - mirror is 1  */

#endif
/*! 
 * \typedef This structure should include all site data you want to transfer 
 */

typedef struct { 

  double f[NVEL],
    g[NVEL];

} Site;

/*------------ Definition of Global structure -----------------------------*/ 

typedef struct {         /* Global variables live here */

  int     input_format,  /* Binary or ASCII i/o        */
          output_format; /* Binary or ASCII i/o        */

  double  rho,           /* Average simulation density */
          phi,           /* Average order parameter    */
          mobility,      /* Mobility: unnormalised     */
          noise,         /* Initial noise amplitude    */
          eta;           /* shear viscosity            */

  FVector force;         /* Momentum increment applied to each site */

  char    input_config[256],
          output_config[256];

} Global;

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
void    RAND_fluctuations(double fr[]);
void   RAND_init_fluctuations(void);
void    MODEL_limited_propagate(void);
void     MISC_set_mean_phi(Float);
Float    MISC_fluid_volume(void);


extern void (*MODEL_write_site)( FILE *, int, int );
extern void (*MODEL_write_phi)( FILE *, int, int );

#endif /* _MODEL_H_ */
