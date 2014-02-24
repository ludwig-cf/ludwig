/*
 * ludcoll.h: header file fo Ludwig collision benchmark. 
 * Alan Gray, November 2013
 */


#include "targetDP.h"

enum lattchunks {ALL,BULK,EDGES};

/* from coords.h */
enum cartesian_directions {X, Y, Z};

#define NDIST 2
#define ND 128
#define NALL (ND+2)
#define NVEL 19
#define NDATA NALL*NALL*NALL*NDIST*NVEL
#define NDIM 3
enum {NHYDRO = 1 + NDIM + NDIM*(NDIM+1)/2};

#define TOLERANCE 1.E-15



void readData(int nsites, int nfields, char* filename, double* targetData);
void compData(double *data, double *dataRef, int size);


TARGET_ENTRY void collision( double* __restrict__ f_d, 
					  const double* __restrict__ ftmp_d, 
					  const double* __restrict__ phi_site_d,		
					  const double* __restrict__ grad_phi_site_d,	
					  const double* __restrict__ delsq_phi_site_d,	
					  const double* __restrict__ force_d, 
			     double* __restrict__ velocity_d);


TARGET void collision_site(
					  double* __restrict__ f_d, 
					  const double* __restrict__ ftmp_d, 
					  const double* __restrict__ phi_site_d,		
					  const double* __restrict__ grad_phi_site_d,	
					  const double* __restrict__ delsq_phi_site_d,	
					  const double* __restrict__ force_d, 
					  double* __restrict__ velocity_d, const int index); 


TARGET void collision_site_NOILP(
					  double* __restrict__ f_d, 
					  const double* __restrict__ ftmp_d, 
					  const double* __restrict__ phi_site_d,		
					  const double* __restrict__ grad_phi_site_d,	
					  const double* __restrict__ delsq_phi_site_d,	
					  const double* __restrict__ force_d, 
					  double* __restrict__ velocity_d, const int index); 



TARGET void get_coords_from_index(int *ii,int *jj,int *kk,int index,int N[3]);

TARGET int get_linear_index(int ii,int jj,int kk,int N[3]);








