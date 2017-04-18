/*****************************************************************************
 *
 *  field_phi_init.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FIELD_PHI_INIT_H
#define LUDWIG_FIELD_PHI_INIT_H

#include "field.h"

typedef struct field_phi_info_s field_phi_info_t;

struct field_phi_info_s {
  double xi0;             /* Equilibrium interfacial width */
  double phi0;            /* Mean composition */
  double phistar;         /* An amplitude (min/max order parameter) */
};

int field_phi_init_drop(double xi, double radius, double phistar,
	field_t * phi);
int field_phi_init_block(double xi, field_t * phi);
int field_phi_init_block_X(double xi, field_t * phi, double block_dimension);
int field_phi_init_block_Y(double xi, field_t * phi, double block_dimension);
int field_phi_init_block_Z(double xi, field_t * phi, double block_dimension);
int field_phi_init_layer_X(double xi, field_t * phi, double layer_size);
int field_phi_init_layer_Y(double xi, field_t * phi, double layer_size);
int field_phi_init_layer_Z(double xi, field_t * phi, double layer_size);
int field_phi_init_bath(field_t * phi);
int field_phi_init_spinodal(int seed, double phi0, double amp, field_t * phi);
int field_phi_init_spinodal_patches(int seed, int patch, double volm1,
	field_t * phi);
int field_phi_init_emulsion(double xi, double radius, double phistar, 
	int Ndrops, double d_centre, field_t * phi);

#endif
