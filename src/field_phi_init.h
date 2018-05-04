/*****************************************************************************
 *
 *  field_phi_init.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018 The University of Edinburgh
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

int field_phi_init_drop(field_t * phi, double xi, double radius,
			double phistar);
int field_phi_init_block(field_t * phi, double xi);
int field_phi_init_block_X(field_t * phi, double xi, double block_dimension);
int field_phi_init_block_Y(field_t * phi, double xi, double block_dimension);
int field_phi_init_block_Z(field_t * phi, double xi, double block_dimension);
int field_phi_init_layer_X(field_t * phi, double xi, double layer_size);
int field_phi_init_layer_Y(field_t * phi, double xi, double layer_size);
int field_phi_init_layer_Z(field_t * phi, double xi, double layer_size);
int field_phi_init_bath(field_t * phi);
int field_phi_init_spinodal(field_t * phi, int seed, double phi0, double amp);
int field_phi_init_spinodal_patches(field_t * phi, int seed, int patch,
				    double volm1);
int field_phi_init_emulsion(field_t * phi, double xi, double radius,
			    double phistar, int Ndrops, double d_centre);

#endif
