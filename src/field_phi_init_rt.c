/****************************************************************************
 *
 *  field_phi_init_rt.c
 *
 *  Run time initialisation for the composition field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>
#include <string.h>

#include "io_harness.h"
#include "field_phi_init_rt.h"

#define  DEFAULT_SEED       13
#define  DEFAULT_NOISE      0.1
#define  DEFAULT_PATCH_SIZE 1
#define  DEFAULT_PATCH_VOL  0.5
#define  DEFAULT_RADIUS     8.0

/*****************************************************************************
 *
 *  field_phi_init_rt
 *
 *  Initial choices for compositional order parameter.
 *
 *  Any information required from the free energy is via
 *  the parameters p.
 *
 *****************************************************************************/

int field_phi_init_rt(pe_t * pe, rt_t * rt, field_phi_info_t param,
		      field_t * phi) {

  int p;
  char value[BUFSIZ];
  char filestub[FILENAME_MAX];
  double radius;

  io_info_t * iohandler = NULL;

  assert(pe);
  assert(rt);
  assert(phi);

  p = rt_string_parameter(rt, "phi_initialisation", value, BUFSIZ);

  /* Default is spinodal */
  if (p == 0 || strcmp(value, "spinodal") == 0) {
    int seed = DEFAULT_SEED;
    double noise0 = DEFAULT_NOISE;  /* Amplitude of random noise */
    pe_info(pe, "Initialising phi for spinodal\n");
    rt_int_parameter(rt, "random_seed", &seed);
    rt_double_parameter(rt, "noise", &noise0);
    field_phi_init_spinodal(seed, param.phi0, noise0, phi);
  }

  if (p != 0 && strcmp(value, "patches") == 0) {
    int seed = DEFAULT_SEED;
    int patch = DEFAULT_PATCH_SIZE;
    double volminus1 = DEFAULT_PATCH_VOL;
    pe_info(pe, "Initialising phi in patches\n");

    rt_int_parameter(rt, "random_seed", &seed);
    rt_int_parameter(rt, "phi_init_patch_size", &patch);
    rt_double_parameter(rt, "phi_init_patch_vol", &volminus1);
    field_phi_init_spinodal_patches(seed, patch, volminus1, phi);
  }

  if (p != 0 && strcmp(value, "block") == 0) {
    pe_info(pe, "Initialisng phi as block\n");
    field_phi_init_block(param.xi0, phi);
  }

  if (p != 0 && strcmp(value, "bath") == 0) {
    pe_info(pe, "Initialising phi for bath\n");
    field_phi_init_bath(phi);
  }

  if (p != 0 && strcmp(value, "drop") == 0) {
    double phistar = 1.0;      /* "Amplitude", can be negative. */
    radius = DEFAULT_RADIUS;
    rt_double_parameter(rt, "phi_init_drop_radius", &radius);
    rt_double_parameter(rt, "phi_init_drop_amplitude", &phistar);
    pe_info(pe, "Initialising droplet radius:     %14.7e\n", radius);
    pe_info(pe, "Initialising droplet amplitude:  %14.7e\n", phistar);
    field_phi_init_drop(param.xi0, radius, phistar, phi);
  }

  if (p != 0 && strcmp(value, "from_file") == 0) {
    pe_info(pe, "Initial order parameter requested from file\n");
    strcpy(filestub, "phi.init"); /* A default */
    rt_string_parameter(rt, "phi_file_stub", filestub, FILENAME_MAX);
    pe_info(pe, "Attempting to read phi from file: %s\n", filestub);

    field_io_info(phi, &iohandler);
    io_read_data(iohandler, filestub, phi);
  }

  return 0;
}
