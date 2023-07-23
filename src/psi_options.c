/*****************************************************************************
 *
 *  psi_options.c
 *
 *  Including default options for the Electrokinetic sector.
 *  And some derived quantities only dependent on the options.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>

#include "psi_options.h"
#include "util.h"

/*****************************************************************************
 *
 *  psi_options_default
 *
 *  The nhalo is that required for the potential and the charge density
 *  fields.
 *
 *****************************************************************************/

psi_options_t psi_options_default(int nhalo) {

  int nk = 2;

  psi_options_t opts = {.nk          = nk,
                        .e           = 1.0,
			.beta        = 1.0,
                        .epsilon1    = 10000.0,
                        .epsilon2    = 10000.0,
                        .e0          = {0.0, 0.0, 0.0},
                        .diffusivity = {0.01, 0.01, 0.01, 0.01},
                        .valency     = {+1, -1, +1, -1},
			.solver      = psi_solver_options_default(),
			.nsolver     = -1,
			.nsmallstep  = 1,
			.diffacc     = 0.0,
			.method      = PSI_FORCE_NONE,
			.psi         = {0},
			.rho         = {0}};

  opts.psi = field_options_ndata_nhalo(1,  nhalo);
  opts.rho = field_options_ndata_nhalo(nk, nhalo);

  return opts;
}

/*****************************************************************************
 *
 *  psi_options_to_json
 *
 *  Return a new cJSON object for the options provided.
 *
 *****************************************************************************/

int psi_options_to_json(const psi_options_t * opts, cJSON ** json) {

  int ifail = 0;

  assert(opts);

  if (json == NULL || *json != NULL) {
    ifail = -1;
  }
  else {
    int nk = opts->nk;
    cJSON * myjson = cJSON_CreateObject();
    cJSON * valencies = cJSON_CreateIntArray(opts->valency, nk);
    cJSON * diffusivities = cJSON_CreateDoubleArray(opts->diffusivity, nk);
    cJSON * electric_field = cJSON_CreateDoubleArray(opts->e0, 3);

    cJSON * solver_options = NULL;
    ifail = psi_solver_options_to_json(&opts->solver, &solver_options);

    cJSON_AddNumberToObject(myjson, "Number of species", nk);
    cJSON_AddNumberToObject(myjson, "Unit charge",       opts->e);
    cJSON_AddNumberToObject(myjson, "Boltzmann factor",  opts->beta);
    cJSON_AddNumberToObject(myjson, "First permittivity", opts->epsilon1);
    cJSON_AddNumberToObject(myjson, "Second permittivity", opts->epsilon2);

    cJSON_AddItemToObject(myjson,   "Valencies", valencies);
    cJSON_AddItemToObject(myjson,   "Diffusivities", diffusivities);
    cJSON_AddItemToObject(myjson,   "External field", electric_field);
    cJSON_AddItemToObject(myjson,   "Solver options", solver_options);

    *json = myjson;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_options_from_json
 *
 *****************************************************************************/

int psi_options_from_json(const cJSON * json, psi_options_t * opts) {

  int ifail = 0;

  if (json == NULL || opts == NULL) {
    ifail = -1;
  }
  else {
    cJSON * nk   = cJSON_GetObjectItem(json, "Number of species");
    cJSON * e    = cJSON_GetObjectItem(json, "Unit charge");
    cJSON * beta = cJSON_GetObjectItem(json, "Boltzmann factor");
    cJSON * epsilon1 = cJSON_GetObjectItem(json, "First permittivity");
    cJSON * epsilon2 = cJSON_GetObjectItem(json, "Second permittivity");
    cJSON * valencies = cJSON_GetObjectItem(json, "Valencies");
    cJSON * diffs = cJSON_GetObjectItem(json, "Diffusivities");
    cJSON * electric = cJSON_GetObjectItem(json, "External field");
    cJSON * solver = cJSON_GetObjectItem(json, "Solver options");

    if (nk)       opts->nk   = cJSON_GetNumberValue(nk);
    if (e)        opts->e    = cJSON_GetNumberValue(e);
    if (beta)     opts->beta = cJSON_GetNumberValue(beta);
    if (epsilon1) opts->epsilon1 = cJSON_GetNumberValue(epsilon1);
    if (epsilon2) opts->epsilon2 = cJSON_GetNumberValue(epsilon2);

    if (valencies) util_json_to_int_array(valencies, opts->valency, opts->nk);
    if (diffs) util_json_to_double_array(diffs, opts->diffusivity, opts->nk);
    if (electric) util_json_to_double_array(electric, opts->e0, 3);

    if (solver) ifail = psi_solver_options_from_json(solver, &opts->solver);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_bjerrum_length1
 *
 *  Is equal to e^2 / 4 pi epsilon1 k_B T
 *
 *  Sensible parameters give return value 0.
 *
 *****************************************************************************/

int psi_bjerrum_length1(const psi_options_t * opts, double * lbjerrum) {

  int ifail = 0;
  PI_DOUBLE(pi);

  assert(opts);
  assert(lbjerrum);

  if (opts->beta <= 0.0) ifail = -1;
  if (opts->epsilon1 <= 0.0) ifail = -1;

  {
    double e = opts->e;
    double kt = 1.0/opts->beta;
    double epsilon1 = opts->epsilon1;

    *lbjerrum = e*e / (4.0*pi*epsilon1*kt);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_bjerrum_length2
 *
 *  Is equal to e^2 / 4 pi epsilon2 k_B T if we have
 *  a dielectric contrast between the electrolytes.
 *
 *****************************************************************************/

int psi_bjerrum_length2(const psi_options_t * opts, double * lbjerrum) {

  int ifail = 0;
  PI_DOUBLE(pi);

  assert(opts);
  assert(lbjerrum);

  if (opts->beta <= 0.0) ifail = -1;
  if (opts->epsilon2 <= 0) ifail = -1;

  {
    double e = opts->e;
    double kt = 1.0/opts->beta;
    double epsilon2 = opts->epsilon2;

    *lbjerrum = e*e / (4.0*pi*epsilon2*kt);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_debye_length1
 *
 *  Returns the Debye length for a simple, symmetric electrolyte.
 *  An ionic strength is required as input (see above); this
 *  accounts for the factor of 8 in the denominator.
 *
 *****************************************************************************/

int psi_debye_length1(const psi_options_t * opts, double rho_b, double * ld) {

  double lb;
  PI_DOUBLE(pi);

  assert(opts);
  assert(ld);

  psi_bjerrum_length1(opts, &lb);
  *ld = 1.0 / sqrt(8.0*pi*lb*rho_b);

  return 0;
}

/*****************************************************************************
 *
 *  psi_debye_length2
 *
 *  Returns the Debye length for the second phase if we
 *  have a dielectric contrast between the electrolytes.
 *  An ionic strength is required as input (see above); this
 *  accounts for the factor of 8 in the denominator.
 *
 *****************************************************************************/

int psi_debye_length2(const psi_options_t * opts, double rho_b, double * ld) {

  double lb;
  PI_DOUBLE(pi);

  assert(opts);
  assert(ld);

  psi_bjerrum_length2(opts, &lb);
  *ld = 1.0 / sqrt(8.0*pi*lb*rho_b);

  return 0;
}
