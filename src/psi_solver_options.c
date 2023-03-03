/*****************************************************************************
 *
 *  psi_solver_options.c
 *
 *  Container for Poission solver options.
 *
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
#include <limits.h>
#include <string.h>

#include "psi_solver_options.h"
#include "util.h"

/*****************************************************************************
 *
 *  psi_poisson_solver_to_string
 *
 *****************************************************************************/

const char * psi_poisson_solver_to_string(psi_poisson_solver_enum_t mytype) {

  const char * str = "invalid";

  switch (mytype) {
  case PSI_POISSON_SOLVER_SOR:
    str = "sor";
    break;
  case PSI_POISSON_SOLVER_PETSC:
    str = "petsc";
    break;
  case PSI_POISSON_SOLVER_NONE:
    str = "none";
    break;
  default:
    str = "invalid";
  }

  return str;
}

/*****************************************************************************
 *
 *  psi_poisson_solver_from_string
 *
 *****************************************************************************/

psi_poisson_solver_enum_t psi_poisson_solver_from_string(const char * str) {

  psi_poisson_solver_enum_t mytype = PSI_POISSON_SOLVER_INVALID;
  char value[BUFSIZ] = {0};

  strncpy(value, str, BUFSIZ-1);
  util_str_tolower(value, strlen(value));

  if (strcmp(value, "sor")   == 0) mytype = PSI_POISSON_SOLVER_SOR;
  if (strcmp(value, "petsc") == 0) mytype = PSI_POISSON_SOLVER_PETSC;
  if (strcmp(value, "none")  == 0) mytype = PSI_POISSON_SOLVER_NONE;

  return mytype;
}

/*****************************************************************************
 *
 *  psi_solver_options_default
 *
 *****************************************************************************/

psi_solver_options_t psi_solver_options_default(void) {

  psi_solver_options_t pso = psi_solver_options_type(PSI_POISSON_SOLVER_SOR);

  return pso;
}

/*****************************************************************************
 *
 *  psi_solver_options_type
 *
 *****************************************************************************/

psi_solver_options_t psi_solver_options_type(psi_poisson_solver_enum_t ptype) {

  psi_solver_options_t pso = {
    .psolver     = ptype,
    .maxits      = 10000,
    .verbose     = 0,
    .nfreq       = INT_MAX,
    .nstencil    = 7,
    .reltol      = 1.0e-08,
    .abstol      = 1.0e-15,
  };

  return pso;
}

/*****************************************************************************
 *
 *  psi_solver_options_to_json
 *
 *  Caller to release new json object when finished.
 *
 *****************************************************************************/

int psi_solver_options_to_json(const psi_solver_options_t * pso,
			       cJSON ** json) {
  int ifail = 0;

  if (pso == NULL || json == NULL || *json != NULL) {
    ifail = -1;
  }
  else {
    cJSON * obj = cJSON_CreateObject();

    cJSON_AddStringToObject(obj, "Solver type",
			    psi_poisson_solver_to_string(pso->psolver));
    cJSON_AddNumberToObject(obj, "Maximum iterations", pso->maxits);
    cJSON_AddNumberToObject(obj, "Level of verbosity", pso->verbose);
    cJSON_AddNumberToObject(obj, "Frequency of output", pso->nfreq);
    cJSON_AddNumberToObject(obj, "Stencil points", pso->nstencil);

    cJSON_AddNumberToObject(obj, "Relative tolerance", pso->reltol);
    cJSON_AddNumberToObject(obj, "Absolute tolerance", pso->abstol);

    *json = obj;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  psi_solver_options_from_json
 *
 *****************************************************************************/

int psi_solver_options_from_json(const cJSON * json,
				 psi_solver_options_t * pso) {

  int ifail = 0;

  if (json == NULL || pso == NULL) {
    ifail = -1;
  }
  else {
    cJSON * psolver = cJSON_GetObjectItem(json, "Solver type");
    cJSON * maxits  = cJSON_GetObjectItem(json, "Maximum iterations");
    cJSON * verbose = cJSON_GetObjectItem(json, "Level of verbosity");
    cJSON * nfreq   = cJSON_GetObjectItem(json, "Frequency of output");
    cJSON * nsten   = cJSON_GetObjectItem(json, "Stencil points");
    cJSON * reltol  = cJSON_GetObjectItem(json, "Relative tolerance");
    cJSON * abstol  = cJSON_GetObjectItem(json, "Absolute tolerance");

    /* There must be at least a solver type ... */

    if (psolver == NULL) {
      pso->psolver = PSI_POISSON_SOLVER_INVALID;
      ifail = -1;
    }
    else {
      char * str = cJSON_GetStringValue(psolver);
      pso->psolver = psi_poisson_solver_from_string(str);

      if (maxits)  pso->maxits   = cJSON_GetNumberValue(maxits);
      if (verbose) pso->verbose  = cJSON_GetNumberValue(verbose);
      if (nfreq)   pso->nfreq    = cJSON_GetNumberValue(nfreq);
      if (nsten)   pso->nstencil = cJSON_GetNumberValue(nsten);
      if (reltol)  pso->reltol   = cJSON_GetNumberValue(reltol);
      if (abstol)  pso->abstol   = cJSON_GetNumberValue(abstol);
    }
  }

  return ifail;
}
