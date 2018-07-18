/*****************************************************************************
 *
 *  pe_fenv.c
 *
 *  Help for inquiries around fenv.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2018 The University of Edinburgh
 *
 *  Contribtuing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <fenv.h>

#include "pe_fenv.h"

const char * const pe_fenv_unset      = "NO ROUNDING DIRECTION";
const char * const pe_fenv_tonearest  = "FE_TONEAREST";
const char * const pe_fenv_upward     = "FE_UPWARD";
const char * const pe_fenv_downward   = "FE_DOWNWARD";
const char * const pe_fenv_towardzero = "FE_TOWARDZERO";

/*****************************************************************************
 *
 *  pe_fegetround_tostring
 *
 *  Return rounding mode as a descriptive string.
 *
 *****************************************************************************/

const char * pe_fegetround_tostring(void) {

  switch (fegetround()) {
  case FE_TONEAREST:
    return pe_fenv_tonearest;
    break;
  case FE_UPWARD:
    return pe_fenv_upward;
    break;
  case FE_DOWNWARD:
    return pe_fenv_downward;
    break;
  case FE_TOWARDZERO:
    return pe_fenv_towardzero;
    break;
  }

  return pe_fenv_unset;
}
