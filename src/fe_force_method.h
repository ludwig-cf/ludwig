/*****************************************************************************
 *
 *  fe_force_method.h
 *
 *  For implementation of force from free energy sector.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Contributing author:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FE_FORCE_METHOD_H
#define LUDWIG_FE_FORCE_METHOD_H

typedef enum {
  FE_FORCE_METHOD_INVALID,
  FE_FORCE_METHOD_NO_FORCE,                   /* Usually no hydrodynamics */
  FE_FORCE_METHOD_STRESS_DIVERGENCE,          /* Current default */
  FE_FORCE_METHOD_PHI_GRADMU,                 /* Simple phi grad mu version */
  FE_FORCE_METHOD_PHI_GRADMU_CORRECTION,      /* Version with conservation */
  FE_FORCE_METHOD_RELAXATION_SYMM,            /* Via LB collision */
  FE_FORCE_METHOD_RELAXATION_ANTI,            /* Antisymmetric case */
  FE_FORCE_METHOD_MAX
} fe_force_method_enum_t;

fe_force_method_enum_t fe_force_method_default(void);
fe_force_method_enum_t fe_force_method_to_enum(const char * string);
const char * fe_force_method_to_string(fe_force_method_enum_t method);

#endif
