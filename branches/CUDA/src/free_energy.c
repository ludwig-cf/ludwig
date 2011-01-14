/*****************************************************************************
 *
 *  free_energy
 *
 *  This is an 'abstract' free energy interface.
 *
 *  Any concrete implementation should implement one or more of the
 *  following functions to override the default 'null' free energy:
 *
 *    double free_energy_density(const int index)
 *    double chemical_potential(const int index, const int nop)
 *    double isotropic_pressure(const int index)
 *    void   chemical_stress(const int index, double s[3][3])
 *
 *  The choice of free energy is also related to the form of the order
 *  parameter, and sets the highest order of spatial derivatives of
 *  the order parameter required for the free energy calculations.
 *
 *  It is assumed that the order parameter and order parameter
 *  gradients are arranged appropriately at the point of selecting
 *  the free energy.
 *
 *  $Id: free_energy.c,v 1.16 2010-10-15 12:40:02 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2009 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include "free_energy.h"


static double fe_fed_null(const int index);
static double fe_mu_null(const int index, const int nop);
static double fe_iso_null(const int index);
static void   fe_pth_null(const int index, double s[3][3]);

static double (* fp_fed)(const int index) = fe_fed_null;
static double (* fp_mu)(const int index, const int nop) = fe_mu_null;
static double (* fp_iso)(const int index) = fe_iso_null;
static void   (* fp_pth)(const int index, double sth[3][3]) = fe_pth_null;
static double kappa_ = 1.0;

/****************************************************************************
 *
 *  fe_density_set
 *
 *  Set the function pointer for the required free_energy density.
 *
 ****************************************************************************/

void fe_density_set(double (* f)(const int)) {

  assert(f);
  fp_fed = f;
  return;
}

/****************************************************************************
 *
 *  fe_chemical_potential_set
 *
 *  Sets the function pointer for the required chemical potential.
 *
 ****************************************************************************/

void fe_chemical_potential_set(double (* f)(const int, const int)) {

  assert(f);
  fp_mu = f;
  return;
}

/****************************************************************************
 *
 *  fe_isotropic_pressure_set
 *
 *  Sets the function pointer for the isotropic pressure.
 *
 ****************************************************************************/

void fe_isotropic_pressure_set(double (* f)(const int)) {

  assert(f);
  fp_iso = f;
  return;
}

/****************************************************************************
 *
 *  fe_chemical_stress_set
 *
 *  Sets the function pointer for the chemical stress.
 *
 ****************************************************************************/

void fe_chemical_stress_set(void (* f)(const int, double [3][3])) {
 
  assert(f);
  fp_pth = f;
  return;
}

/****************************************************************************
 *
 *  fe_density_function(void)
 *
 *  returns a pointer to a function taking (const int index)
 *  and returning the free energy density as a double.
 *
 ****************************************************************************/

double (* fe_density_function(void))(const int index) {

  return fp_fed;
}

/****************************************************************************
 *
 *  fe_chemical_potential_function(void)
 *
 *  returns a pointer to a function taking (const int index, const int nop)
 *  responsible for computing the chemical potential at index for order
 *  parameter nop.
 *
 ****************************************************************************/

double (* fe_chemical_potential_function(void))(const int, const int) {

  return fp_mu;
}

/****************************************************************************
 *
 *  fe_chemical_stress_function(void)
 *
 *  returns a pointer to a function taking (const int index, s[3][3])
 *  (and returning void) to get the chemical stress, s, at position
 *  index.
 *
 ****************************************************************************/

void (* fe_chemical_stress_function(void))(const int index, double s[3][3]) {

  return fp_pth;
}

/****************************************************************************
 *
 *  fe_fed_null
 *
 *  Default free energy density returns zero.
 *
 ****************************************************************************/

static double fe_fed_null(const int index) {

  return 0.0;
}

/****************************************************************************
 *
 *  fe_mu_null
 *
 *  Default chemical potential returns zero.
 *
 ****************************************************************************/

static double fe_mu_null(const int index, const int nop) {

  return 0.0;
}

/****************************************************************************
 *
 *  fe_iso_null
 *
 *  Default isotropic pressure is zero.
 *
 ****************************************************************************/

static double fe_iso_null(const int index) {

  return 0.0;
}

/****************************************************************************
 *
 *  fe_pth_null
 *
 *  Default chemical stress is zero.
 *
 ****************************************************************************/

static void fe_pth_null(const int index, double s[3][3]) {

  int ia, ib;

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      s[ia][ib] = 0.0;
    }
  }

  return;
}

/*****************************************************************************
 *
 *  fe_kappa
 *
 *****************************************************************************/

double fe_kappa(void) {

  return kappa_;
}

/*****************************************************************************
 *
 *  fe_kappa_set
 *
 *****************************************************************************/

void fe_kappa_set(const double kappa) {

  kappa_ = kappa;
  return;
}
