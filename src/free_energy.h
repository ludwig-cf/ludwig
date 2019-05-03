/*****************************************************************************
 *
 *  free_energy.h
 *
 *  The 'abstract' free energy interface.
 *  (Abstract means there's is no corresponding implementation free_energy.c)
 *
 *  For instructions on how to add a free energy, see below.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_FREE_ENERGY_H
#define LUDWIG_FREE_ENERGY_H

/* We want the simd vector length */

#include "memory.h"

enum fe_id_enum {FE_SYMMETRIC,
		 FE_BRAZOVSKII,
                 FE_SURFACTANT1,
		 FE_POLAR,
		 FE_LC,
		 FE_ELECTRO,
		 FE_ELECTRO_SYMMETRIC,
		 FE_LC_DROPLET};

typedef struct fe_s fe_t;
typedef struct fe_vt_s fe_vt_t;

typedef int (* fe_free_ft)(fe_t * fe);
typedef int (* fe_id_ft)(fe_t * fe);
typedef int (* fe_target_ft)(fe_t * fe, fe_t ** target);
typedef int (* fe_fed_ft)(fe_t * fe, int index, double * fed);
typedef int (* fe_mu_ft)(fe_t * fe, int index, double * mu);
typedef int (* fe_str_ft)(fe_t * fe, int index, double s[3][3]);
typedef int (* fe_mu_solv_ft)(fe_t * fe, int index, int k, double * mu);
typedef int (* fe_hvector_ft)(fe_t * fe, int index, double h[3]);
typedef int (* fe_htensor_ft)(fe_t * fe, int index, double h[3][3]);
/* Vectorised versions */
typedef void (* fe_htensor_v_ft)(fe_t * fe, int index,double h[3][3][NSIMDVL]);
typedef void (* fe_stress_v_ft)(fe_t * fe, int index, double s[3][3][NSIMDVL]);

struct fe_vt_s {
  /* Order is important: actual tables must appear thus... */
  fe_free_ft free;              /* Virtual destructor */
  fe_target_ft target;          /* Return target pointer cast to fe_t */
  fe_fed_ft fed;                /* Freee energy density */
  fe_mu_ft mu;                  /* Chemical potential */
  fe_mu_solv_ft mu_solv;        /* Solvation chemical potential */
  fe_str_ft stress;             /* Chemical stress (total) */
  fe_str_ft str_symm;           /* Symmetric stress */
  fe_str_ft str_anti;           /* Antisymmetric stress */
  fe_hvector_ft hvector;        /* Vector molecular field */
  fe_htensor_ft htensor;        /* Tensor molecular field */
  fe_htensor_v_ft htensor_v;    /* Vectorised version */
  fe_stress_v_ft stress_v;      /* Vectorised stress (total) version */
  fe_stress_v_ft str_symm_v;    /* Symmetric part */
  fe_stress_v_ft str_anti_v;    /* Antisymmetric part */
};

struct fe_s {
  fe_vt_t * func;
  int id;                       /* enum id */
  int use_stress_relaxation;    /* use symmetric stress in collision */
};

/*****************************************************************************
 *
 * Abstract free energy, or free energy interface.
 *
 * There are four components about which define what is required
 * by an actual free energy:
 *
 *   1. The fe_id_enum provides a tag to label each free energy.
 *   2. The series of typedefs define the signatures of the functions
 *      which can be provided by a free energy. Not all are relevant
 *      for all free energies depending on whether the order parameter
 *      is scalar, vector, tensor, etc.
 *   3. A virtual function table or "vtable" which holds a series of
 *      pointers to the functions which compute the various quantities.
 *   4. The abstract free energy struct fs_t which holds the vtable and
 *      is used by parts of the code which don't care about the exact
 *      details of particular free energies.
 *
 * Adding a new free energy.
 *
 * If you want to add a new free energy, there are a number of steps to
 * follow.
 *
 *   1. Add a new tag to the fe_id_enum above e.g., FE_SURFACTANT2
 *
 *   2. Create new header and source files to hold the relevant code,
 *      e.g., fe_surfactant2.h and fe_surfactant2.c
 *
 *   3. Define a struct which will hold the relevant information about
 *      the new free energy, e.g.,
 *        struct fe_surf2_s {
 *          fe_t super;
 *          ...
 *        };
 *      The fe_t super component *must* appear first.
 *
 *   4. Define relevant functions with exactly the type signatures
 *      as given above. E.g., in fe_surfactant2.c
 *
 *        int fe_surf2_fed(fe_surf2_t * fe, int index, double * fed)
 *        int fe_surf2_mu(fe_surf2 * fe, int index, double * mu)
 *        int fe_surf2_str(fe_surf2 * fe, int index, double s[3][3])
 *
 *      which should compute the free energy density, the chemical
 *      potential(s), and the stress, respectively. The index argument
 *      is the single location on the lattice.
 *
 *   5. Define a static vtable structure and add the functions from
 *      stage4 to the vtable in the appropriate positions. If
 *      functions are not relevant, a NULL entry is acceptable.
 *
 *         static fe_vt_t fe_surf2_vtable = {
 *            ...
 *         };
 *
 *   6. When the the fe_surf1_t is brought into existance, one should
 *      set the components of the "superclass" appropriately:
 *
 *          fe->super.func = fe_surf2_vtable;
 *          fe->super.id   = FE_SURFACTANT2;
 *
 * Device implementations
 *
 * If a device implementation is required:
 *
 *   1.  The typedef'd functions defined above should have both
 *       __host__ and __device__ attributes.
 *
 *   2. A separate vtable for device version is required which should
 *      be in device memory, e.g., declare
 *
 *      static __constant__ fe_vt_t fe_surf_device_vtable = {
 *        ...
 *      };
 *
 *   3. On creation, the device vtable needs to be copied to the device.
 *      See e.g., symmetric.c for an example.
 *
 *****************************************************************************/

#endif
