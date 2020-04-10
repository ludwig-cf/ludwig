/*****************************************************************************
 *
 *  visc_arrhenius.c
 *
 *  Arrhenius viscosity model.
 *
 *  For a mixture with composition variable phi, where we expect the
 *  composition -phistar <= phi <= +phistar, the viscosity may be
 *  computed as:
 *
 *    eta(phi) = eta_minus^(0.5*(1+phi/phistar)) eta_plus^(0.5*(1-phi/phistar))
 *
 *  where eta_minus is the viscosity of the phi = -phistar phase, and
 *  eta_plus is the viscosity of the +phistar phase.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2020
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "field_s.h"
#include "kernel.h"
#include "visc_arrhenius.h"


static __global__ void visc_update_kernel(kernel_ctxt_t * ktx,
				          visc_arrhenius_param_t visc_param,
					  field_t * phi, hydro_t * hydro);
/* Function table */

static const visc_vt_t vt_ = {
  (visc_free_ft)   visc_arrhenius_free,
  (visc_update_ft) visc_arrhenius_update,
  (visc_stats_ft)  visc_arrhenius_stats
};

/*****************************************************************************
 *
 *  visc_arrhenius_create
 *
 *****************************************************************************/

__host__ int visc_arrhenius_create(pe_t * pe, cs_t * cs, field_t * phi,
				   visc_arrhenius_param_t param,
				   visc_arrhenius_t ** pvisc) {

  visc_arrhenius_t * visc = NULL;

  assert(pe);
  assert(cs);
  assert(phi);

  visc = (visc_arrhenius_t *) calloc(1, sizeof(visc_arrhenius_t));
  assert(visc);
  if (visc == NULL) pe_fatal(pe, "calloc(visc_arrhenius_t) failed\n");

  visc->param =
    (visc_arrhenius_param_t *) calloc(1, sizeof(visc_arrhenius_param_t));
  assert(visc->param);
  if (visc->param == NULL) pe_fatal(pe, "calloc(visc_arrhenius_param_t)\n");
  *visc->param = param;

  visc->pe = pe;
  visc->cs = cs;
  visc->phi = phi;

  visc->super.func = &vt_;
  visc->super.id   = VISC_MODEL_ARRHENIUS;

  *pvisc = visc;

  return 0;
}

/*****************************************************************************
 *
 *  visc_arrhenius_free
 *
 *  Release resources.
 *
 *****************************************************************************/

__host__ int visc_arrhenius_free(visc_arrhenius_t * visc) {

  assert(visc);

  free(visc->param);
  free(visc);

  return 0;
}

/*****************************************************************************
 *
 *  visc_arrhenius_info
 *
 *****************************************************************************/

__host__ int visc_arrhenius_info(visc_arrhenius_t * visc) {

  pe_t * pe = NULL;

  assert(visc);
  assert(visc->pe);

  pe = visc->pe;

  pe_info(pe, "\n");
  pe_info(pe, "Viscosity model\n");
  pe_info(pe, "---------------\n");
  pe_info(pe, "Model:                       %14s\n",   "Arrhenius");
  pe_info(pe, "Viscosity (eta -ve phase):   %14.7e\n", visc->param->eta_minus);
  pe_info(pe, "Viscosity (eta +ve phase):   %14.7e\n", visc->param->eta_plus);
  pe_info(pe, "Composition limit (phistar): %14.7e\n", visc->param->phistar);

  return 0;
}

/*****************************************************************************
 *
 *  visc_arrhenius_stats
 *
 *  Produce statistics to an appropriate output channel.
 *
 *****************************************************************************/

__host__ int visc_arrhenius_stats(visc_arrhenius_t * visc, hydro_t * hydro) {

  assert(0); /* PENDING IMPLEMENTATION */

  return 0;
}

/*****************************************************************************
 *
 *  visc_arrhenius_update
 *
 *****************************************************************************/

__host__ int visc_arrhenius_update(visc_arrhenius_t * visc, hydro_t * hydro) {

  int nlocal[3];
  dim3 nblk, ntpb;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;

  assert(visc);
  assert(hydro);

  cs_nlocal(visc->cs, nlocal);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(visc->cs, NSIMDVL, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(visc_update_kernel, nblk, ntpb, 0, 0, ctxt->target,
		  *visc->param, visc->phi->target, hydro->target);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  visc_update_kernel
 *
 *****************************************************************************/

static __global__ void visc_update_kernel(kernel_ctxt_t * ktx,
				          visc_arrhenius_param_t visc_param,
					  field_t * phi, hydro_t * hydro) {
  int kindex;
  int kiter;

  assert(ktx);
  assert(phi);
  assert(hydro);

  kiter = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiter, 1) {

    int ic, jc, kc, index;
    double phi0;
    double etaplus;
    double etaminus;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = kernel_coords_index(ktx, ic, jc, kc);

    phi0 = phi->data[addr_rank0(phi->nsites, index)];
    phi0 = phi0/visc_param.phistar;

    etaminus = pow(visc_param.eta_minus, 0.5*(1.0 - phi0));
    etaplus  = pow(visc_param.eta_plus,  0.5*(1.0 + phi0));

    hydro->eta[addr_rank0(hydro->nsite, index)] = etaminus*etaplus;
  }

  return;
}
