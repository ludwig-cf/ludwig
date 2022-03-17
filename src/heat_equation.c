/*****************************************************************************
 *
 *  heat_equation.c
 *
 *  The time evolution of the temperature is described
 *
 *     d_t phi + div (u phi - M grad mu - \hat{xi}) = 0.
 *
 *  The equation is solved here via finite difference. The velocity
 *  field u is assumed known from the hydrodynamic sector. Lambda is the
 *  thermal diffusivity
 *  The choice of free energy should be symmetric_oft or any other free 
 *  energy which uses temperature field
 *
 *  Lees-Edwards planes are allowed (but not with noise, at present).
 *  This requires fixes at the plane boudaries to get consistent
 *  fluxes.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2021 The University of Edinburgh
 *
 *  Contributions:
 *  Jérémie Bertrand 
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "field_s.h"
#include "physics.h"
#include "advection_s.h"
#include "advection_bcs.h"
#include "util_sum.h"
#include "util.h"
#include "heat_equation.h"

static int heq_flux_mu1(heq_t * heq, field_t * temperature);
static int heq_flux_mu1_solid(heq_t * heq, field_t * temperature, map_t * map);
static int heq_colloid(heq_t * heq, field_t * temperature, colloids_info_t * cinfo);
static int heq_update_forward_step(heq_t * heq, field_t * temperaturef, map_t * map);

/* Utility container */

typedef struct heq_kernel_s heq_kernel_t;
struct heq_kernel_s {
  double lambda;      /* Mobility */
};

__global__ void heq_flux_mu1_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le,
				       advflux_t * flux, field_t * temperature, double lambda);

__global__ void heq_colloid_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le,
				       field_t * temperature, colloids_info_t * cinfo);


__global__ void heq_flux_mu1_solid_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le, field_t * temperature,
				       advflux_t * flux, map_t * map,
					double lambda);


/* The following is for 2D systems */
__global__ void heq_ufs_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				  field_t * field, advflux_t * flux, map_t * map,
				  int ys, double wz);
__global__ void heq_csum_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				   field_t * field, advflux_t * flux,
				   field_t * csum, int ys, double wz);

/*****************************************************************************
 *
 *  heq_create
 *
 *****************************************************************************/

__host__ int heq_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			   heq_info_t * options,
			   heq_t ** heq) {

  heq_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(le);
  assert(options);
  assert(heq);

  obj = (heq_t *) calloc(1, sizeof(heq_t));
  assert(obj);
  if (obj == NULL) pe_fatal(pe, "calloc(heq_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;
  obj->le = le;
  obj->info = *options;
  advflux_le_create(pe, cs, le, 1, &obj->flux);

  if (obj->info.conserve) {
    field_create(pe, cs, 1, "compensated sum", &obj->csum);
    field_init(obj->csum, 0, NULL);
  }

  pe_retain(pe);
  lees_edw_retain(le);

  *heq = obj;

  return 0;
}

/*****************************************************************************
 *
 *  heq_free
 *
 *****************************************************************************/

__host__ int heq_free(heq_t * heq) {

  assert(heq);

  lees_edw_free(heq->le);
  pe_free(heq->pe);

  if (heq->csum) field_free(heq->csum);
  advflux_free(heq->flux);
  free(heq);
  
  return 0;
}

/*****************************************************************************
 *
 *  heat_equation
 *
 *  Compute the fluxes (advective/diffusive) and compute the update
 *  to the order parameter field phi.
 *
 *  Conservation is ensured by face-flux uniqueness. However, in the
 *  x-direction, the fluxes at the east face and west face of a given
 *  cell must be handled spearately to take account of Lees Edwards
 *  boundaries.
 *
 *  hydro is allowed to be NULL, in which case the dynamics is
 *  just relaxational (no velocity field).
 *
 *  map is also allowed to be NULL, in which case there is no
 *  check for for the surface no flux condition. (This still
 *  may be relevant for diffusive fluxes only, so does not
 *  depend on hydrodynamics.)
 *
 *  The noise_t structure controls random fluxes, if required;
 *  it can be NULL in which case no noise.
 *
 *  TODO:
 *  advection_bcs_wall() may be required if 3rd or 4th order
 *  advection is used in the presence of solid. Not present
 *  at the moment.
 *
 *****************************************************************************/

int heat_equation(heq_t * heq, field_t * temperature,
		      hydro_t * hydro, map_t * map, colloids_info_t * cinfo) {

  int nf;
  field_nf(temperature, &nf);

  assert(nf == 1); 
  assert(heq);
  assert(temperature);


  /* Assign temperature field on colloid nodes
     TODO: Do it from the colloids.c subroutine ? */
  if (cinfo) {
    heq_colloid(heq, temperature, cinfo); 
  }

  /* Advective fluxes */
  if (hydro) {
    hydro_u_halo(hydro); 
    hydro_lees_edwards(hydro);
    advection_x(heq->flux, hydro, temperature);
  }
  else {
    advflux_zero(heq->flux);
  }

  /* Diffusive fluxes */
  heq_flux_mu1(heq, temperature); 

  /* No flux boundaries  */
  if (map) {
    advection_bcs_no_normal_flux(nf, heq->flux, map);
    heq_flux_mu1_solid(heq, temperature, map);
  }

  /* Sum fluxes and update */
  heq_update_forward_step(heq, temperature, map);

  return 0;
}

/*****************************************************************************
 *
 *  heq_colloid
 *
 *  Kernel driver for diffusive flux computation.
 *
 *****************************************************************************/

static int heq_colloid(heq_t * heq, field_t * temperature, colloids_info_t * cinfo) {

  int nlocal[3];
  double lambda;
  dim3 nblk, ntpb;
  kernel_info_t limits;

  physics_t * phys = NULL;
  lees_edw_t * letarget = NULL;
  kernel_ctxt_t * ctxt = NULL;

  assert(heq);
  assert(temperature);
  assert(cinfo);

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &letarget);

  physics_ref(&phys);
  physics_lambda(phys, &lambda);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_colloid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, temperature->target, cinfo->target);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  heq_colloid_kernel
 *
 *  Unvectorised kernel.
 *
 *  Set the temperature of solid node of a colloid to model an assymetrically-
 *  heated Janus colloid. One side is at temperature Tj1, the other at Tj2.
 *
 *  TODO: Angle needs rework but for now if jangle = 1.57 the colloid will be split
 *  in half.
 *
 *****************************************************************************/

__global__ void heq_colloid_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le, 
				        field_t * temperature,
					colloids_info_t * cinfo) {
  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(le);
  assert(cinfo);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index;

    colloid_t * pc = NULL;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = kernel_coords_index(ktx, ic, jc, kc);
    
    colloids_info_map(cinfo, index, &pc); 
  /*  
    if (ic == 20) { 
      if (pc == NULL) printf("%d %d %d pc is NULL\n", ic, jc, kc);
      else printf("%d %d %d pc is not NULL\n", ic, jc, kc);
    }
  */
    if (pc && pc->s.type != COLLOID_TYPE_SUBGRID && pc->s.isjanus) {
      double r[3], r0[3], rb[3], m[3];
      double janus, cosine, norm_rb, norm_m;
      int ia;
      int noffset[3];
      
      cs_nlocal_offset(temperature->cs, noffset);
      r[X] = 1.0*(noffset[X] + ic);
      r[Y] = 1.0*(noffset[Y] + jc);
      r[Z] = 1.0*(noffset[Z] + kc);

      r0[X] = pc->s.r[X];
      r0[Y] = pc->s.r[Y];
      r0[Z] = pc->s.r[Z];
      
      /* Find distance to ndoe */
      cs_minimum_distance(temperature->cs, r0, r, rb);

      /* And identify which side of Janus colloid the node is */
      for (ia = 0; ia < 3; ia++) m[ia] = pc->s.m[ia];
      
      norm_m = sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
      norm_rb = sqrt(rb[0]*rb[0] + rb[1]*rb[1] + rb[2]*rb[2]);

      if (norm_m == 0) pe_fatal(cinfo->pe, "Directional vector must be non zero\n");

      //if (norm_rb == 0) cosine = 0.5;

      else {
        cosine = dot_product(rb, m) / (norm_rb*norm_m+0.0001);
	//printf("%f\n", cosine);
        assert(cosine <= 1.0);
        assert(cosine >= -1.0);
      } 
  
      janus = acos(cosine);

      if (janus >= -pc->s.jangle && janus <= pc->s.jangle) {
        field_scalar_set(temperature, index, pc->s.Tj1);
      }
      else {
        field_scalar_set(temperature, index, pc->s.Tj2);
      }

    /* Next site */
    }
  }
  return;
}


/*****************************************************************************
 *
 *  heq_flux_mu1
 *
 *  Kernel driver for diffusive flux computation.
 *
 *****************************************************************************/

static int heq_flux_mu1(heq_t * heq, field_t * temperature) {

  int nlocal[3];
  double lambda;
  dim3 nblk, ntpb;
  kernel_info_t limits;

  physics_t * phys = NULL;
  lees_edw_t * letarget = NULL;
  kernel_ctxt_t * ctxt = NULL;

  assert(heq);
  assert(temperature);

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &letarget);

  physics_ref(&phys);
  physics_lambda(phys, &lambda);

  limits.imin = 0; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_flux_mu1_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, heq->flux->target, temperature->target, lambda);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  heq_flux_mu1_kernel
 *
 *  Unvectorised kernel.
 *
 *  Accumulate [add to a previously computed advective flux] the
 *  'diffusive' contribution related gradients of temperature. It's
 *  computed everywhere regardless of fluid/solid status but the solid nodes
 *  fluxes are overwritten by heq_flux_mu1_solid.
 *
 *  This is a two point stencil the in the temperature and lambda is constant.
 *
 *****************************************************************************/

__global__ void heq_flux_mu1_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le,
				       advflux_t * flux, field_t * temperature, double lambda) {
  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(le);
  assert(flux);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index0, index1;
    int icm1, icp1;
    double T0, T1;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);

    index0 = lees_edw_index(le, ic, jc, kc);

    field_scalar(temperature, index0, &T0);
    /* x-direction (between ic-1 and ic) */

    index1 = lees_edw_index(le, icm1, jc, kc);
    field_scalar(temperature, index1, &T1);
    flux->fw[addr_rank0(flux->nsite, index0)] -= lambda*(T0 - T1);

    /* ...and between ic and ic+1 */

    index1 = lees_edw_index(le, icp1, jc, kc);
    field_scalar(temperature, index1, &T1);
    flux->fe[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);

    /* y direction */

    index1 = lees_edw_index(le, ic, jc+1, kc);
    field_scalar(temperature, index1, &T1);
    flux->fy[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);

    /* z direction */

    index1 = lees_edw_index(le, ic, jc, kc+1);
    field_scalar(temperature, index1, &T1);
    flux->fz[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);

    /* Next site */
  }
  return;
}


/*****************************************************************************
 *
 *  heq_flux_mu1_solid
 *
 *  Kernel driver for diffusive flux computation.
 *
 *****************************************************************************/

static int heq_flux_mu1_solid(heq_t * heq, field_t * temperature, map_t * map) {

  int nlocal[3];
  double lambda;
  dim3 nblk, ntpb;
  kernel_info_t limits;

  physics_t * phys = NULL;
  lees_edw_t * letarget = NULL;
  kernel_ctxt_t * ctxt = NULL;
  map_t * maptarget = NULL;

  assert(heq);
  assert(map);

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &letarget);
  maptarget = map->target;

  physics_ref(&phys);
  physics_lambda(phys, &lambda);

  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 0; limits.jmax = nlocal[Y];
  limits.kmin = 0; limits.kmax = nlocal[Z];

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_flux_mu1_solid_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, letarget, temperature->target, heq->flux->target, maptarget, lambda);
  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/*****************************************************************************
 *
 *  heq_flux_mu1_solid_kernel
 *
 *  Calculate the fluxes for all boundary (fluid) nodes and solid nodes which 
 *  are next to boundary nodes aka calculate the fluxes for fluid and solid nodes
 *  adjacent to the surface of a colloid
 *  
 *****************************************************************************/

__global__ void heq_flux_mu1_solid_kernel(kernel_ctxt_t * ktx,
				       lees_edw_t * le, field_t * temperature,
				       advflux_t * flux, map_t * map,
					double lambda) {
  int kindex;
  __shared__ int kiterations;

  assert(ktx);
  assert(le);
  assert(flux);
  assert(map);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    int ic, jc, kc;
    int index0, index1;
    int icm1, icp1;
    double T0, T1;

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    icm1 = lees_edw_ic_to_buff(le, ic, -1);
    icp1 = lees_edw_ic_to_buff(le, ic, +1);

    index0 = lees_edw_index(le, ic, jc, kc);
    if (map->status[index0] == MAP_FLUID) {
      field_scalar(temperature, index0, &T0);

      /* x-direction (between ic-1 and ic) */
    
      index1 = lees_edw_index(le, icm1, jc, kc);
      if (map->status[index1] == MAP_COLLOID) {
        field_scalar(temperature, index1, &T1);
        flux->fw[addr_rank0(flux->nsite, index0)] -= lambda*(T0 - T1);
      }
      /* ...and between ic and ic+1 */

      index1 = lees_edw_index(le, icp1, jc, kc);
      if (map->status[index1] == MAP_COLLOID) {
        field_scalar(temperature, index1, &T1);
        flux->fe[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);
      }
      /* y direction */

      index1 = lees_edw_index(le, ic, jc+1, kc);
      if (map->status[index1] == MAP_COLLOID) {
        field_scalar(temperature, index1, &T1);
        flux->fy[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);
      }
   	
      /* z direction */

      index1 = lees_edw_index(le, ic, jc, kc+1);
      if (map->status[index1] == MAP_COLLOID) {
        field_scalar(temperature, index1, &T1);
        flux->fz[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);
	//pe_info(flux->pe, "%d with %d, T0 = %f, T1 =%f, lambda = %f, fy -= %f\n",kc, kc+1, T0, T1, lambda, lambda*(T0 -T1));
      }
    }

    if (map->status[index0] == MAP_COLLOID) {
      field_scalar(temperature, index0, &T0);

      /* x-direction (between ic-1 and ic) */
    
      index1 = lees_edw_index(le, icm1, jc, kc);
      if (map->status[index1] == MAP_FLUID) {
        field_scalar(temperature, index1, &T1);
        flux->fw[addr_rank0(flux->nsite, index0)] -= lambda*(T0 - T1);
      }
      /* ...and between ic and ic+1 */

      index1 = lees_edw_index(le, icp1, jc, kc);
      if (map->status[index1] == MAP_FLUID) {
        field_scalar(temperature, index1, &T1);
        flux->fe[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);
      }
      /* y direction */

      index1 = lees_edw_index(le, ic, jc+1, kc);
      if (map->status[index1] == MAP_FLUID) {
        field_scalar(temperature, index1, &T1);
        flux->fy[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);
      }
   	
      /* z direction */

      index1 = lees_edw_index(le, ic, jc, kc+1);
      if (map->status[index1] == MAP_FLUID) {
        field_scalar(temperature, index1, &T1);
        flux->fz[addr_rank0(flux->nsite, index0)] -= lambda*(T1 - T0);
      }
    }
    /* Next site */
  }
  return;
}


/*****************************************************************************
 *
 *  heq_update_forward_step
 *
 *  Update phi_site at each site in turn via the divergence of the
 *  fluxes. This is an Euler forward step:
 *
 *  phi new = phi old - dt*(flux_out - flux_in)
 *
 *  The time step is the LB time step dt = 1. All sites are processed
 *  to include solid-stored values in the case of Langmuir-Hinshelwood.
 *  It also avoids a conditional on solid/fluid status.
 *
 *****************************************************************************/

static int heq_update_forward_step(heq_t * heq, field_t * temperature, map_t * map) {

  int nlocal[3];
  int xs, ys, zs;
  dim3 nblk, ntpb;
  double wz = 1.0;

  lees_edw_t * le = NULL;
  kernel_info_t limits;
  kernel_ctxt_t * ctxt = NULL;
  map_t * maptarget = NULL;

  lees_edw_nlocal(heq->le, nlocal);
  lees_edw_target(heq->le, &le);
  maptarget = map->target;

  lees_edw_strides(heq->le, &xs, &ys, &zs);
  limits.imin = 1; limits.imax = nlocal[X];
  limits.jmin = 1; limits.jmax = nlocal[Y];
  limits.kmin = 1; limits.kmax = nlocal[Z];
  if (nlocal[Z] == 1) wz = 0.0;

  kernel_ctxt_create(heq->cs, 1, limits, &ctxt);
  kernel_ctxt_launch_param(ctxt, &nblk, &ntpb);

  tdpLaunchKernel(heq_ufs_kernel, nblk, ntpb, 0, 0,
		  ctxt->target, le, temperature->target, heq->flux->target, maptarget, ys, wz);

  tdpAssert(tdpPeekAtLastError());
  tdpAssert(tdpDeviceSynchronize());

  kernel_ctxt_free(ctxt);

  return 0;
}

/******************************************************************************
 *
 *  heq_ufs_kernel
 *
 *  In 2-d systems need to eliminate the z fluxes (no chemical
 *  potential computed in halo region for 2d_5pt_fluid): this
 *  is done via "wz".
 *
 *****************************************************************************/

__global__ void heq_ufs_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				  field_t * field, advflux_t * flux, 
				map_t * map, int ys, double wz) {
  int kindex;
  int kiterations;
  int ic, jc, kc, index;
  double temperature;

  assert(ktx);
  assert(le);
  assert(field);
  assert(flux);
  assert(map);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);
    index = lees_edw_index(le, ic, jc, kc);
    /* no update of the solid nodes */
    if (map->status[index] == MAP_FLUID) {
      field_scalar(field, index, &temperature);
      temperature -= (+ flux->fe[addr_rank0(flux->nsite, index)]
	    - flux->fw[addr_rank0(flux->nsite, index)]
	    + flux->fy[addr_rank0(flux->nsite, index)]
	    - flux->fy[addr_rank0(flux->nsite, index - ys)]
	    + wz*flux->fz[addr_rank0(flux->nsite, index)]
	    - wz*flux->fz[addr_rank0(flux->nsite, index - 1)]);
      field_scalar_set(field, index, temperature);
    }
  }
  return;
}


/******************************************************************************
 *
 *  heq_ufs_kernel
 *
 *  In 2-d systems need to eliminate the z fluxes (no chemical
 *  potential computed in halo region for 2d_5pt_fluid): this
 *  is done via "wz".
 *
 *****************************************************************************/

__global__ void heq_csum_kernel(kernel_ctxt_t * ktx, lees_edw_t *le,
				   field_t * field, advflux_t * flux,
				   field_t * csum, int ys, double wz) {
  int kindex;
  int kiterations;
  int ic, jc, kc, index;

  assert(ktx);
  assert(le);
  assert(field);
  assert(flux);

  kiterations = kernel_iterations(ktx);

  for_simt_parallel(kindex, kiterations, 1) {

    kahan_t temperature = {};

    ic = kernel_coords_ic(ktx, kindex);
    jc = kernel_coords_jc(ktx, kindex);
    kc = kernel_coords_kc(ktx, kindex);

    index = lees_edw_index(le, ic, jc, kc);

    temperature.sum = field->data[addr_rank1(field->nsites, 1, index, 0)];
    temperature.cs  = csum->data[addr_rank1(csum->nsites, 1, index, 0)];

    kahan_add_double(&temperature, -flux->fe[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature,  flux->fw[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature, -flux->fy[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature,  flux->fy[addr_rank0(flux->nsite, index - ys)]);
    kahan_add_double(&temperature, -wz*flux->fz[addr_rank0(flux->nsite, index)]);
    kahan_add_double(&temperature,  wz*flux->fz[addr_rank0(flux->nsite, index - 1)]);

    csum->data[addr_rank1(csum->nsites, 1, index, 0)] = temperature.cs;
    field->data[addr_rank1(field->nsites, 1, index, 0)] = temperature.sum;
  }

  return;
}

