/*****************************************************************************
 *
 *  psi.c
 *
 *  Electrokinetics: field quantites for potential and charge densities,
 *  and a number of other relevant quantities.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Oliver Henrich (ohenrich@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <limits.h>

#include "psi.h"

/*****************************************************************************
 *
 *  psi_create
 *
 *****************************************************************************/

int psi_create(pe_t * pe, cs_t * cs, const psi_options_t * opts,
	       psi_t ** pobj) {

  int ifail = 0;
  psi_t * psi = NULL;

  assert(pobj);

  psi = (psi_t *) calloc(1, sizeof(psi_t));
  if (psi == NULL) goto err;

  ifail = psi_initialise(pe, cs, opts, psi);
  if (ifail != 0) goto err;
  
  *pobj = psi;

  return 0;

 err:
  if (psi) free(psi);
  return -1;
}

/*****************************************************************************
 *
 *  psi_free
 *
 *****************************************************************************/

int psi_free(psi_t ** psi) {

  assert(psi);
  assert(*psi);

  psi_finalise(*psi);
  free(*psi);
  *psi = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  psi_initialise
 *
 *****************************************************************************/

int psi_initialise(pe_t * pe, cs_t * cs, const psi_options_t * opts,
		   psi_t * psi) {

  int ifail = 0;

  assert(pe);
  assert(cs);
  assert(opts);
  assert(psi);

  psi->pe = pe;
  psi->cs = cs;

  psi->nk = opts->nk;
  cs_nsites(cs, &psi->nsites);

  psi->e = opts->e;
  psi->beta = opts->beta;
  psi->epsilon = opts->epsilon1;
  psi->epsilon2 = opts->epsilon2;

  psi->e0[X] = opts->e0[X];
  psi->e0[Y] = opts->e0[Y];
  psi->e0[Z] = opts->e0[Z];

  psi->diffusivity = (double *) calloc(opts->nk, sizeof(double));
  psi->valency = (int *) calloc(opts->nk, sizeof(int));

  if (psi->diffusivity == NULL) pe_fatal(pe, "psi->diffusivity failed\n");
  if (psi->valency == NULL) pe_fatal(pe, "calloc(psi->valency) failed\n");

  for (int n = 0; n < opts->nk; n++) {
    psi->diffusivity[n] = opts->diffusivity[n];
    psi->valency[n]     = opts->valency[n];
  }

  /* Solver options */
  psi->solver = opts->solver;
  ifail = stencil_create(opts->solver.nstencil, &psi->stencil);

  /* Nernst-Planck */
  psi->multisteps = opts->nsmallstep;
  psi->diffacc = opts->diffacc;

  /* Other */
  {
    /* Unfortunately, "rho" is not available for the charge density,
     * as it would conflict with the fluid density. */
    lees_edw_t * le = NULL;
    field_create(pe, cs, le, "psi", &opts->psi, &psi->psi);
    field_create(pe, cs, le, "qsi", &opts->rho, &psi->rho);
  }

  psi->nfreq_io = INT_MAX;

  /* Copy of the options structure */
  psi->options = *opts;

  return ifail;
}

/*****************************************************************************
 *
 *  psi_finalise
 *
 *****************************************************************************/

int psi_finalise(psi_t * psi) {

  assert(psi->psi);

  stencil_free(&psi->stencil);
  field_free(psi->rho);
  field_free(psi->psi);

  free(psi->valency);
  free(psi->diffusivity);

  *psi = (psi_t) {0};

  return 0;
}

/*****************************************************************************
 *
 *  psi_halo_psi
 *
 *****************************************************************************/

int psi_halo_psi(psi_t * psi) {

  assert(psi);

  field_halo(psi->psi);

  return 0;
}

/*****************************************************************************
 *
 *  psi_halo_rho
 *
 *****************************************************************************/

int psi_halo_rho(psi_t * psi) {

  assert(psi);

  field_halo(psi->rho);

  return 0;
}

/*****************************************************************************
 *
 *  psi_nk
 *
 *****************************************************************************/

int psi_nk(psi_t * obj, int * nk) {

  assert(obj);

  *nk = obj->nk;

  return 0;
}

/*****************************************************************************
 *
 *  psi_valency
 *
 *****************************************************************************/

int psi_valency(psi_t * obj, int n, int * iv) {

  assert(obj);
  assert(n < obj->nk);
  assert(iv);

  *iv = obj->valency[n];

  return 0;
}

/*****************************************************************************
 *
 *  psi_diffusivity
 *
 *****************************************************************************/

int psi_diffusivity(psi_t * obj, int n, double * diff) {

  assert(obj);
  assert(n < obj->nk);
  assert(diff);

  *diff = obj->diffusivity[n];

  return 0;
}

/*****************************************************************************
 *
 *  psi_rho_elec
 *
 *  Return the total electric charge density at a point.
 *
 *****************************************************************************/

int psi_rho_elec(psi_t * obj, int index, double * rho) {

  double rho_elec = 0.0;

  assert(obj);
  assert(rho);

  for (int n = 0; n < obj->nk; n++) {
    int irho = addr_rank1(obj->nsites, obj->nk, index, n);
    rho_elec += obj->e*obj->valency[n]*obj->rho->data[irho];
  }
  *rho = rho_elec;

  return 0;
}

/*****************************************************************************
 *
 *  psi_rho
 *
 *****************************************************************************/

int psi_rho(psi_t * obj, int index, int n, double * rho) {

  assert(obj);
  assert(rho);
  assert(n < obj->nk);

  *rho = obj->rho->data[addr_rank1(obj->nsites, obj->nk, index, n)];

  return 0;
}

/*****************************************************************************
 *
 *  psi_rho_set
 *
 *****************************************************************************/

int psi_rho_set(psi_t * obj, int index, int n, double rho) {

  assert(obj);
  assert(n < obj->nk);

  obj->rho->data[addr_rank1(obj->nsites, obj->nk, index, n)] = rho;

  return 0;
}

/*****************************************************************************
 *
 *  psi_psi
 *
 *****************************************************************************/

int psi_psi(psi_t * obj, int index, double * psi) {

  assert(obj);
  assert(psi);

  *psi = obj->psi->data[addr_rank0(obj->nsites, index)];

  return 0;
}

/*****************************************************************************
 *
 *  psi_psi_set
 *
 ****************************************************************************/

int psi_psi_set(psi_t * obj, int index, double psi) {

  assert(obj);

  obj->psi->data[addr_rank0(obj->nsites, index)] = psi;

  return 0;
}

/*****************************************************************************
 *
 *  psi_unit_charge
 *
 *****************************************************************************/

int psi_unit_charge(psi_t * obj, double * eunit) {

  assert(obj);
  assert(eunit);

  *eunit = obj->e;

  return 0;
}

/*****************************************************************************
 *
 *  psi_beta
 *
 *****************************************************************************/

int psi_beta(psi_t * obj, double * beta) {

  assert(obj);
  assert(beta);

  *beta = obj->beta;

  return 0;
}

/*****************************************************************************
 *
 *  psi_epsilon
 *
 *****************************************************************************/

int psi_epsilon(psi_t * obj, double * epsilon) {

  assert(obj);
  assert(epsilon);

  *epsilon = obj->epsilon;

  return 0;
}

/*****************************************************************************
 *
 *  psi_epsilon2
 *
 *****************************************************************************/

int psi_epsilon2(psi_t * obj, double * epsilon2) {

  assert(obj);
  assert(epsilon2);

  *epsilon2 = obj->epsilon2;

  return 0;
}

/*****************************************************************************
 *
 *  psi_ionic_strength
 *
 *  This is (1/2) \sum_k z_k^2 rho_k. This is a number density, and
 *  doesn't contain the unit charge.
 *
 *****************************************************************************/

int psi_ionic_strength(psi_t * psi, int index, double * sion) {

  assert(psi);
  assert(sion);

  *sion = 0.0;
  for (int n = 0; n < psi->nk; n++) {
    *sion += 0.5*psi->valency[n]*psi->valency[n]
      *psi->rho->data[addr_rank1(psi->nsites, psi->nk, index, n)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_surface_potential
 *
 *  Returns the surface potential of a double layer for a simple,
 *  symmetric electrolyte. The surface charge sigma and bulk ionic
 *  strength rho_b of one species are required as input.
 *
 *  See, e.g., Lyklema "Fundamentals of Interface and Colloid Science"
 *             Volume II Eqs. 3.5.13 and 3.5.14.
 *
 *****************************************************************************/

int psi_surface_potential(psi_t * obj, double sigma, double rho_b,
			  double *sp) {
  double p;

  assert(obj);
  assert(sp);
  assert(obj->nk == 2);
  assert(obj->valency[0] == -obj->valency[1]);

  p = 1.0 / sqrt(8.0*obj->epsilon*rho_b / obj->beta);

  *sp = fabs(2.0 / (obj->valency[0]*obj->e*obj->beta)
	     *log(-p*sigma + sqrt(p*p*sigma*sigma + 1.0)));

  return 0;
}

/*****************************************************************************
 *
 *  psi_reltol
 *
 *  Only returns the default value; there is no way to set as yet; no test.
 *
 *****************************************************************************/

int psi_reltol(psi_t * obj, double * reltol) {

  assert(obj);
  assert(reltol);

  *reltol = obj->solver.reltol;

  return 0;
}

/*****************************************************************************
 *
 *  psi_abstol
 *
 *  Only returns the default value; there is no way to set as yet; no test.
 *
 *****************************************************************************/

int psi_abstol(psi_t * obj, double * abstol) {

  assert(obj);
  assert(abstol);

  *abstol = obj->solver.abstol;

  return 0;
}

/*****************************************************************************
 *
 *  psi_multisteps
 *
 *****************************************************************************/

int psi_multisteps(psi_t * obj, int * multisteps) {

  assert(obj);
  assert(multisteps);

  *multisteps = obj->multisteps;

  return 0;
}

/*****************************************************************************
 *
 *  psi_multistep_timestep
 *
 *****************************************************************************/

int psi_multistep_timestep(psi_t * obj, double * dt) {

  assert(obj);

  *dt = 1.0/obj->multisteps;

  return 0;
}

/*****************************************************************************
 *
 *  psi_maxits
 *
 *****************************************************************************/

int psi_maxits(psi_t * obj, int * maxits) {

  assert(obj);
  assert(maxits);

  *maxits = obj->solver.maxits;

  return 0;
}

/*****************************************************************************
 *
 *  psi_diffacc_set
 *
 *****************************************************************************/

int psi_diffacc_set(psi_t * obj, double diffacc) {

  assert(obj);
  assert(diffacc>=0);

  obj->diffacc = diffacc;

  return 0;
}

/*****************************************************************************
 *
 *  psi_diffacc
 *
 *****************************************************************************/

int psi_diffacc(psi_t * obj, double * diffacc) {

  assert(obj);
  assert(diffacc);

  *diffacc = obj->diffacc;

  return 0;
}

/*****************************************************************************
 *
 *  psi_zero_mean
 *
 *  Shift the potential by the current mean value.
 *
 *****************************************************************************/

int psi_zero_mean(psi_t * psi) {

  int ic, jc, kc, index;
  int nlocal[3];
  int nhalo;

  double psi0;
  double sum_local;
  double psi_offset;
  double ltot[3];

  MPI_Comm comm;

  assert(psi);

  cs_ltot(psi->cs, ltot);
  cs_nhalo(psi->cs, &nhalo);
  cs_nlocal(psi->cs, nlocal);  
  cs_cart_comm(psi->cs, &comm);

  sum_local = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);

	psi_psi(psi, index, &psi0);
	sum_local += psi0;
      }
    }
  }

  MPI_Allreduce(&sum_local, &psi_offset, 1, MPI_DOUBLE, MPI_SUM, comm);

  psi_offset /= (ltot[X]*ltot[Y]*ltot[Z]);

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);

	psi_psi(psi, index, &psi0);
	psi0 -= psi_offset;
	psi_psi_set(psi, index, psi0);
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_halo_psijump
 *
 *  Creates an offset in the halo region of the electrostatic potential
 *  to model an overall external electric field across the physical domain.
 *  The routine has to be called after each halo swap to add the offset.
 *
 *  The external field has to be oriented along one of the cardinal coordinate
 *  and the boundary conditions along this dimension have to be periodic.
 *
 *****************************************************************************/

int psi_halo_psijump(psi_t * psi) {

  int nhalo;
  int nlocal[3], ntotal[3], noffset[3];
  int index, index1;
  int ic, jc, kc, nh;
  int mpi_cartsz[3];
  int mpicoords[3];
  int periodic[3];
  double eps;
  double beta;

  double * psidata = psi->psi->data;

  assert(psi);

  cs_nhalo(psi->cs, &nhalo);
  cs_nlocal(psi->cs, nlocal);
  cs_ntotal(psi->cs, ntotal);
  cs_nlocal_offset(psi->cs, noffset);
  cs_cartsz(psi->cs, mpi_cartsz);
  cs_cart_coords(psi->cs, mpicoords);
  cs_periodic(psi->cs, periodic);

  psi_epsilon(psi, &eps);
  psi_beta(psi, &beta);

  if (mpicoords[X] == 0) {

    for (nh = 0; nh < nhalo; nh++) {
      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	  index = cs_index(psi->cs, 0 - nh, jc, kc);

	  if (periodic[X]) {
	    /* Add external potential */
	    psidata[addr_rank0(psi->nsites, index)] += psi->e0[X]*ntotal[X];
	  }
	  else{
	    /* Borrow fluid site ic = 1 */
	    index1 = cs_index(psi->cs, 1, jc, kc);
	    psidata[addr_rank0(psi->nsites, index)] =
	      psidata[addr_rank0(psi->nsites, index1)];   
	  }
	}
      }
    }

  }

  if (mpicoords[X] == mpi_cartsz[X]-1) {

    for (nh = 0; nh < nhalo; nh++) {
      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	  index = cs_index(psi->cs, nlocal[0] + 1 + nh, jc, kc);

	  if (periodic[X]) {
	    /* Subtract external potential */
	    psidata[addr_rank0(psi->nsites, index)] -= psi->e0[X]*ntotal[X];
	  }
	  else {
	    /* Borrow fluid site at end ... */
	    index1 = cs_index(psi->cs, nlocal[X], jc, kc);
	    psidata[addr_rank0(psi->nsites, index)] =
	      psidata[addr_rank0(psi->nsites, index1)];   
	  }
	}
      }
    }  
  }

  if (mpicoords[Y] == 0) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	  index = cs_index(psi->cs, ic, 0 - nh, kc);

	    if (periodic[Y]) {
	      /* Add external potential */
	      psidata[addr_rank0(psi->nsites, index)] += psi->e0[Y]*ntotal[Y];
	    }
	    else {
	      /* Not periodic ... just borrow from fluid site jc = 1 */
	      index1 = cs_index(psi->cs, ic, 1, kc);
	      psidata[addr_rank0(psi->nsites, index)] =
		psidata[addr_rank0(psi->nsites, index1)];   
	    }
	}
      }
    }  

  }

  if (mpicoords[Y] == mpi_cartsz[Y]-1) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	  index = cs_index(psi->cs, ic, nlocal[Y] + 1 + nh, kc);

	  if (periodic[Y]) {
	    /* Subtract external potential */
	    psidata[addr_rank0(psi->nsites, index)] -= psi->e0[Y]*ntotal[Y];
	  }
	  else {
	    /* Borrow fluid site at end */
	    index1 = cs_index(psi->cs, ic, nlocal[Y], kc);
	    psidata[addr_rank0(psi->nsites, index)] =
	      psidata[addr_rank0(psi->nsites, index1)];   
	  }
	}
      }
    }  

  }

  if (mpicoords[Z] == 0) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

	  index = cs_index(psi->cs, ic, jc, 0 - nh);

	  if (periodic[Z]) {
	    /* Add external potential */
	    psidata[addr_rank0(psi->nsites, index)] += psi->e0[Z]*ntotal[Z];
	  }
	  else {
	    /* Borrow fluid site kc = 1 */
	    index1 = cs_index(psi->cs, ic, jc, 1);
	    psidata[addr_rank0(psi->nsites, index)] =
	      psidata[addr_rank0(psi->nsites, index1)];   
	  }
	}
      }
    }  

  }

  if (mpicoords[Z] == mpi_cartsz[Z]-1) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

	  index = cs_index(psi->cs, ic, jc, nlocal[Z] + 1 + nh);

	  if (periodic[Z]) {
	    /* Subtract external potential */
	    psidata[addr_rank0(psi->nsites, index)] -= psi->e0[Z]*ntotal[Z];
	  }
	  else {
	    /* Borrow fluid site at end ... */
	    index1 = cs_index(psi->cs, ic, jc, nlocal[Z]);
	    psidata[addr_rank0(psi->nsites, index)] =
	      psidata[addr_rank0(psi->nsites, index1)];   
	  }
	}
      }
    }  

  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_method
 *
 *****************************************************************************/

int psi_force_method(psi_t * psi, int * flag) {

  assert(psi);

  *flag = psi->method;

  return 0;
}

/*****************************************************************************
 *
 *  psi_force_method
 *
 *****************************************************************************/

int psi_force_method_set(psi_t * psi, int flag) {

  assert(psi);
  assert(flag >= 0 && flag < PSI_FORCE_NTYPES);

  psi->method = flag;

  return 0;
}

/*****************************************************************************
 *
 *  psi_output_step
 *
 *****************************************************************************/

int psi_output_step(psi_t * psi, int its) {

  assert(psi);

  return (its % psi->nfreq_io == 0);
}

/*****************************************************************************
 *
 *  psi_electroneutral
 *
 *  To ensure overall electroneutrality, we consider the following:
 *
 *   (1) assume surface charges have been assigned
 *   (2) assume the fluid is initialised with the backgound charge density
 *       of the electrolyte 
 *   (3) assume some number of colloids has been initialised, each
 *       with a given charge.
 *
 *  We can then:
 *
 *   (1) compute the total charge in the system along with the
 *       total discrete solid volume, i.e. colloid and boundary sites
 *   (2) add the appropriate countercharge to the fluid sites to
 *       make the system overall electroneutral.
 *
 *  Note:
 *   (1) net colloid charge \sum_k z_k q_k is computed for k = 2.
 *   (2) the countercharge is distributed only in one fluid species.
 *
 *  This is a collective call in MPI.
 *
 *****************************************************************************/

int psi_electroneutral(psi_t * psi, map_t * map) {

  int ic, jc, kc, index;
  int nlocal[3];
  int n, nk;

  int vf = 0; /* total fluid volume */
  int vc = 0; /* total colloid volume */
  int vb = 0; /* total boundary volume */

  int nc;             /* species for countercharge */
  int valency[2];
  double qloc, qtot;  /* local and global charge */
  double rho, rhoi;   /* charge and countercharge densities */
  int status;

  MPI_Comm comm;

  psi_nk(psi, &nk);
  assert(nk == 2);

  cs_cart_comm(psi->cs, &comm);
  cs_nlocal(psi->cs, nlocal);

  /* determine total fluid, colloid and boundary volume */
  map_volume_allreduce(map, MAP_FLUID, &vf);
  map_volume_allreduce(map, MAP_COLLOID, &vc);
  map_volume_allreduce(map, MAP_BOUNDARY, &vb);

  qloc = 0.0;
  qtot = 0.0;

  for (n = 0; n < nk; n++) {
    psi_valency(psi, n, valency + n);
  }

  /* accumulate local charge */
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(psi->cs, ic, jc, kc);

	for (n = 0; n < nk; n++) {
	  psi_rho(psi, index, n, &rho);
	  qloc += valency[n]*rho;
	}

      }
    }
  }

  MPI_Allreduce(&qloc, &qtot, 1, MPI_DOUBLE, MPI_SUM, comm);

  /* calculate and apply countercharge on fluid */
  rhoi = fabs(qtot) / vf;

  nc = -1;
  if (qtot*valency[0] >= 0) nc = 1;
  if (qtot*valency[1] >= 0) nc = 0;
  assert(nc == 0 || nc == 1);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(psi->cs, ic, jc, kc);
	map_status(map, index, &status);

        if (status == MAP_FLUID) {
          psi_rho(psi, index, nc, &rho);
          rho += rhoi;
          psi_rho_set(psi, index, nc, rho);
        }

        /* Next site */
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_io_write
 *
 *  Convenience to write both psi, rho with extra information.
 *
 *****************************************************************************/

int psi_io_write(psi_t * psi, int nstep) {

  int ifail = 0;
  io_event_t io1 = {0};
  io_event_t io2 = {0};
  const char * extra = "electrokinetics";
  cJSON * json = NULL;

  ifail = psi_options_to_json(&psi->options, &json);
  if (ifail == 0) {
    io1.extra_name = extra;
    io2.extra_name = extra;
    io1.extra_json  = json;
    io2.extra_json  = json;
  }

  ifail += field_io_write(psi->psi, nstep, &io1);
  ifail += field_io_write(psi->rho, nstep, &io2);

  cJSON_Delete(json);

  return ifail;
}
