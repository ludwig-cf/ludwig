/*****************************************************************************
 *
 *  psi.c
 *
 *  Electrokinetics: field quantites for potential and charge densities,
 *  and a number of other relevant quantities.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <mpi.h>

#include "pe.h"
#include "coords.h"
#include "coords_field.h"
#include "io_harness.h"
#include "util.h"
#include "map.h"
#include "psi.h"
#include "psi_s.h"
#include "psi_gradients.h"
#include "physics.h"

static const double  e_unit_default = 1.0;              /* Default unit charge */
static const double  reltol_default = FLT_EPSILON;      /* Solver tolerance */
static const double  abstol_default = 0.01*FLT_EPSILON;
static const int     maxits_default = 10000;            /* Default number of iterations in Poisson solver */
static const int multisteps_default = 1;                /* Default number of multisteps in NPE */
static const int  skipsteps_default = 1;                /* Default number of skipped timesteps in Poisson solver */
static const double diffacc_default = 0;                /* Default diffusive accuracy in NPE for constant no. of multisteps */ 

static int psi_read(FILE * fp, int index, void * self);
static int psi_write(FILE * fp, int index, void * self);
static int psi_read_ascii(FILE * fp, int index, void * self);
static int psi_write_ascii(FILE * fp, int index, void * self);

/*****************************************************************************
 *
 *  psi_halo_psi
 *
 *****************************************************************************/

int psi_halo_psi(psi_t * psi) {

  int nhalo;
  assert(psi);

  nhalo = coords_nhalo();
  coords_field_halo(nhalo, 1, psi->psi, MPI_DOUBLE, psi->psihalo);

  return 0;
}

/*****************************************************************************
 *
 *  psi_halo_rho
 *
 *****************************************************************************/

int psi_halo_rho(psi_t * psi) {

  int nhalo;
  assert(psi);

  nhalo = coords_nhalo();
  coords_field_halo(nhalo, psi->nk, psi->rho, MPI_DOUBLE, psi->rhohalo);

  return 0;
}

/*****************************************************************************
 *
 *  psi_create
 *
 *  Initialise the electric potential, nk charge density fields.
 *
 *****************************************************************************/

int psi_create(int nk, psi_t ** pobj) {

  int nsites;
  int nhalo;
  psi_t * psi = NULL;

  assert(pobj);
  assert(nk > 1);

  nsites = coords_nsites();
  nhalo = coords_nhalo();

  psi = calloc(1, sizeof(psi_t));
  if (psi == NULL) fatal("Allocation of psi failed\n");

  psi->nk = nk;
  psi->psi = calloc(nsites, sizeof(double));
  if (psi->psi == NULL) fatal("Allocation of psi->psi failed\n");

  psi->rho = calloc(nk*nsites, sizeof(double));
  if (psi->rho == NULL) fatal("Allocation of psi->rho failed\n");

  psi->diffusivity = calloc(nk, sizeof(double));
  if (psi->diffusivity == NULL) fatal("calloc(psi->diffusivity) failed\n");

  psi->valency = calloc(nk, sizeof(int));
  if (psi->valency == NULL) fatal("calloc(psi->valency) failed\n");

  psi->e = e_unit_default;
  psi->reltol = reltol_default;
  psi->abstol = abstol_default;
  psi->maxits = maxits_default;
  psi->multisteps = multisteps_default;
  psi->skipsteps = skipsteps_default;
  psi->diffacc = diffacc_default;

  coords_field_init_mpi_indexed(nhalo, 1, MPI_DOUBLE, psi->psihalo);
  coords_field_init_mpi_indexed(nhalo, psi->nk, MPI_DOUBLE, psi->rhohalo);
  *pobj = psi; 

  return 0;
}

/*****************************************************************************
 *
 *  psi_io_info
 *
 *****************************************************************************/

int psi_io_info(psi_t * obj, io_info_t ** info) {

  assert(obj);
  assert(info);

  *info = obj->info;

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
 *  psi_valency_set
 *
 *****************************************************************************/

int psi_valency_set(psi_t * obj, int n, int iv) {

  assert(obj);
  assert(n < obj->nk);

  obj->valency[n] = iv;

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
 *  psi_diffusivity_set
 *
 *****************************************************************************/

int psi_diffusivity_set(psi_t * obj, int n, double diff) {

  assert(obj);
  assert(n < obj->nk);

  obj->diffusivity[n] = diff;

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
 *  psi_init_io_info
 *
 *  The I/O grid will be requested with Cartesian extent as given.
 *
 *  Register all the I/O functions, and set the input/output format
 *  appropriately.
 *
 *****************************************************************************/

int psi_init_io_info(psi_t * obj, int grid[3], int form_in, int form_out) {

  assert(obj);
  assert(grid);
  assert(obj->info == NULL);

  obj->info = io_info_create_with_grid(grid);
  if (obj->info == NULL) fatal("io_info_create(psi) failed\n");

  io_info_set_name(obj->info, "Potential and charge densities");

  io_info_read_set(obj->info, IO_FORMAT_BINARY, psi_read);
  io_info_read_set(obj->info, IO_FORMAT_ASCII, psi_read_ascii);
  io_info_write_set(obj->info, IO_FORMAT_BINARY, psi_write);
  io_info_write_set(obj->info, IO_FORMAT_ASCII, psi_write_ascii);

  io_info_set_bytesize(obj->info, (2 + obj->nk)*sizeof(double));

  io_info_format_set(obj->info, form_in, form_out);
  io_info_metadata_filestub_set(obj->info, "psi");

  return 0;
}

/*****************************************************************************
 *
 *  psi_free
 *
 *****************************************************************************/

void psi_free(psi_t * obj) {

  int n;

  assert(obj);

  for (n = 0; n < 3; n++) {
    MPI_Type_free(&obj->psihalo[n]);
    MPI_Type_free(&obj->rhohalo[n]);
  }

  if (obj->info) io_info_destroy(obj->info);

  free(obj->valency);
  free(obj->diffusivity);
  free(obj->rho);
  free(obj->psi);
  free(obj);
  obj = NULL;

  return;
}

/*****************************************************************************
 *
 *  psi_write_ascii
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

static int psi_write_ascii(FILE * fp, int index, void * self) {

  int n, nwrite;
  double rho_el;
  psi_t * obj = self;

  assert(obj);
  assert(fp);

  nwrite = fprintf(fp, "%22.15e ", obj->psi[index]);
  if (nwrite != 23) fatal("fprintf(psi) failed at index %d\n", index);

  for (n = 0; n < obj->nk; n++) {
    nwrite = fprintf(fp, "%22.15e ", obj->rho[obj->nk*index + n]);
    if (nwrite != 23) fatal("fprintf(rho) failed at index %d %d\n", index, n);
  }
  
  psi_rho_elec(obj, index, &rho_el);
  nwrite = fprintf(fp, "%22.15e ", rho_el);
  if (nwrite != 23) fatal("fprintf(rho_el) failed at index %d\n", index);

  nwrite = fprintf(fp, "\n");
  if (nwrite != 1) fatal("fprintf() failed at index %d\n", index);

  return 0;
}

/*****************************************************************************
 *
 *  psi_read_ascii
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

static int psi_read_ascii(FILE * fp, int index, void * self) {

  int n, nread;
  double rho_el;
  psi_t * obj = self;

  assert(fp);
  assert(self);

  nread = fscanf(fp, "%le", obj->psi + index);
  if (nread != 1) fatal("fscanf(psi) failed for %d\n", index);

  for (n = 0; n < obj->nk; n++) {
    nread = fscanf(fp, "%le", obj->rho + obj->nk*index + n);
    if (nread != 1) fatal("fscanf(rho) failed for %d %d\n", index, n);
  }

  nread = fscanf(fp, "%le", &rho_el);
  if (nread != 1) fatal("fscanf(rho_el) failed for %d %d\n", index, n);

  return 0;
}

/*****************************************************************************
 *
 *  psi_write
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

static int psi_write(FILE * fp, int index, void * self) {

  int n;
  double rho_el;
  psi_t * obj = self;

  assert(fp);
  assert(obj);

  n = fwrite(obj->psi + index, sizeof(double), 1, fp);
  if (n != 1) fatal("fwrite(psi) failed at index %d\n", index);

  n = fwrite(obj->rho + obj->nk*index, sizeof(double), obj->nk, fp);
  if (n != obj->nk) fatal("fwrite(rho) failed at index %d", index);

  psi_rho_elec(obj, index, &rho_el);
  n = fwrite(&rho_el, sizeof(double), 1, fp);
  if (n != 1) fatal("fwrite(rho_el) failed at index %d", index);

  return 0;
}

/*****************************************************************************
 *
 *  psi_read
 *
 *  Returns 0 on success.
 *
 *****************************************************************************/

static int psi_read(FILE * fp, int index, void * self) {

  int n;
  double rho_el;
  psi_t * obj = self;

  assert(fp);
  assert(obj);

  n = fread(obj->psi + index, sizeof(double), 1, fp);
  if (n != 1) fatal("fread(psi) failed at index %d\n", index);

  n = fread(obj->rho + obj->nk*index, sizeof(double), obj->nk, fp);
  if (n != obj->nk) fatal("fread(rho) failed at index %d\n", index);

  n = fread(&rho_el, sizeof(double), 1, fp);
  if (n != 1) fatal("fread(rho_el) failed at index %d\n", index);

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

  int n;
  double rho_elec = 0.0;

  assert(obj);
  assert(rho);

  for (n = 0; n < obj->nk; n++) {
    rho_elec += obj->e*obj->valency[n]*obj->rho[obj->nk*index + n];
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

  *rho = obj->rho[obj->nk*index + n];

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

  obj->rho[obj->nk*index + n] = rho;

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

  *psi = obj->psi[index];

  return 0;
}

/*****************************************************************************
 *
 *  psi_psi_set
 *
 ****************************************************************************/

int psi_psi_set(psi_t * obj, int index, double psi) {

  assert(obj);

  obj->psi[index] = psi;

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
 *  psi_unit_charge_set
 *
 *****************************************************************************/

int psi_unit_charge_set(psi_t * obj, double eunit) {

  assert(obj);

  obj->e = eunit;

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
 *  psi_beta_set
 *
 *****************************************************************************/

int psi_beta_set(psi_t * obj, double beta) {

  assert(obj);

  obj->beta = beta;

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
 *  psi_epsilon_set
 *
 *****************************************************************************/

int psi_epsilon_set(psi_t * obj, double epsilon) {

  assert(obj);

  obj->epsilon = epsilon;

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
 *  psi_epsilon2_set
 *
 *****************************************************************************/

int psi_epsilon2_set(psi_t * obj, double epsilon2) {

  assert(obj);

  obj->epsilon2 = epsilon2;

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

  int n;
  assert(psi);
  assert(sion);

  *sion = 0.0;
  for (n = 0; n < psi->nk; n++) {
    *sion += 0.5*psi->valency[n]*psi->valency[n]*psi->rho[psi->nk*index + n];
  }

  return 0;
}

/*****************************************************************************
 *
 *  psi_bjerrum_length
 *
 *  Is equal to e^2 / 4 pi epsilon k_B T
 *
 *****************************************************************************/

int psi_bjerrum_length(psi_t * obj, double * lb) {

  assert(obj);
  assert(lb);

  *lb = obj->e*obj->e*obj->beta / (4.0*pi_*obj->epsilon);

  return 0;
}


/*****************************************************************************
 *
 *  psi_bjerrum_length2
 *
 *  Is equal to e^2 / 4 pi epsilon2 k_B T if we have
 *  a dielectric contrast between the electrolytes.
 *
 *****************************************************************************/

int psi_bjerrum_length2(psi_t * obj, double * lb) {

  assert(obj);
  assert(lb);

  *lb = obj->e*obj->e*obj->beta / (4.0*pi_*obj->epsilon2);

  return 0;
}

/*****************************************************************************
 *
 *  psi_debye_length
 *
 *  Returns the Debye length for a simple, symmetric electrolyte.
 *  An ionic strength is required as input (see above); this
 *  accounts for the factor of 8 in the denominator.
 *
 *****************************************************************************/

int psi_debye_length(psi_t * obj, double rho_b, double * ld) {

  double lb;

  assert(obj);
  assert(ld);

  psi_bjerrum_length(obj, &lb);
  *ld = 1.0 / sqrt(8.0*pi_*lb*rho_b);

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

int psi_debye_length2(psi_t * obj, double rho_b, double * ld) {

  double lb;

  assert(obj);
  assert(ld);

  psi_bjerrum_length2(obj, &lb);
  *ld = 1.0 / sqrt(8.0*pi_*lb*rho_b);

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

  *reltol = obj->reltol;

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

  *abstol = obj->abstol;

  return 0;
}

/*****************************************************************************
 *
 *  psi_reltol_set
 *
 *****************************************************************************/

int psi_reltol_set(psi_t * obj, double reltol) {

  assert(obj);

  obj->reltol = reltol;

  return 0;
}

/*****************************************************************************
 *
 *  psi_abstol_set
 *
 *****************************************************************************/

int psi_abstol_set(psi_t * obj, double abstol) {

  assert(obj);

  obj->abstol = abstol;

  return 0;
}

/*****************************************************************************
 *
 *  psi_multisteps_set
 *
 *****************************************************************************/

int psi_multisteps_set(psi_t * obj, int multisteps) {

  assert(obj);
  assert(multisteps);

  obj->multisteps = multisteps;

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
 *  psi_maxits_set
 *
 *****************************************************************************/

int psi_maxits_set(psi_t * obj, int maxits) {

  assert(obj);
  assert(maxits);

  obj->maxits = maxits;

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

  *maxits = obj->maxits;

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
 *  psi_skipsteps_set
 *
 *****************************************************************************/

int psi_skipsteps_set(psi_t * obj, double skipsteps) {

  assert(obj);
  assert(skipsteps>=1);

  obj->skipsteps = skipsteps;

  return 0;
}

/*****************************************************************************
 *
 *  psi_skipsteps
 *
 *****************************************************************************/

int psi_skipsteps(psi_t * obj) {

  assert(obj);

  return obj->skipsteps;
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

  MPI_Comm comm;

  assert(psi);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);  
  comm = cart_comm();

  sum_local = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_psi(psi, index, &psi0);
	sum_local += psi0;
      }
    }
  }

  MPI_Allreduce(&sum_local, &psi_offset, 1, MPI_DOUBLE, MPI_SUM, comm);

  psi_offset /= (L(X)*L(Y)*L(Z));

  for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	index = coords_index(ic, jc, kc);

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
  double e0[3];
  double eps, reps;
  double beta;
  assert(psi);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_ntotal(ntotal);
  coords_nlocal_offset(noffset);

  physics_e0(e0);
  psi_epsilon(psi, &eps);
  reps = 1.0/eps;
  psi_beta(psi, &beta);

  if (cart_coords(X) == 0) {

    for (nh = 0; nh < nhalo; nh++) {
      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	  index = coords_index(0 - nh, jc, kc);

	  if (is_periodic(X)) {
	    psi->psi[index] += e0[X]*ntotal[X];   
	  }
	  else{
	    index1 = coords_index(1, jc, kc);
	    psi->psi[index] = psi->psi[index1];   
	  }

	}
      }
    }

  }

  if (cart_coords(X) == cart_size(X)-1) {

    for (nh = 0; nh < nhalo; nh++) {
      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	    index = coords_index(nlocal[0] + 1 + nh, jc, kc);

	    if (is_periodic(X)) {
	      psi->psi[index] -= e0[X]*ntotal[X];   
	    }
	    else{
	      index1 = coords_index(nlocal[0], jc, kc);
	      psi->psi[index] = psi->psi[index1];   
	    }

	}
      }
    }  

  }

  if (cart_coords(Y) == 0) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	    index = coords_index(ic, 0 - nh, kc);

	    if (is_periodic(Y)) {
	      psi->psi[index] += e0[Y]*ntotal[Y];   
	    }
	    else{
	      index1 = coords_index(ic, 1, kc);
	      psi->psi[index] = psi->psi[index1];   
	    }

	}
      }
    }  

  }

  if (cart_coords(Y) == cart_size(Y)-1) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {

	  index = coords_index(ic, nlocal[1] + 1 + nh, kc);

	  if (is_periodic(Y)) {
	    psi->psi[index] -= e0[Y]*ntotal[Y];   
	  }
	  else{
	    index1 = coords_index(ic, nlocal[1], kc);
	    psi->psi[index] = psi->psi[index1];   
	  }

	}
      }
    }  

  }

  if (cart_coords(Z) == 0) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

	  index = coords_index(ic, jc, 0 - nh);

	  if (is_periodic(Z)) {
	    psi->psi[index] += e0[Z]*ntotal[Z];   
	  }
	  else{
	    index1 = coords_index(ic, jc, 1);
	    psi->psi[index] = psi->psi[index1];   
	  }

	}
      }
    }  

  }

  if (cart_coords(Z) == cart_size(Z)-1) {

    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
	for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

	  index = coords_index(ic, jc, nlocal[2] + 1 + nh);

	  if (is_periodic(Z)) {
	    psi->psi[index] -= e0[Z]*ntotal[Z];   
	  }
	  else{
	    index1 = coords_index(ic, jc, nlocal[2]);
	    psi->psi[index] = psi->psi[index1];   
	  }

	}
      }
    }  

  }

  return 0;
}
