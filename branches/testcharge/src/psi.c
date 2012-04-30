/*****************************************************************************
 *
 *  psi.c
 *
 *  Note most of this has been coded  int function(...) in
 *  anticipation of exception handling.
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
#include "io_harness.h"
#include "psi.h"
#include "psi_s.h"

static const double e_unit_default = 1.0; /* Default unit charge */
static const double reltol_default = FLT_EPSILON; /* Solver tolerance */
static const double abstol_default = 0.01*FLT_EPSILON;

psi_t * psi_ = NULL;

static int psi_init_mpi_indexed(psi_t * obj);
static int psi_read(FILE * fp, const int ic, const int jc, const int kc);
static int psi_write(FILE * fp, const int ic, const int jc, const int kc);
static int psi_read_ascii(FILE * fp, const int, const int, const int);
static int psi_write_ascii(FILE * fp, const int, const int, const int);

/*****************************************************************************
 *
 *  psi_init
 *
 *  Initialise psi_ and return.
 *
 *****************************************************************************/

int psi_init(int nk, psi_t ** refpsi_) {

  assert(refpsi_);

  psi_create(nk, &psi_);
  *refpsi_ = psi_;

  return 0;
}

/*****************************************************************************
 *
 *  psi_halo_psi
 *
 *****************************************************************************/

int psi_halo_psi(psi_t * psi) {

  assert(psi);

  psi_halo(1, psi->psi, psi->psihalo);

  return 0;
}

/*****************************************************************************
 *
 *  psi_halo_rho
 *
 *****************************************************************************/

int psi_halo_rho(psi_t * psi) {

  assert(psi);

  psi_halo(psi->nk, psi->rho, psi->rhohalo);

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
  psi_t * psi = NULL;

  assert(pobj);
  assert(nk > 1);

  nsites = coords_nsites();

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

  psi_init_mpi_indexed(psi);
  *pobj = psi; 

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
 *  psi_init_mpi_indexed
 *
 *  Here we define MPI_Type_indexed structures to take care of the
 *  halo swaps. These structures may be understood by comparing
 *  the extents and strides with the loops in the 'serial' halo swap
 *  in psi_halo().
 *
 *  The indexed structures are used so that the recieves in the
 *  different coordinate directions do not overlap anywhere. The
 *  receives may then all be posted independently.
 *
 *  Assumes the field storage is f[index][nk] where index is the
 *  usual spatial index returned by coords_index(), and nk is the
 *  number of fields (always 1 for psi, and usually 2 for rho).
 *
 *****************************************************************************/

static int psi_init_mpi_indexed(psi_t * obj) {

  int nk;
  int nhalo;
  int nlocal[3], nh[3];
  int ic, jc, n;

  int ncount;        /* Count for the indexed type */
  int * blocklen;    /* Array of block lengths */
  int * displace;    /* Array of displacements */

  assert(obj);

  nk = obj->nk;
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  nh[X] = nlocal[X] + 2*nhalo;
  nh[Y] = nlocal[Y] + 2*nhalo;
  nh[Z] = nlocal[Z] + 2*nhalo;

  /* X direction */
  /* For psi, we require nlocal[Y] contiguous strips of length nlocal[Z],
   * repeated for each halo layer. The strides start at zero, and
   * increment nh[Z] for each strip. */

  ncount = nhalo*nlocal[Y];

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nlocal[Z];
  }

  for (n = 0; n < nhalo; n++) {
    for (jc = 0; jc < nlocal[Y]; jc++) {
      displace[n*nlocal[Y] + jc] = n*nh[Y]*nh[Z] + jc*nh[Z];
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &obj->psihalo[X]);
  MPI_Type_commit(&obj->psihalo[X]);

  /* For rho, just multiply block lengths and displacements by nk */

  for (n = 0; n < ncount; n++) {
    blocklen[n] *= nk;
  }

  for (n = 0; n < nhalo; n++) {
    for (jc = 0; jc < nlocal[Y]; jc++) {
      displace[n*nlocal[Y] + jc] *= nk;
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &obj->rhohalo[X]);
  MPI_Type_commit(&obj->rhohalo[X]);

  free(displace);
  free(blocklen);

  /* Y direction */
  /* We can use nh[X] contiguous strips of nlocal[Z], repeated for
   * each halo region. The strides start at zero, and increment by
   * nh[Y]*nh[Z] for each strip. */

  ncount = nhalo*nh[X];

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nlocal[Z];
  }

  for (n = 0; n < nhalo; n++) {
    for (ic = 0; ic < nh[X] ; ic++) {
      displace[n*nh[X] + ic] = n*nh[Z] + ic*nh[Y]*nh[Z];
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &obj->psihalo[Y]);
  MPI_Type_commit(&obj->psihalo[Y]);

  for (n = 0; n < ncount; n++) {
    blocklen[n] *= nk;
  }

  for (n = 0; n < nhalo; n++) {
    for (ic = 0; ic < nh[X] ; ic++) {
      displace[n*nh[X] + ic] *= nk;
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &obj->rhohalo[Y]);
  MPI_Type_commit(&obj->rhohalo[Y]);

  free(displace);
  free(blocklen);

  /* Z direction */
  /* Here, we need nh[X]*nh[Y] small contiguous strips of nhalo, with
   * a stride between each of nh[Z]. */

  ncount = nh[X]*nh[Y];

  blocklen = calloc(ncount, sizeof(int));
  displace = calloc(ncount, sizeof(int));
  if (blocklen == NULL) fatal("calloc(blocklen) failed\n");
  if (displace == NULL) fatal("calloc(displace) failed\n");

  for (n = 0; n < ncount; n++) {
    blocklen[n] = nhalo;
  }

  for (ic = 0; ic < nh[X]; ic++) {
    for (jc = 0; jc < nh[Y]; jc++) {
      displace[ic*nh[Y] + jc] = ic*nh[Y]*nh[Z]+ jc*nh[Z];
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &obj->psihalo[Z]);
  MPI_Type_commit(&obj->psihalo[Z]);

  for (n = 0; n < ncount; n++) {
    blocklen[n] *= nk;
  }

  for (ic = 0; ic < nh[X]; ic++) {
    for (jc = 0; jc < nh[Y]; jc++) {
      displace[ic*nh[Y] + jc] *= nk;
    }
  }

  MPI_Type_indexed(ncount, blocklen, displace, MPI_DOUBLE, &obj->rhohalo[Z]);
  MPI_Type_commit(&obj->rhohalo[Z]);

  free(displace);
  free(blocklen);

  return 0;
}

/*****************************************************************************
 *
 *  psi_init_io_info
 *
 *****************************************************************************/

int psi_init_io_info(psi_t * obj, int grid[3]) {

  assert(obj);
  assert(grid);
  assert(obj->info == NULL);

  obj->info = io_info_create_with_grid(grid);
  if (obj->info == NULL) fatal("io_info_create(psi) failed\n");

  io_info_set_name(obj->info, "Potential and charge densities");
  io_info_set_read_binary(obj->info, psi_read);
  io_info_set_write_binary(obj->info, psi_write);
  io_info_set_read_ascii(obj->info, psi_read_ascii);
  io_info_set_write_ascii(obj->info, psi_write_ascii);
  io_info_set_bytesize(obj->info, (1 + obj->nk)*sizeof(double));

  io_info_set_format_binary(obj->info);

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
 *  psi_halo
 *
 *  A general halo swap where:
 *    nf is the number of 3-D fields
 *    f is the base address of the field
 *    halo are thre X, Y, Z MPI datatypes for the swap
 *
 *****************************************************************************/

int psi_halo(int nf, double * f, MPI_Datatype halo[3]) {

  int n, nh;
  int nhalo;
  int nlocal[3];
  int ic, jc, kc;
  int pback, pforw;          /* MPI ranks of 'left' and 'right' neighbours */
  int ihalo, ireal;          /* Indices of halo and 'real' lattice regions */
  MPI_Comm comm;             /* Cartesian communicator */
  MPI_Request req_recv[6];
  MPI_Request req_send[6];
  MPI_Status  status[6];

  const int btagx = 639, btagy = 640, btagz = 641;
  const int ftagx = 642, ftagy = 643, ftagz = 644;

  assert(f);

  comm = cart_comm();
  nhalo = coords_nhalo();
  coords_nlocal(nlocal);

  for (n = 0; n < 6; n++) {
    req_send[n] = MPI_REQUEST_NULL;
    req_recv[n] = MPI_REQUEST_NULL;
  }

  /* Post all recieves */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ihalo = nf*coords_index(nlocal[X] + 1, 1, 1);
    MPI_Irecv(f + ihalo,  1, halo[X], pforw, btagx, comm, req_recv);
    ihalo = nf*coords_index(1 - nhalo, 1, 1);
    MPI_Irecv(f + ihalo,  1, halo[X], pback, ftagx, comm, req_recv + 1);
  }

  if (cart_size(Y) > 1) {
    pback = cart_neighb(BACKWARD, Y);
    pforw = cart_neighb(FORWARD, Y);
    ihalo = nf*coords_index(1 - nhalo, nlocal[Y] + 1, 1);
    MPI_Irecv(f + ihalo,  1, halo[Y], pforw, btagy, comm, req_recv + 2);
    ihalo = nf*coords_index(1 - nhalo, 1 - nhalo, 1);
    MPI_Irecv(f + ihalo,  1, halo[Y], pback, ftagy, comm, req_recv + 3);
  }

  if (cart_size(Z) > 1) {
    pback = cart_neighb(BACKWARD, Z);
    pforw = cart_neighb(FORWARD, Z);
    ihalo = nf*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z] + 1);
    MPI_Irecv(f + ihalo,  1, halo[Z], pforw, btagz, comm, req_recv + 4);
    ihalo = nf*coords_index(1 - nhalo, 1 - nhalo, 1 - nhalo);
    MPI_Irecv(f + ihalo,  1, halo[Z], pback, ftagz, comm, req_recv + 5);
  }

  /* Now the sends */

  if (cart_size(X) > 1) {
    pback = cart_neighb(BACKWARD, X);
    pforw = cart_neighb(FORWARD, X);
    ireal = nf*coords_index(1, 1, 1);
    MPI_Issend(f + ireal, 1, halo[X], pback, btagx, comm, req_send);
    ireal = nf*coords_index(nlocal[X] - nhalo + 1, 1, 1);
    MPI_Issend(f + ireal, 1, halo[X], pforw, ftagx, comm, req_send + 1);
  }
  else {
    for (nh = 0; nh < nhalo; nh++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {
        for (kc = 1 ; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    ihalo = n + nf*coords_index(0 - nh, jc, kc);
            ireal = n + nf*coords_index(nlocal[X] - nh, jc, kc);
	    f[ihalo] = f[ireal];
	    ihalo = n + nf*coords_index(nlocal[X] + 1 + nh, jc,kc);
	    ireal = n + nf*coords_index(1 + nh, jc, kc);
	    f[ihalo] = f[ireal];
          }
        }
      }
    }
  }

  /* X recvs to be complete before Y sends */
  MPI_Waitall(2, req_recv, status);

  if (cart_size(Y) > 1) {
    pback = cart_neighb(BACKWARD, Y);
    pforw = cart_neighb(FORWARD, Y);
    ireal = nf*coords_index(1 - nhalo, 1, 1);
    MPI_Issend(f + ireal, 1, halo[Y], pback, btagy, comm, req_send + 2);
    ireal = nf*coords_index(1 - nhalo, nlocal[Y] - nhalo + 1, 1);
    MPI_Issend(f + ireal, 1, halo[Y], pforw, ftagy, comm, req_send + 3);
  }
  else {
    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1-nhalo; ic <= nlocal[X] + nhalo; ic++) {
        for (kc = 1; kc <= nlocal[Z]; kc++) {
	  for (n = 0; n < nf; n++) {
	    ihalo = n + nf*coords_index(ic, 0 - nh, kc);
	    ireal = n + nf*coords_index(ic, nlocal[Y] - nh, kc);
	    f[ihalo] = f[ireal];
	    ihalo = n + nf*coords_index(ic, nlocal[Y] + 1 + nh, kc);
	    ireal = n + nf*coords_index(ic, 1 + nh, kc);
	    f[ihalo] = f[ireal];
	  }
        }
      }
    }
  }

  /* Y recvs to be complete before Z sends */
  MPI_Waitall(2, req_recv + 2, status);

  if (cart_size(Z) > 1) {
    pback = cart_neighb(BACKWARD, Z);
    pforw = cart_neighb(FORWARD, Z);
    ireal = nf*coords_index(1 - nhalo, 1 - nhalo, 1);
    MPI_Issend(f + ireal, 1, halo[Z], pback, btagz, comm, req_send + 4);
    ireal = nf*coords_index(1 - nhalo, 1 - nhalo, nlocal[Z] - nhalo + 1);
    MPI_Issend(f + ireal, 1, halo[Z], pforw, ftagz, comm, req_send + 5);
  }
  else {
    for (nh = 0; nh < nhalo; nh++) {
      for (ic = 1 - nhalo; ic <= nlocal[X] + nhalo; ic++) {
        for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {
	  for (n = 0; n < nf; n++) {
	    ihalo = n + nf*coords_index(ic, jc, 0 - nh);
	    ireal = n + nf*coords_index(ic, jc, nlocal[Z] - nh);
	    f[ihalo] = f[ireal];
	    ihalo = n + nf*coords_index(ic, jc, nlocal[Z] + 1 + nh);
	    ireal = n + nf*coords_index(ic, jc, 1 + nh);
	    f[ihalo] = f[ireal];
	  }
	}
      }
    }
  }

  /* Finish */
  MPI_Waitall(2, req_recv + 4, status);
  MPI_Waitall(6, req_send, status);

  return 0;
}

/*****************************************************************************
 *
 *  psi_write_ascii
 *
 *****************************************************************************/

static int psi_write_ascii(FILE * fp, const int ic, const int jc,
			   const int kc) {
  int index;
  int n, nwrite;

  assert(fp);

  index = coords_index(ic, jc, kc);

  nwrite = fprintf(fp, "%22.15e ", psi_->psi[index]);
  if (nwrite != 23) fatal("fprintf(psi) failed at index %d\n", index);

  for (n = 0; n < psi_->nk; n++) {
    nwrite = fprintf(fp, "%22.15e ", psi_->rho[psi_->nk*index + n]);
    if (nwrite != 23) fatal("fprintf(psi) failed at index %d %d\n", index, n);
  }

  nwrite = fprintf(fp, "\n");
  if (nwrite != 1) fatal("fprintf() failed at index %d\n", index);

  return n;
}

/*****************************************************************************
 *
 *  psi_read_ascii
 *
 *****************************************************************************/

static int psi_read_ascii(FILE * fp, const int ic, const int jc,
			  const int kc) {
  int index;
  int n, nread;

  assert(fp);

  index = coords_index(ic, jc, kc);

  nread = fscanf(fp, "%le", psi_->psi + index);
  if (nread != 1) fatal("fscanf(psi) failed for %d\n", index);

  for (n = 0; n < psi_->nk; n++) {
    nread = fscanf(fp, "%le", psi_->rho + psi_->nk*index + n);
    if (nread != 1) fatal("fscanf(rho) failed for %d %d\n", index, n);
  }

  return n;
}

/*****************************************************************************
 *
 *  psi_write
 *
 *****************************************************************************/

static int psi_write(FILE * fp, const int ic, const int jc, const int kc) {

  int index;
  int n;

  assert(fp);

  index = coords_index(ic, jc, kc);
  n = fwrite(psi_->psi + index, sizeof(double), 1, fp);
  if (n != 1) fatal("fwrite(psi) failed at index %d\n", index);

  n = fwrite(psi_->rho + psi_->nk*index, sizeof(double), psi_->nk, fp);
  if (n != psi_->nk) fatal("fwrite(rho) failed at index %d", index);

  return n;
}

/*****************************************************************************
 *
 *  psi_read
 *
 *****************************************************************************/

static int psi_read(FILE * fp, const int ic, const int jc, const int kc) {

  int index;
  int n;

  assert(fp);

  index = coords_index(ic, jc, kc);
  n = fread(psi_->psi + index, sizeof(double), 1, fp);
  if (n != 1) fatal("fread(psi) failed at index %d\n", index);

  n = fread(psi_->rho + psi_->nk*index, sizeof(double), psi_->nk, fp);
  if (n != psi_->nk) fatal("fread(rho) failed at index %d\n", index);

  return n;
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

  *lb = obj->e*obj->e*obj->beta / (4.0*M_PI*obj->epsilon);

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
  assert(rho_b > 0.0);
  assert(ld);

  psi_bjerrum_length(obj, &lb);
  *ld = 1.0 / sqrt(8.0*M_PI*lb*rho_b);

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
 *****************************************************************************/

int psi_abstol(psi_t * obj, double * abstol) {

  assert(obj);
  assert(abstol);

  *abstol = obj->abstol;

  return 0;
}
