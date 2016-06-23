/*****************************************************************************
 *
 *  hydro.c
 *
 *  Hydrodynamic quantities: velocity, body force on fluid.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Alan Gray (alang@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h> 

#include "pe.h"
#include "coords.h"
#include "kernel.h"
#include "coords_field.h"
#include "leesedwards.h"
#include "io_harness.h"
#include "util.h"
#include "control.h" /* Can we move this into LE please */
#include "hydro_s.h"

static int hydro_lees_edwards_parallel(hydro_t * obj);
static int hydro_u_write(FILE * fp, int index, void * self);
static int hydro_u_write_ascii(FILE * fp, int index, void * self);
static int hydro_u_read(FILE * fp, int index, void * self);

static __global__
void hydro_field_set(hydro_t * hydro, double * field, const double z[NHDIM]);


/*****************************************************************************
 *
 *  hydro_create
 *
 *  We typically require a halo region for the velocity which is only
 *  one lattice site in width, i.e., nhcomm = 1. This is independent
 *  of the width of the halo region specified for coords object.
 *
 *****************************************************************************/

__host__ int hydro_create(int nhcomm, hydro_t ** pobj) {

  int ndevice;
  double * tmp;
  hydro_t * obj = (hydro_t *) NULL;

  assert(pobj);

  obj = (hydro_t *) calloc(1, sizeof(hydro_t));
  if (obj == NULL) fatal("calloc(hydro) failed\n");

  obj->nhcomm = nhcomm;

  obj->nsite = le_nsites();
  obj->u = (double *) calloc(NHDIM*obj->nsite, sizeof(double));
  if (obj->u == NULL) fatal("calloc(hydro->u) failed\n");

  obj->f = (double *) calloc(NHDIM*obj->nsite, sizeof(double));
  if (obj->f == NULL) fatal("calloc(hydro->f) failed\n");

  halo_swap_create(nhcomm, NHDIM, obj->nsite, &obj->halo);
  assert(obj->halo);

  halo_swap_handlers_set(obj->halo, halo_swap_pack_rank1, halo_swap_unpack_rank1);

  /* Allocate target copy of structure (or alias) */

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {

    targetCalloc((void **) &obj->target, sizeof(hydro_t));

    targetCalloc((void **) &tmp, NHDIM*obj->nsite*sizeof(double));
    copyToTarget(&obj->target->u, &tmp, sizeof(double *)); 

    targetCalloc((void **) &tmp, NHDIM*obj->nsite*sizeof(double));
    copyToTarget(&obj->target->f, &tmp, sizeof(double *)); 

    copyToTarget(&obj->target->nsite, &obj->nsite, sizeof(int));
  }

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_free
 *
 *****************************************************************************/

__host__ void hydro_free(hydro_t * obj) {

  int ndevice;
  double * tmp;

  assert(obj);

  targetGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    copyFromTarget(&tmp, &obj->target->u, sizeof(double *)); 
    targetFree(tmp);
    copyFromTarget(&tmp, &obj->target->f, sizeof(double *)); 
    targetFree(tmp);
    targetFree(obj->target);
  }

  halo_swap_free(obj->halo);
  free(obj->f);
  free(obj->u);
  free(obj);

  return;
}

/*****************************************************************************
 *
 *  hydro_memcpy
 *
 *****************************************************************************/

__host__ int hydro_memcpy(hydro_t * obj, int flag) {

  int ndevice;
  double * tmpu;
  double * tmpf;

  assert(obj);

  targetGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    /* Ensure we alias */
    assert(obj->target == obj);
  }
  else {
    copyFromTarget(&tmpf, &obj->target->f, sizeof(double *));
    copyFromTarget(&tmpu, &obj->target->u, sizeof(double *));

    switch (flag) {
    case cudaMemcpyHostToDevice:
      copyToTarget(tmpu, obj->u, NHDIM*obj->nsite*sizeof(double));
      copyToTarget(tmpf, obj->f, NHDIM*obj->nsite*sizeof(double));
      copyToTarget(&obj->target->nsite, &obj->nsite, sizeof(int));
      break;
    case cudaMemcpyDeviceToHost:
      copyFromTarget(obj->f, tmpf, NHDIM*obj->nsite*sizeof(double));
      copyFromTarget(obj->u, tmpu, NHDIM*obj->nsite*sizeof(double));
      break;
    default:
      fatal("Bad flag in hydro_memcpy\n");
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_halo
 *
 *****************************************************************************/

int hydro_u_halo(hydro_t * obj) {

  assert(obj);

  halo_swap_driver(obj->halo, obj->target->u);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_init_io_info
 *
 *  There is no read for the velocity; this should come from the
 *  distribution.
 *
 *****************************************************************************/

int hydro_init_io_info(hydro_t * obj, int grid[3], int form_in, int form_out) {

  assert(obj);
  assert(grid);
  assert(obj->info == NULL);

  obj->info = io_info_create_with_grid(grid);
  if (obj->info == NULL) fatal("io_info_create(hydro) failed\n");

  io_info_set_name(obj->info, "Velocity field");
  io_info_write_set(obj->info, IO_FORMAT_BINARY, hydro_u_write);
  io_info_write_set(obj->info, IO_FORMAT_ASCII, hydro_u_write_ascii);
  io_info_read_set(obj->info, IO_FORMAT_BINARY, hydro_u_read);
  io_info_set_bytesize(obj->info, NHDIM*sizeof(double));

  io_info_format_set(obj->info, form_in, form_out);
  io_info_metadata_filestub_set(obj->info, "vel");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_io_info
 *
 *****************************************************************************/

int hydro_io_info(hydro_t * obj, io_info_t ** info) {

  assert(obj);
  assert(obj->info); /* Should have been initialised */

  *info = obj->info;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local_set
 *
 *****************************************************************************/

int hydro_f_local_set(hydro_t * obj, int index, const double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->f[addr_hydro(index, ia)] = force[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local
 *
 *****************************************************************************/

int hydro_f_local(hydro_t * obj, int index, double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    force[ia] = obj->f[addr_hydro(index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_f_local_add
 *
 *  Accumulate (repeat, accumulate) the fluid force at site index.
 *
 *****************************************************************************/

int hydro_f_local_add(hydro_t * obj, int index, const double force[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->f[addr_hydro(index, ia)] += force[ia]; 
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_set
 *
 *****************************************************************************/

int hydro_u_set(hydro_t * obj, int index, const double u[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    obj->u[addr_hydro(index, ia)] = u[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u
 *
 *****************************************************************************/

int hydro_u(hydro_t * obj, int index, double u[3]) {

  int ia;

  assert(obj);

  for (ia = 0; ia < 3; ia++) {
    u[ia] = obj->u[addr_hydro(index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_zero
 *
 *****************************************************************************/

__host__ int hydro_u_zero(hydro_t * obj, const double uzero[3]) {

  dim3 nblk, ntpb;

  assert(obj);

  kernel_launch_param(obj->nsite, &nblk, &ntpb);
  __host_launch_kernel(hydro_field_set, nblk, ntpb,
		       obj->target, obj->target->u, uzero);
  targetDeviceSynchronise();

  return 0;
}


/*****************************************************************************
 *
 *  hydro_f_zero
 *
 *****************************************************************************/

__host__ int hydro_f_zero(hydro_t * obj, const double fzero[3]) {

  dim3 nblk, ntpb;

  assert(obj);

  kernel_launch_param(obj->nsite, &nblk, &ntpb);
  __host_launch_kernel(hydro_field_set, nblk, ntpb,
		       obj->target, obj->target->f, fzero);
  targetDeviceSynchronise();

  return 0;
}

/*****************************************************************************
 *
 *  hydro_zero
 *
 *****************************************************************************/

static __global__
void hydro_field_set(hydro_t * hydro, double * field, const double z[NHDIM]) {

  int kindex;

  __target_simt_parallel_for(kindex, hydro->nsite, 1) {
    int ia;
    for (ia = 0; ia < NHDIM; ia++) {
      field[addr_rank1(hydro->nsite, NHDIM, kindex, ia)] = z[ia];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  hydro_lees_edwards
 *
 *  Compute the 'look-across-the-boundary' values of the velocity field,
 *  and update the velocity buffer region accordingly.
 *
 *  The communication might be improved:
 *  - only one buffer either side of the planes needs to be set?
 *  - only one communication per y sub domain if more than one buffer?
 *
 *****************************************************************************/

int hydro_lees_edwards(hydro_t * obj) {

  int nhalo;
  int nlocal[3]; /* Local system size */
  int ib;        /* Index in buffer region */
  int ib0;       /* buffer region offset */
  int ic;        /* Index corresponding x location in real system */

  int jc, kc, ia, index0, index1, index2;

  double dy;     /* Displacement for current ic->ib pair */
  double fr;     /* Fractional displacement */
  double t;      /* Time */
  int jdy;       /* Integral part of displacement */
  int j1, j2;    /* j values in real system to interpolate between */

  double ule[3]; /* +/- velocity jump at plane */

  assert(obj);


  if (cart_size(Y) > 1) {
    if (le_get_nxbuffer())
      hydro_lees_edwards_parallel(obj);
  }
  else {

    ule[X] = 0.0;
    ule[Y] = 0.0;  /* Only y component will be non-zero */
    ule[Z] = 0.0;

    nhalo = coords_nhalo();
    coords_nlocal(nlocal);
    ib0 = nlocal[X] + nhalo + 1;

    t = 1.0*get_step();

    for (ib = 0; ib < le_get_nxbuffer(); ib++) {

      ic = le_index_buffer_to_real(ib);
      dy = le_buffer_displacement(ib, t);

      /* This is a slightly awkward way to compute the velocity
       * jump: just the (+/-) displacement devided by time. */

      ule[Y] = dy/t; /* STEADY SHEAR ONLY */

      dy = fmod(dy, L(Y));
      jdy = floor(dy);
      fr  = dy - jdy;

      for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

	/* Actually required here is j1 = jc - jdy - 1, but there's
	 * horrible modular arithmetic for the periodic boundaries
	 * to ensure 1 <= j1,j2 <= nlocal[Y] */

	j1 = 1 + (jc - jdy - 2 + 2*nlocal[Y]) % nlocal[Y];
	j2 = 1 + j1 % nlocal[Y];

	/* If nhcomm < nhalo, we could use nhcomm here in the kc loop.
	 * (As j1 and j2 are always in the domain proper, jc can use nhalo.) */

	/* SHIT This kc loop was dubious +/- nhalo wrong */
	for (kc = 1 - obj->nhcomm; kc <= nlocal[Z] + obj->nhcomm; kc++) {
	  index0 = le_site_index(ib0 + ib, jc, kc);
	  index1 = le_site_index(ic, j1, kc);
	  index2 = le_site_index(ic, j2, kc);
	  for (ia = 0; ia < 3; ia++) {
	    obj->u[addr_hydro(index0, ia)] = ule[ia] +
	      obj->u[addr_hydro(index1, ia)]*fr +
	      obj->u[addr_hydro(index2, ia)]*(1.0 - fr);
	  }
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  hydro_lees_edwards_parallel
 *
 *  The Lees Edwards transformation for the velocity field in parallel.
 *  This is a linear interpolation.
 *
 *  Note that we communicate with up to 3 processors in each direction;
 *  this avoids having to update the halos completely.
 *
 *****************************************************************************/

static int hydro_lees_edwards_parallel(hydro_t * obj) {

  int      nlocal[3];      /* Local system size */
  int      noffset[3];     /* Local starting offset */
  int ib;                  /* Index in buffer region */
  int ib0;                 /* buffer region offset */
  int ic;                  /* Index corresponding x location in real system */
  int jc, kc, j1, j2;
  int n1, n2, n3;
  double dy;               /* Displacement for current ic->ib pair */
  double fr;               /* Fractional displacement */
  double t;                /* time */
  int jdy;                 /* Integral part of displacement */
  int index, ia;
  int nhalo;
  double ule[3];

  int nsend;
  int nrecv;
  int      nrank_s[3];     /* send ranks */
  int      nrank_r[3];     /* recv ranks */
  const int tag0 = 1256;
  const int tag1 = 1257;
  const int tag2 = 1258;

  double * sbuf = NULL;   /* Send buffer */
  double * rbuf = NULL;   /* Interpolation buffer */

  MPI_Comm    le_comm;
  MPI_Request request[6];
  MPI_Status  status[3];

  assert(obj);

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  coords_nlocal_offset(noffset);
  ib0 = nlocal[X] + nhalo + 1;

  le_comm = le_communicator();

  /* Allocate the temporary buffer */

  nsend = NHDIM*nlocal[Y]*(nlocal[Z] + 2*nhalo);
  nrecv = NHDIM*(nlocal[Y] + 2*nhalo + 1)*(nlocal[Z] + 2*nhalo);

  sbuf = (double *) calloc(nsend, sizeof(double));
  rbuf = (double *) calloc(nrecv, sizeof(double));
 
  if (sbuf == NULL) fatal("hydrodynamics: malloc(le sbuf) failed\n");
  if (rbuf == NULL) fatal("hydrodynamics: malloc(le rbuf) failed\n");

  t = 1.0*get_step();

  ule[X] = 0.0;
  ule[Z] = 0.0;

  /* One round of communication for each buffer plane */

  for (ib = 0; ib < le_get_nxbuffer(); ib++) {

    ic = le_index_buffer_to_real(ib);

    /* Work out the displacement-dependent quantities */

    dy = le_buffer_displacement(ib, t);
    ule[Y] = dy/t; /* STEADY SHEAR ONLY */
    dy = fmod(dy, L(Y));
    jdy = floor(dy);
    fr  = dy - jdy;

    /* First j1 required is j1 = jc - jdy - 1 with jc = 1 - nhalo.
     * Modular arithmetic ensures 1 <= j1 <= N_total(Y). */

    jc = noffset[Y] + 1 - nhalo;
    j1 = 1 + (jc - jdy - 2 + 2*N_total(Y)) % N_total(Y);

    le_jstart_to_ranks(j1, nrank_s, nrank_r);

    /* Local quantities: given a local starting index j2, we receive
     * n1 + n2 sites into the buffer, and send n1 sites starting with
     * j2, and the remaining n2 sites from starting position nhalo. */

    j2 = 1 + (j1 - 1) % nlocal[Y];

    n1 = (nlocal[Y] - j2 + 1)*(nlocal[Z] + 2*nhalo);
    n2 = imin(nlocal[Y], j2 + 2*nhalo)*(nlocal[Z] + 2*nhalo);
    n3 = imax(0, j2 - nlocal[Y] + 2*nhalo)*(nlocal[Z] + 2*nhalo);

    assert((n1+n2+n3) == (nlocal[Y] + 2*nhalo + 1)*(nlocal[Z] + 2*nhalo));

    /* Post receives, sends and wait for receives. */

    MPI_Irecv(rbuf, NHDIM*n1, MPI_DOUBLE, nrank_r[0], tag0, le_comm, request);
    MPI_Irecv(rbuf + NHDIM*n1, NHDIM*n2, MPI_DOUBLE, nrank_r[1], tag1,
	      le_comm, request + 1);
    MPI_Irecv(rbuf + NHDIM*(n1 + n2), NHDIM*n3, MPI_DOUBLE, nrank_r[2], tag2,
	      le_comm, request + 2);

    /* Load send buffer */

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = le_site_index(ic, jc, kc);
	for (ia = 0; ia < NHDIM; ia++) {
	  j1 = (jc - 1)*NHDIM*(nlocal[Z] + 2*nhalo) + NHDIM*(kc + nhalo - 1) + ia;
	  assert(j1 >= 0 && j1 < nsend);
	  sbuf[j1] = obj->u[addr_hydro(index, ia)];
	}
      }
    }

    j1 = (j2 - 1)*NHDIM*(nlocal[Z] + 2*nhalo);
    MPI_Issend(sbuf + j1, NHDIM*n1, MPI_DOUBLE, nrank_s[0], tag0,
	       le_comm, request + 3);
    MPI_Issend(sbuf     , NHDIM*n2, MPI_DOUBLE, nrank_s[1], tag1,
	       le_comm, request + 4);
    MPI_Issend(sbuf     , NHDIM*n3, MPI_DOUBLE, nrank_s[2], tag2,
	       le_comm, request + 5);

    MPI_Waitall(3, request, status);

    /* Perform the actual interpolation from temporary buffer to
     * buffer region. */

    for (jc = 1 - nhalo; jc <= nlocal[Y] + nhalo; jc++) {

      j1 = (jc + nhalo - 1    )*(nlocal[Z] + 2*nhalo);
      j2 = (jc + nhalo - 1 + 1)*(nlocal[Z] + 2*nhalo);

      for (kc = 1 - nhalo; kc <= nlocal[Z] + nhalo; kc++) {
	index = le_site_index(ib0 + ib, jc, kc);
	for (ia = 0; ia < NHDIM; ia++) {
	  obj->u[addr_hydro(index, ia)] = ule[ia]
	    + fr*rbuf[NHDIM*(j1 + kc + nhalo - 1) + ia]
	    + (1.0 - fr)*rbuf[NHDIM*(j2 + kc + nhalo - 1) + ia];
	}
      }
    }

    MPI_Waitall(3, request + 3, status);
  }

  free(sbuf);
  free(rbuf);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_write
 *
 *****************************************************************************/

static int hydro_u_write(FILE * fp, int index, void * arg) {

  int n;
  double u[3];
  hydro_t * obj = (hydro_t*) arg;

  assert(fp);
  assert(obj);

  hydro_u(obj, index, u);
  n = fwrite(u, sizeof(double), NHDIM, fp);
  if (n != NHDIM) fatal("fwrite(hydro->u) failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_write_ascii
 *
 *****************************************************************************/

static int hydro_u_write_ascii(FILE * fp, int index, void * arg) {

  int n;
  double u[3];
  hydro_t * obj = (hydro_t *) arg;

  assert(fp);
  assert(obj);

  hydro_u(obj, index, u);

  n = fprintf(fp, "%22.15e %22.15e %22.15e\n", u[X], u[Y], u[Z]);

  /* Expect total of 69 characters ... */
  if (n != 69) fatal("fprintf(hydro->u) failed\n");

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_read
 *
 *****************************************************************************/

int hydro_u_read(FILE * fp, int index, void * self) {

  int n;
  double u[3];
  hydro_t * obj = (hydro_t *) self;

  assert(fp);
  assert(obj);

  n = fread(u, sizeof(double), NHDIM, fp);
  if (n != NHDIM) fatal("fread(hydro->u) failed\n");

  hydro_u_set(obj, index, u);

  return 0;
}

/*****************************************************************************
 *
 *  hydro_u_gradient_tensor
 *
 *  Return the velocity gradient tensor w_ab = d_b u_a at
 *  the site (ic, jc, kc).
 *
 *  The differencing is 2nd order centred.
 *
 *  This must take account of the Lees Edwards planes in  the x-direction.
 *
 *****************************************************************************/

int hydro_u_gradient_tensor(hydro_t * obj, int ic, int jc, int kc,
			    double w[3][3]) {

  int im1, ip1;
  double tr;

  assert(obj);

  im1 = le_index_real_to_buffer(ic, -1);
  im1 = le_site_index(im1, jc, kc);
  ip1 = le_index_real_to_buffer(ic, +1);
  ip1 = le_site_index(ip1, jc, kc);

  w[X][X] = 0.5*(obj->u[addr_hydro(ip1, X)] - obj->u[addr_hydro(im1, X)]);
  w[Y][X] = 0.5*(obj->u[addr_hydro(ip1, Y)] - obj->u[addr_hydro(im1, Y)]);
  w[Z][X] = 0.5*(obj->u[addr_hydro(ip1, Z)] - obj->u[addr_hydro(im1, Z)]);

  im1 = le_site_index(ic, jc - 1, kc);
  ip1 = le_site_index(ic, jc + 1, kc);

  w[X][Y] = 0.5*(obj->u[addr_hydro(ip1, X)] - obj->u[addr_hydro(im1, X)]);
  w[Y][Y] = 0.5*(obj->u[addr_hydro(ip1, Y)] - obj->u[addr_hydro(im1, Y)]);
  w[Z][Y] = 0.5*(obj->u[addr_hydro(ip1, Z)] - obj->u[addr_hydro(im1, Z)]);

  im1 = le_site_index(ic, jc, kc - 1);
  ip1 = le_site_index(ic, jc, kc + 1);

  w[X][Z] = 0.5*(obj->u[addr_hydro(ip1, X)] - obj->u[addr_hydro(im1, X)]);
  w[Y][Z] = 0.5*(obj->u[addr_hydro(ip1, Y)] - obj->u[addr_hydro(im1, Y)]);
  w[Z][Z] = 0.5*(obj->u[addr_hydro(ip1, Z)] - obj->u[addr_hydro(im1, Z)]);

  /* Enforce tracelessness */

  tr = r3_*(w[X][X] + w[Y][Y] + w[Z][Z]);
  w[X][X] -= tr;
  w[Y][Y] -= tr;
  w[Z][Z] -= tr;

  return 0;
}

/*****************************************************************************
 *
 *  hydro_correct_momentum
 *
 *****************************************************************************/

int hydro_correct_momentum(hydro_t * hydro) {

  int ic, jc, kc, index;
  int nlocal[3];

  double f[3];
  double flocal[3] = {0.0, 0.0, 0.0};
  double fsum[3];
  double rv; 
  
  MPI_Comm comm;

  if (hydro == NULL) return 0;

  coords_nlocal(nlocal);
  comm = cart_comm();

  /* Compute force without correction. */
  
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {
    
        index = coords_index(ic, jc, kc);
        hydro_f_local(hydro, index, f); 

        flocal[X] += f[X];
        flocal[Y] += f[Y];
        flocal[Z] += f[Z];
    
      }   
    }   
  }
    
  /* calculate the total force per fluid node */

  MPI_Allreduce(flocal, fsum, 4, MPI_DOUBLE, MPI_SUM, comm);

  rv = 1.0/(L(X)*L(Y)*L(Z));
  f[X] = -fsum[X]*rv;
  f[Y] = -fsum[Y]*rv;
  f[Z] = -fsum[Z]*rv;

  /* Now add correction */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = coords_index(ic, jc, kc);
        hydro_f_local_add(hydro, index, f); 

      }   
    }   
  }
  
  return 0;
}
