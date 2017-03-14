/*****************************************************************************
 *
 *  noise.c
 *
 *  This is to generatate random numbers used for isothermal fluctuations
 *  (aka 'noise') in the density and for fluctuations in various order
 *  parameters.
 *
 *  The idea here is to make this decomposition-independent by allowing
 *  each lattice site to retain its own random number generator state.
 *
 *  The final Gaussian deviates use the discrete generator described
 *  by Ladd in Computer Physics Communications 180, 2140--2142 (2009).
 *
 *  The uniform random numbers are generated via an RNG proposed
 *  by Marsaglia (unpublished, 1999);  It is referred to as KISS99 in
 *  L'Ecuyer and Simard in ACM TOMS 33 Article 22 (2007). The state is
 *  4 --- 4-byte --- unsigned integers.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2017 Kevin Stratford
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "memory.h"
#include "noise.h"

/* The implementation is based on the following opaque object, which
 * holds the uniform random number generator state for all sites
 * (here 4*4 byte integer). It also holds a table of the discrete
 * values. */

struct noise_s {
  pe_t * pe;                /* Parallel environment */
  cs_t * cs;                /* Coordinate system */
  int master_seed;          /* Overall noise seed */
  int nsites;               /* Total number of lattice sites */
  int on[NOISE_END];        /* Noise on or off for different noise_enum_t */
  unsigned int * state;     /* Local state */
  double rtable[8];         /* Look up table following Ladd (2009). */
  io_info_t * info;
  noise_t * target;
};

static int noise_write(FILE * fp, int index, void * self);
static int noise_read(FILE * fp, int index, void * self);

/*****************************************************************************
 *
 *  noise_create
 *
 *  TODO: remove separate create/init in favour of create => use;
 *        separation can cause confusion.
 *        Note if cs is not initialised then cannot allocate state here. 
 *
 *****************************************************************************/

__host__ int noise_create(pe_t * pe, cs_t * cs, noise_t ** pobj) {

  int ndevice;
  noise_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(pobj);

  obj = (noise_t *) calloc(1, sizeof(noise_t));
  if (obj == NULL) pe_fatal(pe, "calloc(noise_t) failed\n");

  obj->pe = pe;
  obj->cs = cs;

  /* Here are the tabulated discrete random values with unit variance */

  obj->rtable[0] = -sqrt(2.0 + sqrt(2.0));
  obj->rtable[1] = -sqrt(2.0 - sqrt(2.0));
  obj->rtable[2] = 0.0;
  obj->rtable[3] = 0.0;
  obj->rtable[4] = 0.0;
  obj->rtable[5] = 0.0;
  obj->rtable[6] = +sqrt(2.0 - sqrt(2.0));
  obj->rtable[7] = +sqrt(2.0 + sqrt(2.0));

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    obj->target = obj;
  }
  else {
    tdpMalloc((void **) &obj->target, sizeof(noise_t));
    tdpMemset(obj->target, 0, sizeof(noise_t));
    tdpMemcpy(obj->target->rtable, obj->rtable, 8*sizeof(double),
	      tdpMemcpyHostToDevice);
  }

  *pobj = obj;

  return 0;
}

/*****************************************************************************
 *
 *  noise_free
 *
 *****************************************************************************/

__host__ int noise_free(noise_t * obj) {

  assert(obj);

  if (obj->target != obj && obj->state) {
    unsigned int * tmp = NULL;
    tdpMemcpy(&tmp, &obj->target->state, sizeof(unsigned int *),
	      tdpMemcpyDeviceToHost);
    tdpFree(obj->target);
  }

  if (obj->state) free(obj->state);
  free(obj);

  return 0;
}

/*****************************************************************************
 *
 *  noise_init
 *
 *  Decomposition independent initialisation.
 *
 *  The seed is based on a global (ig, jg, kg) position index
 *  for which we need to set only the local sites (plus halo).
 *
 *  The halo extends 1 point into the halo to allow mid-point
 *  random numbers to be computed. The halo points must have
 *  the appropriate initialisation based on global (ig, jg, kg).
 * 
 *****************************************************************************/

__host__ int noise_init(noise_t * obj, int master_seed) {

  int ic, jc, kc, index;
  int ig, jg, kg;
  int ndevice;
  int nextra;
  int nstat;
  int nlocal[3];
  int noffset[3];
  int ntotal[3];

  unsigned int state0[NNOISE_STATE] = {13, 12953, 712357, 22383979};
  unsigned int state[NNOISE_STATE];
  unsigned int state_local[NNOISE_STATE];

  assert(obj);

  /* Can take the full default if valid seed is not provided */

  if (master_seed > 0) {
    obj->master_seed = master_seed;
    state0[0] = master_seed;
  }

  cs_nsites(obj->cs, &obj->nsites);
  nstat = NNOISE_STATE*obj->nsites;
  assert(obj->nsites > 0);

  obj->state = (unsigned int *) calloc(nstat, sizeof(unsigned int));
  if (obj->state == NULL) pe_fatal(obj->pe, "calloc(obj->state) failed\n");

  nextra = 1;
  cs_ntotal(obj->cs, ntotal);
  cs_nlocal(obj->cs, nlocal);
  cs_nlocal_offset(obj->cs, noffset);

  for (ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {

    ig = noffset[X] + ic;
    if (ig < 1) ig += ntotal[X];
    if (ig > ntotal[X]) ig -= ntotal[X];

    for (jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {

      jg = noffset[Y] + jc;
      if (jg < 1) jg += ntotal[Y];
      if (jg > ntotal[Y]) jg -= ntotal[Y];

      for (kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

        kg = noffset[Z] + kc;
        if (kg < 1) kg += ntotal[Z];
        if (kg > ntotal[Z]) kg -= ntotal[Z];

	/* Set state */

	state_local[0] = state0[0] + ig;
	state_local[1] = state0[1] + jg;
	state_local[2] = state0[2] + kg;
	state_local[3] = state0[3];

	/* At this point, state_local[0] looks the same for all
	 * jc, kc etc. So run through generator once to produce
	 * unique seeds at each lattice point, which are the
	 * ones we use. */

	state[0] = noise_uniform(state_local);
	state[1] = noise_uniform(state_local);
	state[2] = noise_uniform(state_local);
	state[3] = noise_uniform(state_local);

	index = cs_index(obj->cs, ic, jc, kc);
	noise_state_set(obj, index, state);
      }
    }
  }

  tdpGetDeviceCount(&ndevice);

  if (ndevice > 0) {
    unsigned int * tmp = NULL;
    tdpMalloc((void **) &tmp, nstat*sizeof(unsigned int));
    tdpMemcpy(&obj->target->state, &tmp, sizeof(unsigned int *),
	      tdpMemcpyHostToDevice);
  }

  noise_memcpy(obj, tdpMemcpyHostToDevice);

  return 0;
}

/*****************************************************************************
 *
 *  noise_target
 *
 *****************************************************************************/

__host__ int noise_target(noise_t * obj, noise_t ** target) {

  assert(obj);
  assert(target);

  *target = obj->target;

  return 0;
}

/*****************************************************************************
 *
 *  noise_memcpy
 *
 *****************************************************************************/

__host__ int noise_memcpy(noise_t * obj, tdpMemcpyKind flag) {

  int ndevice;

  assert(obj);

  tdpGetDeviceCount(&ndevice);

  if (ndevice == 0) {
    assert(obj->target == obj);
  }
  else {
    int nstat;
    unsigned int * tmp = NULL;

    nstat = NNOISE_STATE*obj->nsites;
    tdpMemcpy(&tmp, &obj->target->state, sizeof(unsigned int *),
	      tdpMemcpyDeviceToHost);

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpMemcpy(&obj->target->nsites, &obj->nsites, sizeof(int), flag);
      tdpMemcpy(tmp, obj->state, nstat*sizeof(unsigned int), flag);
      break;
    case tdpMemcpyDeviceToHost:
      tdpMemcpy(obj->state, tmp, nstat*sizeof(unsigned int), flag);
      break;
    default:
      pe_fatal(obj->pe, "Bad flag in noise_memcpy\n");
      break;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_init_io_info
 *
 *  Note only binary I/O at the moment.
 *
 *****************************************************************************/

__host__ int noise_init_io_info(noise_t * obj, int grid[3], int form_in,
				int form_out) {

  const char * name = "Lattice noise RNG state";
  const char * stubname = "noise";
  io_info_arg_t args;

  assert(obj);

  args.grid[X] = grid[X];
  args.grid[Y] = grid[Y];
  args.grid[Z] = grid[Z];

  io_info_create(obj->pe, obj->cs, &args, &obj->info);
  if (obj->info == NULL) pe_fatal(obj->pe, "io_info_create(noise) failed\n");

  io_info_set_name(obj->info, name);
  io_info_write_set(obj->info, IO_FORMAT_BINARY, noise_write);
  io_info_write_set(obj->info, IO_FORMAT_ASCII, noise_write);
  io_info_read_set(obj->info, IO_FORMAT_BINARY, noise_read);
  io_info_read_set(obj->info, IO_FORMAT_ASCII, noise_read);
  io_info_set_bytesize(obj->info, NNOISE_STATE*sizeof(unsigned int));

  io_info_format_set(obj->info, form_in, form_out);
  io_info_metadata_filestub_set(obj->info, stubname);

  return 0;
}

/*****************************************************************************
 *
 *  noise_state_set
 *
 *****************************************************************************/

__host__ __device__ int noise_state_set(noise_t * obj, int index,
					unsigned int newstate[NNOISE_STATE]) {
  int ia;

  assert(obj);
  assert(index >= 0);
  assert(index < obj->nsites);

  for (ia = 0; ia < NNOISE_STATE; ia++) {
    obj->state[addr_rank1(obj->nsites,NNOISE_STATE,index,ia)] = newstate[ia];
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_state
 *
 *  Return the state of the generator at index.
 *
 *****************************************************************************/

__host__ __device__ int noise_state(noise_t * obj, int index,
				    unsigned int state[NNOISE_STATE]) {
  int ia;

  assert(obj);
  assert(index >= 0);
  assert(index < obj->nsites);

  for (ia = 0; ia < NNOISE_STATE; ia++) {
    state[ia] = obj->state[addr_rank1(obj->nsites, NNOISE_STATE, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_reap
 *
 *  Return NNOISE_MAX numbers via noise_reap_n().
 *
 *****************************************************************************/

__host__ __device__
int noise_reap(noise_t * obj, int index, double * reap) {

  assert(obj);
  assert(index >= 0);
  assert(index < obj->nsites);

  noise_reap_n(obj, index, NNOISE_MAX, reap);

  return 0;
}

/*****************************************************************************
 *
 *  noise_reap_n
 *
 *  Return nmax discrete random numbers for site index.
 *  These have mean zero and variance unity.
 *
 *****************************************************************************/

__host__ __device__
int noise_reap_n(noise_t * obj, int index, int nmax, double * reap) {

  unsigned int iuniform;
  unsigned int state[NNOISE_STATE];
  int ia;

  assert(obj);
  assert(index >= 0);
  assert(index < obj->nsites);
  assert(nmax <= NNOISE_MAX);

  noise_state(obj, index, state);
  iuniform = noise_uniform(state);
  noise_state_set(obj, index, state);

  /* Remove the leading two bits, and index the table using each of the
   * remaining three bits in turn. */

  iuniform >>= 2;

  for (ia = 0; ia < nmax; ia++) {
    reap[ia] = obj->rtable[iuniform & 7];
    iuniform >>= 3;
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_uniform_double_reap
 *
 *  Return a single double a full precission.
 *
 *****************************************************************************/

__host__ __device__
int noise_uniform_double_reap(noise_t * obj, int index, double * reap) {

  unsigned int iuniform;
  unsigned int state[NNOISE_STATE];

  assert(obj);
  assert(index >= 0);
  assert(index < obj->nsites);

  noise_state(obj, index, state);
  iuniform = noise_uniform(state);
  noise_state_set(obj, index, state);

  *reap = (1.0/UINT_MAX)*iuniform;

  return 0;
}

/*****************************************************************************
 *
 *  noise_uniform
 *
 *  Return a uniformly distributed integer following Marsaglia 1999.
 *  The range is 0 - 2^32 - 1.
 *
 *  This implementation is a direct rip-off of (ahem, `based upon') the
 *  implementation of L'Ecuyer and Simard found in the
 *  testu01 package (v1.2.2)
 *  http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
 *
 *****************************************************************************/

__host__ __device__
unsigned int noise_uniform(unsigned int state[NNOISE_STATE]) {

  unsigned int b;

  assert(NNOISE_STATE == 4);

  state[0] = 69069*state[0] + 1234567;
  b = state[1] ^ (state[1] << 17);
  b ^= (b >> 13);
  state[1] = b ^ (b << 5);
  state[2] = 36969*(state[2] & 0xffff) + (state[2] >> 16);
  state[3] = 18000*(state[3] & 0xffff) + (state[3] >> 16);
  b = (state[2] << 16) + state[3];

  return (state[1] + (state[0] ^ b));
}

/*****************************************************************************
 *
 *  noise_write
 *
 *  Implements io_rw_cb_ft for write.
 *
 *****************************************************************************/

static int noise_write(FILE * fp, int index, void * self) {

  noise_t * obj = (noise_t *) self;
  int n;
  unsigned int state[NNOISE_STATE];

  assert(fp);
  assert(obj);

  noise_state(obj, index, state);
  n = fwrite(state, sizeof(unsigned int), NNOISE_STATE, fp);
  if (n != NNOISE_STATE) pe_fatal(obj->pe, "fwrite(noise) index %d\n", index);

  return 0;
}

/*****************************************************************************
 *
 *  noise_read
 *
 *  Implements io_rw_cb_t for read.
 *
 *****************************************************************************/

static int noise_read(FILE * fp, int index, void * self) {

  noise_t * obj = (noise_t *) self;
  int n;
  unsigned int state[NNOISE_STATE];

  assert(obj);
  assert(fp);

  n = fread(state, sizeof(unsigned int), NNOISE_STATE, fp);
  if (n != NNOISE_STATE) pe_fatal(obj->pe, "fread(noise) index %d\n", index);
  noise_state_set(obj, index, state);

  return 0;
}

/*****************************************************************************
 *
 *  noise_present
 *
 *  Returns 0 or 1 (off / on) for given noise sector.
 *
 *****************************************************************************/

__host__ __device__
int noise_present(noise_t * noise, noise_enum_t type, int * present) {

  assert(noise);
  assert(type < NOISE_END);
  assert(present);

  *present = noise->on[type];

  return 0;
}

/*****************************************************************************
 *
 *  noise_present_set
 *
 *  Set status for given noise sector.
 *
 *****************************************************************************/

int noise_present_set(noise_t * noise, noise_enum_t type, int present) {

  assert(noise);
  assert(type < NOISE_END);

  noise->on[type] = present;

  return 0;
}
