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
 *  The halo for the RNG is 1 point in the case that fluxes are required.
 *  This will will need to be handled if restarting from a saved file.
 *  It is not at present.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2013-2025 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "noise.h"

static const int nchar_per_ascii_int32_ = NOISE_RECORD_LENGTH_ASCII;
static int noise_initialise_state(noise_t * ns);

/*****************************************************************************
 *
 *  noise_create
 *
 *****************************************************************************/

int noise_create(pe_t * pe, cs_t * cs, const noise_options_t * options,
		 noise_t ** ns) {

  noise_t * obj = NULL;

  assert(pe);
  assert(cs);
  assert(options);
  assert(ns);

  obj = (noise_t *) calloc(1, sizeof(noise_t));
  if (obj == NULL) goto err;
  if (noise_initialise(pe, cs, options, obj) != 0) goto err;

  *ns = obj;

  return 0;

 err:
  free(obj);

  return -1;
}

/*****************************************************************************
 *
 *  noise_create_seed
 *
 *  Convenience for seed-only argument
 *
 *  If the input seed is negative, it may get converted to
 *  an unsigned int which is positive. This may not be what
 *  one expects. So check.
 *
 *****************************************************************************/

int noise_create_seed(pe_t * pe, cs_t * cs, int seed, noise_t ** ns) {

  noise_options_t opts = noise_options_default();

  if (seed > 1) opts = noise_options_seed(seed);

  return noise_create(pe, cs, &opts, ns);
}

/*****************************************************************************
 *
 *  noise_free
 *
 *****************************************************************************/

int noise_free(noise_t ** ns) {

  assert(ns);

  noise_finalise(*ns);
  free(*ns);
  *ns = NULL;

  return 0;
}

/*****************************************************************************
 *
 *  noise_initialise
 *
 *****************************************************************************/

int noise_initialise(pe_t * pe, cs_t * cs, const noise_options_t * options,
		     noise_t * ns) {

  int ndevice = 0;
  int nstate = 0;

  assert(pe);
  assert(cs);
  assert(options);
  assert(ns);

  ns->pe = pe;
  ns->cs = cs;

  cs_nsites(cs, &ns->nsites);
  assert(ns->nsites > 0);

  /* Check validity of options here */
  if (noise_options_valid(options) == 0) {
    pe_warn(pe, "One or more noise options is invalid\n");
    goto err;
  }

  ns->options = *options;

  /* Allocate state */
  nstate = NNOISE_STATE*ns->nsites;

  ns->state = (unsigned int *) malloc(nstate*sizeof(unsigned int));
  assert(ns->state);
  if (ns->state == NULL) {
    pe_warn(pe, "noise_initialise: calloc(ns->state) failed\n");
    goto err;
  }

  /* Here are the tabulated discrete random values with unit variance */

  ns->rtable[0] = -sqrt(2.0 + sqrt(2.0));
  ns->rtable[1] = -sqrt(2.0 - sqrt(2.0));
  ns->rtable[2] = 0.0;
  ns->rtable[3] = 0.0;
  ns->rtable[4] = 0.0;
  ns->rtable[5] = 0.0;
  ns->rtable[6] = +sqrt(2.0 - sqrt(2.0));
  ns->rtable[7] = +sqrt(2.0 + sqrt(2.0));

  /* I/O details and metadata initialisation */
  {
    int ifail = 0;
    io_element_t element = {};

    io_element_t ascii = {
      .datatype = MPI_CHAR,
      .datasize = sizeof(char),
      .count    = 4*nchar_per_ascii_int32_ + 1,
      .endian   = io_endianness()
    };
    io_element_t binary = {
      .datatype = MPI_UNSIGNED,
      .datasize = sizeof(unsigned int),
      .count    = NNOISE_STATE,
      .endian   = io_endianness()
    };

    ns->ascii = ascii;
    ns->binary = binary;

    if (options->iodata.input.iorformat == IO_RECORD_ASCII) element = ascii;
    if (options->iodata.input.iorformat == IO_RECORD_BINARY) element = binary;

    ifail = io_metadata_initialise(cs, &options->iodata.input, &element,
				   &ns->input);
    if (ifail != 0) {
      pe_warn(pe, "Failed to initlaise noise input metadata\n");
      goto err;
    }

    if (options->iodata.output.iorformat == IO_RECORD_ASCII) element = ascii;
    if (options->iodata.output.iorformat == IO_RECORD_BINARY) element = binary;

    ifail = io_metadata_initialise(cs, &options->iodata.output, &element,
				   &ns->output);
    if (ifail != 0) {
      pe_warn(pe, "Failed to initialise noise output metadata\n");
      goto err;
    }
  }

  /* Device allocations */

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    ns->target = ns;
  }
  else {
    unsigned int * tmp = NULL;
    tdpAssert( tdpMalloc((void **) &ns->target, sizeof(noise_t)) );
    tdpAssert( tdpMemset(ns->target, 0, sizeof(noise_t)) );
    tdpAssert( tdpMemcpy(ns->target->rtable, ns->rtable, 8*sizeof(double),
			 tdpMemcpyHostToDevice) );
    tdpAssert( tdpMalloc((void **) &tmp, nstate*sizeof(unsigned int)) );
    tdpAssert( tdpMemcpy(&ns->target->state, &tmp, sizeof(unsigned int *),
			 tdpMemcpyHostToDevice) );
  }

  noise_initialise_state(ns);
  noise_memcpy(ns, tdpMemcpyHostToDevice);

  return 0;

 err:
  noise_finalise(ns);

  return -1;
}

/*****************************************************************************
 *
 *  noise_finalise
 *
 *****************************************************************************/

int noise_finalise(noise_t * ns) {

  int ndevice = 0;

  assert(ns);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice > 0) {
    unsigned int * state = NULL;
    tdpAssert( tdpMemcpy(&state, &ns->target->state, sizeof(unsigned int *),
			 tdpMemcpyDeviceToHost) );
    if (state) tdpAssert( tdpFree(state) );
    tdpAssert( tdpFree(ns->target) );
  }

  /* Release only if initialised as has MPI_Comm_free() */
  if (ns->output.cs) io_metadata_finalise(&ns->output);
  if (ns->input.cs)  io_metadata_finalise(&ns->input);

  free(ns->state);

  *ns = (noise_t) {};

  return 0;
}

/*****************************************************************************
 *
 *  noise_initialise_state
 *
 *  The default values are set as state0[]. The overall seed is introduced
 *  as state0[0].
 *
 *****************************************************************************/

static int noise_initialise_state(noise_t * ns) {

  int nextra = -1;
  int nlocal[3]  = {0};
  int ntotal[3]  = {0};
  int noffset[3] = {0};
  unsigned int state0[NNOISE_STATE] = {13, 12953, 712357, 22383979};

  assert(ns);
  assert(ns->state);
  assert(ns->options.seed > 0);

  /* This is historical: state0[0] is the single user-supplied seed */

  state0[0] = ns->options.seed;

  cs_ntotal(ns->cs, ntotal);
  cs_nlocal(ns->cs, nlocal);
  cs_nlocal_offset(ns->cs, noffset);

  nextra = ns->options.nextra;

  #pragma omp parallel for collapse(3)
  for (int ic = 1 - nextra; ic <= nlocal[X] + nextra; ic++) {
    for (int jc = 1 - nextra; jc <= nlocal[Y] + nextra; jc++) {
      for (int kc = 1 - nextra; kc <= nlocal[Z] + nextra; kc++) {

	int ig = noffset[X] + ic;
	int jg = noffset[Y] + jc;
        int kg = noffset[Z] + kc;

	unsigned int state_local[NNOISE_STATE] = {0};

	if (ig < 1) ig += ntotal[X];
	if (ig > ntotal[X]) ig -= ntotal[X];
	if (jg < 1) jg += ntotal[Y];
	if (jg > ntotal[Y]) jg -= ntotal[Y];
        if (kg < 1) kg += ntotal[Z];
        if (kg > ntotal[Z]) kg -= ntotal[Z];

	/* Set state */

	state_local[0] = state0[0] + ig;
	state_local[1] = state0[1] + jg;
	state_local[2] = state0[2] + kg;
	state_local[3] = state0[3];

	/* At this point, state_local[3] looks the same for all
	 * jc, kc etc. So run through generator once to produce
	 * unique seeds at each lattice point, which are the
	 * ones we use. */

	{
	  int index = cs_index(ns->cs, ic, jc, kc);
	  unsigned int state[NNOISE_STATE] = {0};
	  state[0] = ns_uniform(state_local);
	  state[1] = ns_uniform(state_local);
	  state[2] = ns_uniform(state_local);
	  state[3] = ns_uniform(state_local);

	  noise_state_set(ns, index, state);
	}
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_memcpy
 *
 *****************************************************************************/

int noise_memcpy(noise_t * ns, tdpMemcpyKind flag) {

  int ndevice = 0;

  assert(ns);

  tdpAssert( tdpGetDeviceCount(&ndevice) );

  if (ndevice == 0) {
    assert(ns->target == ns);
  }
  else {
    int nstat = NNOISE_STATE*ns->nsites;
    unsigned int * tmp = NULL;

    tdpAssert( tdpMemcpy(&tmp, &ns->target->state, sizeof(unsigned int *),
			 tdpMemcpyDeviceToHost) );

    switch (flag) {
    case tdpMemcpyHostToDevice:
      tdpAssert( tdpMemcpy(&ns->target->nsites, &ns->nsites, sizeof(int),
			   flag) );
      tdpAssert( tdpMemcpy(tmp, ns->state, nstat*sizeof(unsigned int), flag) );
      break;
    case tdpMemcpyDeviceToHost:
      tdpAssert( tdpMemcpy(ns->state, tmp, nstat*sizeof(unsigned int), flag) );
      break;
    default:
      pe_fatal(ns->pe, "Bad flag in noise_memcpy\n");
      break;
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_state_set
 *
 *****************************************************************************/

int noise_state_set(noise_t * ns, int index, const unsigned int newstate[4]) {

  assert(ns);
  assert(0 <= index && index < ns->nsites);
  assert(NNOISE_STATE == 4);

  for (int ia = 0; ia < NNOISE_STATE; ia++) {
    ns->state[addr_rank1(ns->nsites, NNOISE_STATE, index, ia)] = newstate[ia];
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

int noise_state(const noise_t * ns, int index, unsigned int state[4]) {

  assert(ns);
  assert(0 <= index && index < ns->nsites);
  assert(NNOISE_STATE == 4);

  for (int ia = 0; ia < NNOISE_STATE; ia++) {
    state[ia] = ns->state[addr_rank1(ns->nsites, NNOISE_STATE, index, ia)];
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_uniform_double_reap
 *
 *  Return a single double a full precision.
 *
 *****************************************************************************/

int noise_uniform_double_reap(noise_t * obj, int index, double * reap) {

  unsigned int iuniform;
  unsigned int state[NNOISE_STATE];

  assert(obj);
  assert(index >= 0);
  assert(index < obj->nsites);

  noise_state(obj, index, state);
  iuniform = ns_uniform(state);
  noise_state_set(obj, index, state);

  *reap = (1.0/UINT_MAX)*iuniform;

  return 0;
}

/*****************************************************************************
 *
 *  noise_read_buf
 *
 *****************************************************************************/

int noise_read_buf(noise_t * ns, int index, const char * buf) {

  assert(ns);
  assert(buf);

  for (int is = 0; is < NNOISE_STATE; is++) {
    size_t ioff = is*sizeof(unsigned int);
    int iaddr = addr_rank1(ns->nsites, NNOISE_STATE, index, is);
    memcpy(ns->state + iaddr, buf + ioff, sizeof(unsigned int));
  }

  return 0;
}


/*****************************************************************************
 *
 *  noise_read_buf_ascii
 *
 *****************************************************************************/

int noise_read_buf_ascii(noise_t * ns, int index, const char * buf) {

  const int nchar = nchar_per_ascii_int32_;
  int ifail = 0;

  for (int is = 0; is < NNOISE_STATE; is++) {
    char tmp[BUFSIZ] = {0};
    int iaddr = addr_rank1(ns->nsites, NNOISE_STATE, index, is);
    memcpy(tmp, buf + is*nchar*sizeof(char), nchar*sizeof(char));
    if (1 != sscanf(tmp, "%10u", ns->state + iaddr)) ifail -= 1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  noise_write_buf
 *
 *****************************************************************************/

int noise_write_buf(const noise_t * ns, int index, char * buf) {

  const size_t nbyte = sizeof(unsigned int);

  assert(ns);
  assert(buf);

  for (int is = 0; is < NNOISE_STATE; is++) {
    int iaddr = addr_rank1(ns->nsites, NNOISE_STATE, index, is);
    memcpy(buf + is*nbyte, ns->state + iaddr, nbyte);
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_write_buf_ascii
 *
 *****************************************************************************/

int noise_write_buf_ascii(const noise_t * ns, int index, char * buf) {

  int ifail = 0;
  const int nchar = nchar_per_ascii_int32_;

  assert(ns);
  assert(buf);

  for (int is = 0; is < NNOISE_STATE; is++) {
    char tmp[BUFSIZ] = {0};
    int iaddr = addr_rank1(ns->nsites, NNOISE_STATE, index, is);
    int nr = snprintf(tmp, nchar + 1, " %10u", ns->state[iaddr]);
    if (nr != nchar) ifail = -1;
    memcpy(buf + is*nchar*sizeof(char), tmp, nchar*sizeof(char));
  }

  /* New line */
  {
    char tmp[BUFSIZ] = {0};
    int nr = snprintf(tmp, 2, "\n");
    if (nr != 1) ifail = -2;
    memcpy(buf + NNOISE_STATE*nchar*sizeof(char), tmp, sizeof(char));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  noise_io_aggr_pack
 *
 *****************************************************************************/

int noise_io_aggr_pack(const noise_t * ns, io_aggregator_t * aggr) {

  assert(ns);
  assert(aggr && aggr->buf);

  #pragma omp parallel
  {
    int iasc = (ns->options.iodata.output.iorformat == IO_RECORD_ASCII);
    int ibin = (ns->options.iodata.output.iorformat == IO_RECORD_BINARY);
    assert(iasc ^ ibin);

    #pragma omp parallel for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Write state for (ic, jc, kc) */
      int index = cs_index(ns->cs, ic, jc, kc);
      size_t offset = ib*aggr->szelement;
      if (iasc) noise_write_buf_ascii(ns, index, aggr->buf + offset);
      if (ibin) noise_write_buf(ns, index, aggr->buf + offset);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_io_aggr_unpack
 *
 *****************************************************************************/

int noise_io_aggr_unpack(noise_t * ns, const io_aggregator_t * aggr) {

  assert(ns);
  assert(aggr && aggr->buf);

  #pragma omp parallel
  {
    int iasc = (ns->options.iodata.input.iorformat == IO_RECORD_ASCII);
    int ibin = (ns->options.iodata.input.iorformat == IO_RECORD_BINARY);
    assert(iasc ^ ibin);

    #pragma omp parallel for
    for (int ib = 0; ib < cs_limits_size(aggr->lim); ib++) {
      int ic = cs_limits_ic(aggr->lim, ib);
      int jc = cs_limits_jc(aggr->lim, ib);
      int kc = cs_limits_kc(aggr->lim, ib);

      /* Read state for (ic, jc, kc) */
      int index = cs_index(ns->cs, ic, jc, kc);
      size_t offset = ib*aggr->szelement;
      if (iasc) noise_read_buf_ascii(ns, index, aggr->buf + offset);
      if (ibin) noise_read_buf(ns, index, aggr->buf + offset);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  noise_io_write
 *
 *****************************************************************************/

int noise_io_write(noise_t * ns, int timestep, io_event_t * event) {

  int ifail = 0;

  assert(ns);

  {
    const io_metadata_t * meta = &ns->output;
    io_impl_t * io = NULL;
    char filename[BUFSIZ] = {0};

    io_subfile_name(&meta->subfile, ns->options.filestub, timestep, filename,
		    BUFSIZ);
    ifail = io_impl_create(meta, &io);

    if (ifail == 0) {
      int ierr = MPI_SUCCESS;
      io_event_record(event, IO_EVENT_AGGR);
      /* Include device->host copy if reelevant */
      noise_memcpy(ns, tdpMemcpyDeviceToHost);
      noise_io_aggr_pack(ns, io->aggr);

      io_event_record(event, IO_EVENT_WRITE);
      ierr = io->impl->write(io, filename);
      io->impl->free(&io);

      if (ierr != MPI_SUCCESS) {
	int len = 0;
	char msg[MPI_MAX_ERROR_STRING] = {0};
	MPI_Error_string(ierr, msg, &len);
	pe_info(ns->pe, "Error: could not write noise state %s\n", filename);
	pe_info(ns->pe, "Error: %s\n", msg);
	pe_exit(ns->pe, "Will not continue. Stopping\n");
      }

      if (meta->options.report) {
	pe_info(ns->pe, "Wrote noise state to file: %s\n", filename);
      }
      io_event_report_write(event, meta, ns->options.filestub);
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  noise_io_read
 *
 *****************************************************************************/

int noise_io_read(noise_t * ns, int timestep, io_event_t * event) {

  int ifail = 0;
  io_impl_t * io = NULL;
  char filename[BUFSIZ] ={0};
  const io_metadata_t * meta = &ns->output;

  io_subfile_name(&meta->subfile, ns->options.filestub, timestep, filename,
		  BUFSIZ);
  ifail = io_impl_create(meta, &io);

  if (ifail == 0) {
    int ierr = io->impl->read(io, filename);

    if (ierr != MPI_SUCCESS) {
      int len = 0;
      char msg[MPI_MAX_ERROR_STRING] = {0};
      MPI_Error_string(ierr, msg, &len);
      pe_info(ns->pe, "Error: could not read noise state %s\n", filename);
      pe_info(ns->pe, "Error: %s\n", msg);
      pe_exit(ns->pe, "Cannot not continue. Stopping\n");
    }

    io_event_record(event, IO_EVENT_AGGR);
    noise_io_aggr_unpack(ns, io->aggr);
    noise_memcpy(ns, tdpMemcpyHostToDevice);
    io->impl->free(&io);
    if (meta->options.report) pe_info(ns->pe, "Read %s\n", filename);
  }

  return ifail;
}
