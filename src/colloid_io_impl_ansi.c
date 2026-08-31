/*****************************************************************************
 *
 *  colloid_io_impl_ansi.c
 *
 *  An implementation of the oiriginal "ansi" approach for colloid input
 *  and output to one file (only). The data order in the resulting single
 *  file is decomposition dependent and the call serialises at root.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025-2026 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "colloid_io_impl_ansi.h"
#include "colloid_array_util.h"
#include "util_fopen.h"

/* Function table */
static colloid_io_impl_vt_t vt_ = {
    (colloid_io_impl_free_ft)  colloid_io_ansi_free,
    (colloid_io_impl_read_ft)  colloid_io_ansi_read,
    (colloid_io_impl_write_ft) colloid_io_ansi_write};

static int colloid_io_ansi_check_read(colloid_io_ansi_t * io, int ngroup);
static int colloid_io_ansi_errno_to_mpierr(int ierr);

int colloid_io_ansi_write_ascii(FILE * fp, const colloid_array_t * buf);
int colloid_io_ansi_write_binary(FILE * fp, const colloid_array_t * buf);

int colloid_io_ansi_read_ascii(colloid_io_ansi_t * io, FILE * fp);
int colloid_io_ansi_read_binary(colloid_io_ansi_t * io, FILE * fp);

/*****************************************************************************
 *
 *  colloid_io_ansi_create
 *
 *****************************************************************************/

int colloid_io_ansi_create(const colloids_info_t * info,
                           colloid_io_ansi_t **    io) {

  int                 ifail   = 0;
  colloid_io_ansi_t * io_inst = NULL;

  io_inst = (colloid_io_ansi_t *) calloc(1, sizeof(colloid_io_ansi_t));
  if (io_inst == NULL) goto err;

  ifail = colloid_io_ansi_initialise(info, io_inst);
  if (ifail != 0) goto err;

  *io = io_inst;

  return 0;

err:
  free(io_inst);

  return -1;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_free
 *
 *****************************************************************************/

void colloid_io_ansi_free(colloid_io_ansi_t ** io) {

  assert(io);

  if (*io) {
    colloid_io_ansi_finalise(*io);
    free(*io);
    *io = NULL;
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_initialise
 *
 *****************************************************************************/

int colloid_io_ansi_initialise(const colloids_info_t * info,
                               colloid_io_ansi_t *     io) {
  assert(info);
  assert(io);

  *io = (colloid_io_ansi_t) {0};

  io->super.impl = &vt_;
  io->info       = (colloids_info_t *) info;

  /* In principle, we could have multiple files for input and/or output
   * but here we demand a single file for both input and output. */
  /* The general case would only be required for backwards compatibility
   * if the original implementation were removed. */

  {
    int iogrid[3] = {1, 1, 1};
    int ifail     = io_subfile_create(io->info->cs, iogrid, &io->subfile);
    if (ifail != 0) goto err;
  }

  {
    int      rank   = -1;
    MPI_Comm parent = io->info->cs->commcart;
    MPI_Comm_rank(parent, &rank);
    MPI_Comm_split(parent, io->subfile.index, rank, &io->comm);
  }

  return 0;

err:
  *io = (colloid_io_ansi_t) {0};

  return -1;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_finalise
 *
 *****************************************************************************/

int colloid_io_ansi_finalise(colloid_io_ansi_t * io) {

  assert(io);

  MPI_Comm_free(&io->comm);

  *io      = (colloid_io_ansi_t) {0};
  io->comm = MPI_COMM_NULL;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_write
 *
 *  Write information on the colloids in the local domain proper (not
 *  including the halos) to the specified file.
 *
 *  In parallel, one file per io group is used. Processes within a
 *  group aggregate data to root, and root writes. So this does not
 *  scale very well.
 *
 *****************************************************************************/

int colloid_io_ansi_write(colloid_io_ansi_t * io, const char * filename) {

  int   ifail  = 0;
  int   myrank = -1;
  int   nranks = -1;
  int   nlocal = 0;
  int   ntotal = 0;
  int   nalloc = 1;
  int * nclist = NULL;
  int * displ  = NULL;

  /* consolidated buffers for particle data */
  colloid_array_t cbuf = {0};
  colloid_array_t rbuf = {0};

  assert(io);

  colloids_info_ntotal(io->info, &ntotal);
  if (ntotal == 0) return 0; /* All ranks must agree */

  MPI_Comm_rank(io->comm, &myrank);
  MPI_Comm_size(io->comm, &nranks);

  /* Gather a list of colloid numbers from each rank in the group */

  nclist = (int *) calloc(nranks, sizeof(int));
  assert(nclist);
  if (nclist == NULL) goto err;

  colloids_info_nlocal(io->info, &nlocal);

  /* We must allow that a given rank has no colloids ... */
  /* ... but no zero-sized allocations please. */
  assert(nlocal >= 0);
  if (nlocal > 0) nalloc = nlocal;

  MPI_Gather(&nlocal, 1, MPI_INT, nclist, 1, MPI_INT, 0, io->comm);

  /* Allocate local buffer, pack. */

  ifail = colloid_array_alloc(0, nalloc, &cbuf);
  if (ifail != 0) goto err;

  {
    colloid_t * pc = NULL;
    colloids_info_list_local_build(io->info);
    colloids_info_local_head(io->info, &pc);

    for (int n = 0; pc; pc = pc->nextlocal, n += 1) {
      memcpy(&cbuf.data[n], &pc->s, sizeof(colloid_state_t));
    }
  }

  if (myrank == 0) {

    /* Work out displacements and the total */
    /* Allocate total receive buffer */

    displ = (int *) malloc(nranks*sizeof(int));
    assert(displ);
    if (displ == NULL) goto err;

    displ[0] = 0;
    for (int n = 1; n < nranks; n++) {
      displ[n] = displ[n - 1] + nclist[n - 1];
    }

    ntotal = 0;
    for (int n = 0; n < nranks; n++) {
      ntotal    += nclist[n];
      displ[n]  *= sizeof(colloid_state_t);  /* to bytes */
      nclist[n] *= sizeof(colloid_state_t);  /* ditto */
    }

    ifail = colloid_array_alloc(0, ntotal, &rbuf);
    if (ifail != 0) goto err;
  }

  MPI_Gatherv(cbuf.data, nlocal * sizeof(colloid_state_t), MPI_BYTE, rbuf.data,
              nclist, displ, MPI_BYTE, 0, io->comm);

  if (myrank == 0) {
    int asc = (io->info->options.output.iorformat == IO_RECORD_ASCII);
    int bin = (io->info->options.output.iorformat == IO_RECORD_BINARY);

    FILE * fp = util_fopen(filename, "w");

    if (fp == NULL) ifail = errno;
    if (ifail != 0) goto err;

    if (asc) ifail = colloid_io_ansi_write_ascii(fp, &rbuf);
    if (bin) ifail = colloid_io_ansi_write_binary(fp, &rbuf);

    /* Keep ifail from write function, and assume fclose() is ok... */
    fclose(fp);
  }

err:

  colloid_array_free(&rbuf);
  colloid_array_free(&cbuf);

  free(displ);
  free(nclist);

  ifail = colloid_io_ansi_errno_to_mpierr(ifail);

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_write_ascii
 *
 *  Write nc states to file. Return number of bytes written.
 *
 *****************************************************************************/

int colloid_io_ansi_write_ascii(FILE * fp, const colloid_array_t * buf) {

  int ifail = 0;

  assert(fp);
  assert(buf);

  fprintf(fp, "%22d\n", buf->ntotal); /* HEADER */

  for (int n = 0; n < buf->ntotal; n++) {
    ifail += colloid_state_write_ascii(&buf->data[n], fp);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_write_binary
 *
 *  Write nc states to file. One could try to have some error information
 *  related to the header/data, but it does not fit easily in the MPI
 *  error return value.
 *
 *****************************************************************************/

int colloid_io_ansi_write_binary(FILE * fp, const colloid_array_t * buf) {

  int ifail = MPI_SUCCESS;
  size_t nw = 0; /* No. of items written */

  assert(fp);
  assert(buf);

  /* Header */
  nw = fwrite(&buf->ntotal, sizeof(int), 1, fp);
  if (nw != 1) goto err;

  /* Data */
  nw = fwrite(buf->data, sizeof(colloid_state_t), buf->ntotal, fp);
  if (nw != (size_t) buf->ntotal) goto err;

  return ifail;

err:
  ifail = errno;
  ifail = colloid_io_ansi_errno_to_mpierr(ifail);
  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_read
 *
 *  This is the driver routine to read colloid information from file.
 *
 *  The read is a free-for-all in which all processes in the group
 *  read the entire file. Each colloid is then added or discarded
 *  based on its position.
 *
 *****************************************************************************/

int colloid_io_ansi_read(colloid_io_ansi_t * io, const char * filename) {

  assert(io);
  assert(filename);

  int    ifail = 0;
  FILE * fp    = util_fopen(filename, "r");

  if (fp == NULL) ifail = errno;
  if (ifail != 0) goto err;

  {
    int asc = (io->info->options.input.iorformat == IO_RECORD_ASCII);
    int bin = (io->info->options.input.iorformat == IO_RECORD_BINARY);

    if (asc) ifail = colloid_io_ansi_read_ascii(io, fp);
    if (bin) ifail = colloid_io_ansi_read_binary(io, fp);

    fclose(fp);
    if (ifail != 0) goto err; /* keep ifail from read function */
  }

  /* Check data for old-style "type" entries */
  /* This is scheduled for removal */
  {
    /* Check for old 'type' component */
    int    ntype         = 0;
    int    ntype_updates = colloids_type_check(io->info);
    pe_t * pe            = io->info->pe;
    MPI_Allreduce(&ntype_updates, &ntype, 1, MPI_INT, MPI_SUM, io->comm);
    if (ntype > 0) {
      pe_info(pe, "One or more colloids were updated as this looks\n");
      pe_info(pe, "like an old format input file. See note at\n");
      pe_info(pe, "https://ludwig.epcc.ed.ac.uk/outputs/colloid.html\n");
    }
    {
      /* Any ellipsoids must be checked. */
      int nbad = colloids_ellipsoid_abc_check(io->info);
      if (nbad > 0) {
        pe_exit(pe, "One or more ellipses in input file fail checks\n");
      }
    }
  }

err:
  ifail = colloid_io_ansi_errno_to_mpierr(ifail);

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_read_ascii
 *
 *****************************************************************************/

int colloid_io_ansi_read_ascii(colloid_io_ansi_t * io, FILE * fp) {

  int ifail = 0;
  int nread = 0; /* Header; number of colloids in this file */

  assert(io);
  assert(fp);

  int nr = fscanf(fp, "%22d\n", &nread);
  if (nr != 1) goto err;

  for (int n = 0; n < nread; n++) {
    colloid_state_t s     = {0};
    colloid_t *     pc    = NULL;

    ifail = colloid_state_read_ascii(&s, fp);
    if (ifail != 0) goto err;

    colloids_info_add_local(io->info, &s, &pc);
  }

err:

  ifail = colloid_io_ansi_check_read(io, nread);

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_read_binary
 *
 *****************************************************************************/

int colloid_io_ansi_read_binary(colloid_io_ansi_t * io, FILE * fp) {

  int ifail = 0;
  int nread = 0; /* Header: number of colloid entries in this file. */

  assert(io);
  assert(fp);

  size_t nr = fread(&nread, sizeof(int), 1, fp); /* HEADER */
  if (nr != 1) goto err;

  for (int n = 0; n < nread; n++) {
    colloid_state_t s     = {0};
    colloid_t *     pc    = NULL;

    ifail = colloid_state_read_binary(&s, fp);
    if (ifail != 0) goto err;

    colloids_info_add_local(io->info, &s, &pc);
  }

  /* Collective, so cannot be by-passed. Fall through... */

err:

  ifail = colloid_io_ansi_check_read(io, nread);

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_check_read
 *
 *  Check the number of colloids in the list is consistent with that
 *  in the file (ngroup), and set the total.
 *
 *  If we haven't lost any particles, we can set the global total and
 *  proceed.
 *
 *  Returns MPI_SUCCESS or MPI_ERR_IO if not correct.
 *
 *****************************************************************************/

static int colloid_io_ansi_check_read(colloid_io_ansi_t * io, int ngroup) {

  int ifail = MPI_SUCCESS;

  int nlocal = 0;
  int ntotal = 0;

  assert(io);

  colloids_info_nlocal(io->info, &nlocal);

  MPI_Allreduce(&nlocal, &ntotal, 1, MPI_INT, MPI_SUM, io->comm);
  if (ntotal != ngroup) ifail = MPI_ERR_IO;

  if (ifail == MPI_SUCCESS) {
    colloids_info_ntotal_set(io->info);
    colloids_info_ntotal(io->info, &ntotal);
    pe_info(io->info->pe, "Read a total of %d colloids from file\n", ntotal);
  }
  else {
    pe_t * pe = io->info->pe;
    pe_info(pe, "Error reading colloid file (group %d)\n", io->subfile.index);
    pe_info(pe, "Header says: %d colloids expected.\n", ngroup);
    pe_info(pe, "Read and initialised in cell list only: %d\n", ntotal);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_ansi_errno_to_mpierr
 *
 *  Translate a standard errno.h errno to MPI_ERR value.
 *  This allows returning a uniform failure code.
 *
 *****************************************************************************/

static int colloid_io_ansi_errno_to_mpierr(int ierr) {

  int ifail = MPI_SUCCESS;

  assert(MPI_SUCCESS == 0); /* We agree on what is success */

  if (ierr == ENOENT) {
    ifail = MPI_ERR_NO_SUCH_FILE;
  }
  else if (ierr != 0) {
    ifail = MPI_ERR_IO;
  }

  return ifail;
}
