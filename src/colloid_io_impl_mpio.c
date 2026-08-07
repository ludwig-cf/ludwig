/*****************************************************************************
 *
 *  colloid_io_impl_mpio.c
 *
 *  MPI/IO implementation for input and output to a single file, the
 *  ordering of which is decomposition independent.
 *
 *  There is a requirement that the indices of the colloids are
 *  {1, ..., ntotal} without gaps.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025-2026 The University of Edinvburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "colloid_state_io.h"
#include "colloid_io_impl_mpio.h"
#include "util_fopen.h"

/* Function table */
static colloid_io_impl_vt_t vt_ = {
    (colloid_io_impl_free_ft)  colloid_io_mpio_free,
    (colloid_io_impl_read_ft)  colloid_io_mpio_read,
    (colloid_io_impl_write_ft) colloid_io_mpio_write};

typedef struct sort_by_pointer_s {
  colloid_t * pc;
  int         index;
} sort_by_pointer_t;

int colloid_io_mpio_read_ascii(colloid_io_mpio_t * io, FILE * fp);
int colloid_io_mpio_read_binary(colloid_io_mpio_t * io, FILE * fp);

/*****************************************************************************
 *
 *  compare_by_pointer
 *
 *  Comparison function for array of sort_by_pointer_t for qsort() etc.
 *  The sort is on the global index.
 *
 *****************************************************************************/

int compare_by_pointer(const void * aarg, const void * barg) {

  sort_by_pointer_t * a        = (sort_by_pointer_t *) aarg;
  sort_by_pointer_t * b        = (sort_by_pointer_t *) barg;
  int                 icompare = 0;

  if (a->index > b->index) icompare = +1;
  if (a->index < b->index) icompare = -1;

  assert(icompare != 0); /* cannot be two identical colloid indices */

  return icompare;
}

/*****************************************************************************
 *
 * colloid_io_mpio_create
 *
 *****************************************************************************/

int colloid_io_mpio_create(const colloids_info_t * info,
                           colloid_io_mpio_t **    io) {

  int                 ifail   = 0;
  colloid_io_mpio_t * io_inst = NULL;

  io_inst = (colloid_io_mpio_t *) calloc(1, sizeof(colloid_io_mpio_t));
  if (io_inst == NULL) goto err;

  ifail = colloid_io_mpio_initialise(info, io_inst);
  if (ifail != 0) goto err;

  *io = io_inst;

  return 0;

err:
  free(io_inst);

  return -1;
}

/*****************************************************************************
 *
 *  colloid_io_mpio_free
 *
 *****************************************************************************/

void colloid_io_mpio_free(colloid_io_mpio_t ** io) {

  assert(io);

  if (*io) {
    colloid_io_mpio_finalise(*io);
    free(*io);
    *io = NULL;
  }

  return;
}

/*****************************************************************************
 *
 *  colloid_io_mpio_initialise
 *
 *****************************************************************************/

int colloid_io_mpio_initialise(const colloids_info_t * info,
                               colloid_io_mpio_t *     io) {
  assert(info);
  assert(io);

  *io = (colloid_io_mpio_t) {0};

  io->super.impl = &vt_;
  io->info       = (colloids_info_t *) info;

  /* We will always use one file at the moment. */
  /* There is no evidence that more than one file is required in the
   * MPI/IO picture. */

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
  *io = (colloid_io_mpio_t) {0};

  return -1;
}

/*****************************************************************************
 *
 *  colloid_io_mpio_finalise
 *
 *****************************************************************************/

int colloid_io_mpio_finalise(colloid_io_mpio_t * io) {

  assert(io);

  MPI_Comm_free(&io->comm);

  *io      = (colloid_io_mpio_t) {0};
  io->comm = MPI_COMM_NULL;

  return 0;
}

/*****************************************************************************
 *
 *  colloid_io_mpio_read
 *
 *  Read is a case of reading all colloids in the file, and adding
 *  locally only those with relevant positions.
 *
 *  A standard C fread per process. One could try MPI_File_read().
 *
 *****************************************************************************/

int colloid_io_mpio_read(colloid_io_mpio_t * io, const char * filename) {

  int ifail = 0;

  assert(io);
  assert(filename);

  FILE * fp = util_fopen(filename, "r");

  if (fp == NULL) ifail = errno;
  if (ifail != 0) goto err;

  {
    int asc = (io->info->options.input.iorformat == IO_RECORD_ASCII);
    int bin = (io->info->options.input.iorformat == IO_RECORD_BINARY);

    if (asc) ifail = colloid_io_mpio_read_ascii(io, fp);
    if (bin) ifail = colloid_io_mpio_read_binary(io, fp);

    fclose(fp);
  }

err:
  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_mpio_write
 *
 *  The write will generate a file with colloids sorted into unique id
 *  order. This is used to provide the file offset for MPI/IO.
 *
 *  1. Form an array of size nlocal to hold colloid ids in sorted
 *     order.
 *  2. Write a contiguous buffer of colloid data in the sorted order.
 *     The buffer is of type MPI_CHAR and size per element depends on
 *     the ascii or binary option.
 *  3. Form an MPI_Type_indexed_block() with the sorted ids to describe
 *     the file 'filetype'
 *  4. set view MPI_File_set_view(fh, disp, etype, filetpye, datarep, info)
 *  5. MPI_File_write_all(fh, ...)
 *
 *****************************************************************************/

int colloid_io_mpio_write(colloid_io_mpio_t * io, const char * filename) {

  int ifail = 0;

  /* 1. Form the sorted list of colloid ids (local). */

  /* For example, locally we might have colloids with unique ids
   * 10, 4, and 12 in that order (from local pointer list) . So
   * the local indices would be [0, 1, 2]. We would need to write
   * these local indices in the order [1, 0, 2], ie., ids [4, 10, 12]. */

  /* The corresponding global displacements are the (index - 1) in
   * the range [0, ntotal - 1]. */

  int                 nlocal = 0;
  int                 ntotal = 0;
  size_t              nbufsz = 0;
  int *               idisp  = NULL;
  char *              buf    = NULL;
  colloid_t *         pc     = NULL;
  sort_by_pointer_t * sp     = NULL;

  assert(io);
  assert(filename);

  int asc = (io->info->options.output.iorformat == IO_RECORD_ASCII);
  int bin = (io->info->options.output.iorformat == IO_RECORD_BINARY);

  colloids_info_ntotal(io->info, &ntotal);
  colloids_info_nlocal(io->info, &nlocal);

  /* If ntotal is zero, no colloids */
  if (ntotal == 0) return 0;

  /* If nlocal is zero, we must still make the collective call */
  /* (1 + nlocal) to ensure no zero-sized allocations */

  if (asc) nbufsz = LUDWIG_COLLOID_IO_BUFSZ;
  if (bin) nbufsz = sizeof(colloid_state_t);

  buf = (char *) calloc(nbufsz*(1 + nlocal), sizeof(char));
  sp  = (sort_by_pointer_t *) calloc(1 + nlocal, sizeof(sort_by_pointer_t));

  if (buf == NULL || sp == NULL) goto err;

  /* The current list must be consistent with nlocal so .. */
  /* .. ensure the cell list is up-to-date before output */
  /* The alternative is to use the true cell list. */

  colloids_info_list_local_build(io->info);
  colloids_info_local_head(io->info, &pc);

  for (int n = 0; pc; pc = pc->nextlocal, n++) {
    sp[n].index = pc->s.index;
    sp[n].pc    = pc;
  }

  qsort(sp, nlocal, sizeof(sort_by_pointer_t), compare_by_pointer);

  /* Write the ordered particle data to the contiguous buffer */
  /* Displacements are global indices - 1 */

  idisp = (int *) calloc(1 + nlocal, sizeof(int));
  if (idisp == NULL) goto err;

  for (int ib = 0; ib < nlocal; ib++) {
    pc        = sp[ib].pc;
    idisp[ib] = sp[ib].pc->s.index - 1;

    if (asc) ifail = colloid_state_io_write_buf_ascii(&pc->s, buf + ib*nbufsz);
    if (bin) ifail = colloid_state_io_write_buf(&pc->s, buf + ib*nbufsz);

    if (ifail != 0) goto err;
  }

  {
    MPI_Datatype etype    = MPI_DATATYPE_NULL;
    MPI_Datatype filetype = MPI_DATATYPE_NULL;

    MPI_Type_contiguous(nbufsz, MPI_CHAR, &etype);
    MPI_Type_commit(&etype);
    MPI_Type_create_indexed_block(nlocal, 1, idisp, etype, &filetype);
    MPI_Type_commit(&filetype);

    /* Clobber any existing file (this is "standard operating procedure") */
    /* Ignore any errors (e.g., file does not exist) */

    ifail = MPI_File_delete(filename, MPI_INFO_NULL);

    /* File */
    {
      MPI_File   fh    = MPI_FILE_NULL;
      MPI_Info   info  = MPI_INFO_NULL;
      MPI_Offset hdisp = 0;

      int amode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
      int rank  = 0;

      /* Size of header (bytes), i.e., displacement of first record */
      /* The ASCII header is 23 characters (see format below) */
      /* The binary header is just one int (ntotal) */
      if (asc) hdisp = 23*sizeof(char);
      if (bin) hdisp = sizeof(int);

      ifail = MPI_File_open(io->comm, filename, amode, info, &fh);
      if (ifail != MPI_SUCCESS) goto err_file;

      MPI_Comm_rank(io->comm, &rank);

      if (rank == 0) {
        /* Write header */
        char hdr[BUFSIZ] = {0};

        if (asc) snprintf(hdr, hdisp + 1, "%22d\n", ntotal);
        if (bin) memcpy(hdr, &ntotal, sizeof(int));

        assert(sizeof(int) == 4*sizeof(char)); /* For binary case. */

        ifail =
            MPI_File_write_at(fh, 0, hdr, hdisp, MPI_CHAR, MPI_STATUS_IGNORE);
        if (ifail != MPI_SUCCESS) {
          goto err_file;
        }
      }

      ifail = MPI_File_set_view(fh, hdisp, etype, filetype, "native", info);
      if (ifail != MPI_SUCCESS) goto err_file;
      ifail = MPI_File_write_all(fh, buf, nlocal, etype, MPI_STATUS_IGNORE);
      if (ifail != MPI_SUCCESS) goto err_file;
      ifail = MPI_File_close(&fh);
    }
  err_file:

    MPI_Type_free(&filetype);
    MPI_Type_free(&etype);
  }

err:
  free(idisp);
  free(sp);
  free(buf);

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_mpio_read_ascii
 *
 *****************************************************************************/

int colloid_io_mpio_read_ascii(colloid_io_mpio_t * io, FILE * fp) {

  int ifail = 0;
  int nread = 0;

  assert(io);
  assert(fp);

  int nr = fscanf(fp, "%22d", &nread);
  if (nr != 1) goto err;

  for (int n = 0; n < nread; n++) {
    char buf[LUDWIG_COLLOID_IO_BUFSZ + 1] = {0}; /* Make sure we have a \0 */
    colloid_state_t s                     = {0};

    nr = fread(buf, LUDWIG_COLLOID_IO_BUFSZ*sizeof(char), 1, fp);
    if (nr != 1) goto err;

    ifail = colloid_state_io_read_buf_ascii(&s, buf);
    if (ifail != 0) goto err;

    /* Add if local and assign the state */
    {
      colloid_t * pc = NULL;
      colloids_info_add_local_with_state(io->info, &s, &pc);
    }
  }

err:

  /* This is collective so cannot by-pass on error */
  /* Set the total number of colloids (all ranks). This must agree with
   * the number in the file header. */
  {
    int nlocal = 0;
    colloids_info_nlocal(io->info, &nlocal);
    MPI_Allreduce(&nlocal, &io->info->ntotal, 1, MPI_INT, MPI_SUM, io->comm);
    if (io->info->ntotal != nread) ifail = -1;
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_io_mpio_read_binary
 *
 *****************************************************************************/

int colloid_io_mpio_read_binary(colloid_io_mpio_t * io, FILE * fp) {

  int ifail = 0;
  int nread = 0;

  assert(io);
  assert(fp);

  /* Header */
  int nr = fread(&nread, sizeof(int), 1, fp);
  if (nr != 1) goto err;

  /* Data: one at a time at the moment */
  for (int n = 0; n < nread; n++) {
    char buf[1 + sizeof(colloid_state_t)] = {0}; /* Make sure we have a \0 */
    colloid_state_t s                     = {0};

    nr = fread(buf, sizeof(colloid_state_t), 1, fp);
    if (nr != 1) goto err;

    ifail = colloid_state_io_read_buf(&s, buf);
    if (ifail != 0) goto err;

    /* Add if local and assign the state */
    {
      colloid_t * pc = NULL;
      colloids_info_add_local_with_state(io->info, &s, &pc);
    }
  }

err:

  /* This is collective so cannot by-pass on error */
  /* The only check we can make here is that the global total is correct
   * compared with the number declared in the file */
  /* True for single file. */
  assert(io->subfile.nfile == 1);

  {
    int nlocal = 0;
    colloids_info_nlocal(io->info, &nlocal);
    MPI_Allreduce(&nlocal, &io->info->ntotal, 1, MPI_INT, MPI_SUM, io->comm);
    if (io->info->ntotal != nread) ifail = -1;
  }

  return ifail;
}
