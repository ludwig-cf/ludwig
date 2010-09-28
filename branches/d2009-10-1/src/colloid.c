/*****************************************************************************
 *
 *  colloid.c
 *
 *  State type for particles including bounce-back on links, wetting,
 *  magnetic dipoles, and squirmers.
 *
 *  $Id: colloid.c,v 1.1.2.2 2010-09-28 16:21:57 kevin Exp $
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>

#include "colloid.h"

/*****************************************************************************
 *
 *  colloid_state_read_ascii
 *
 *  Returns the number of complete structures read (0 or 1).
 *
 *****************************************************************************/

int colloid_state_read_ascii(colloid_state_t * ps, FILE * fp) {

  int nread = 0;
  int iread = 0;

  const char * sformat = "%22le\n";
  const char * vformat = "%22le %22le %22le\n";

  assert(ps);
  assert(fp);

  nread += fscanf(fp, "%22d\n", &ps->index);
  nread += fscanf(fp, "%22d\n", &ps->rebuild);
  nread += fscanf(fp, sformat, &ps->a0);
  nread += fscanf(fp, sformat, &ps->ah);
  nread += fscanf(fp, vformat, &ps->r[0], &ps->r[1], &ps->r[2]);
  nread += fscanf(fp, vformat, &ps->v[0], &ps->v[1], &ps->v[2]);
  nread += fscanf(fp, vformat, &ps->w[0], &ps->w[1], &ps->w[2]);
  nread += fscanf(fp, vformat, &ps->s[0], &ps->s[1], &ps->s[2]);
  nread += fscanf(fp, vformat, &ps->m[0], &ps->m[1], &ps->m[2]);
  nread += fscanf(fp, sformat, &ps->b1);
  nread += fscanf(fp, sformat, &ps->b2);
  nread += fscanf(fp, sformat, &ps->c);
  nread += fscanf(fp, sformat, &ps->h);
  nread += fscanf(fp, vformat, &ps->dr[0], &ps->dr[1], &ps->dr[2]);
  nread += fscanf(fp, sformat, &ps->deltaphi);
  nread += fscanf(fp, sformat, &ps->spare1);
  nread += fscanf(fp, vformat, &ps->spare2[0], &ps->spare2[1], &ps->spare2[2]);

  /* ... makes a total of 31 items for 1 structure */

  if (nread == 31) iread = 1;

  /* Always set the rebuild flag (even if file has zero) */

  ps->rebuild = 1;

  return iread;
}

/*****************************************************************************
 *
 *  colloid_state_read_binary
 *
 *  Returns the number of complete structures read (0 or 1)
 *
 *****************************************************************************/

int colloid_state_read_binary(colloid_state_t * ps, FILE * fp) {

  int nread;

  assert(ps);
  assert(fp);

  nread = fread(ps, sizeof(colloid_state_t), 1, fp);

  return nread;
}

/*****************************************************************************
 *
 *  colloid_state_write_ascii
 *
 *  Returns the number of complete structures written (0 or 1).
 *
 *****************************************************************************/

int colloid_state_write_ascii(colloid_state_t s, FILE * fp) {

  int nwrite = 0;
  int iwrite = 0;

  const char * sformat = "%22.15e\n";
  const char * vformat = "%22.15e %22.15e %22.15e\n";

  assert(fp);

  nwrite += fprintf(fp, "%22d\n", s.index);
  nwrite += fprintf(fp, "%22d\n", s.rebuild);
  nwrite += fprintf(fp, sformat, s.a0);
  nwrite += fprintf(fp, sformat, s.ah);
  nwrite += fprintf(fp, vformat, s.r[0], s.r[1], s.r[2]);
  nwrite += fprintf(fp, vformat, s.v[0], s.v[1], s.v[2]);
  nwrite += fprintf(fp, vformat, s.w[0], s.w[1], s.w[2]);
  nwrite += fprintf(fp, vformat, s.s[0], s.s[1], s.s[2]);
  nwrite += fprintf(fp, vformat, s.m[0], s.m[1], s.m[2]);
  nwrite += fprintf(fp, sformat, s.b1);
  nwrite += fprintf(fp, sformat, s.b2);
  nwrite += fprintf(fp, sformat, s.c);
  nwrite += fprintf(fp, sformat, s.h);
  nwrite += fprintf(fp, vformat, s.dr[0], s.dr[1], s.dr[2]);
  nwrite += fprintf(fp, sformat, s.deltaphi);
  nwrite += fprintf(fp, sformat, s.spare1);
  nwrite += fprintf(fp, vformat, s.spare2[0], s.spare2[1], s.spare2[2]);

  /* ... should be 31 items of 23 characters */

  if (nwrite == 31*23) iwrite = 1;

  return iwrite;
}

/*****************************************************************************
 *
 *  colloid_state_write_binary
 *
 *  Returns the number of complete structures written (0 or 1)
 *
 *****************************************************************************/

int colloid_state_write_binary(colloid_state_t s, FILE * fp) {

  int nwrite;

  assert(fp);

  nwrite = fwrite(&s, sizeof(colloid_state_t), 1, fp);

  return nwrite;
}
