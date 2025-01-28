/*****************************************************************************
 *
 *  colloid_state_io.c
 *
 *  Basic colloid state i/o functions of ascii/binary.
 *
 *****************************************************************************/

#include <string.h>

#include "colloid_state_io.h"

/* Override COLLOID_IO_VERSION here: */
#define IO_VERSION 240

/*****************************************************************************
 *
 *  colloid_state_io_write_buf
 *
 *  Write exactly sizeof(colloid_state_t) bytes to the buffer, for the
 *  colloid with local ordinal index index.
 *
 *****************************************************************************/

int colloid_state_io_write_buf(const colloid_state_t * s, char * buf) {

  int ifail = 0;

  if (s == NULL || buf == NULL) {
    ifail = -1;
  }
  else {
    memcpy(buf, s, sizeof(colloid_state_t));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_state_io_write_buf_ascii
 *
 *  Write (ASCII) state of colloid with ordinal index to buf (without a '\0').
 *
 *****************************************************************************/

int colloid_state_io_write_buf_ascii(const colloid_state_t * s, char * buf) {

  int ifail = 0;

  if (s == NULL || buf == NULL) {
    ifail = -1;
  }
  else {
    const size_t item = 25*sizeof(char); /* Single datum is 25 char ... */
    const char * i1format = "%24d\n";
    const char * d3format = "%24.15e ";  /* space */
    const char * d1format = "%24.15e\n"; /* new line */

    int nwrite = 0;                     /* no. chars that was wrote */
    char cbuf[1 + NTOT_VAR*25] = {0};   /* buffer add 1 for the '\0' */

    /* Write to the local buffer cbuf , then copy to buf without the '\0` */

    nwrite += snprintf(cbuf          , 1 + item, i1format, s->index);
    nwrite += snprintf(cbuf +  1*item, 1 + item, i1format, s->rebuild);
    nwrite += snprintf(cbuf +  2*item, 1 + item, i1format, s->nbonds);
    nwrite += snprintf(cbuf +  3*item, 1 + item, i1format, s->nangles);
    nwrite += snprintf(cbuf +  4*item, 1 + item, i1format, s->isfixedr);
    nwrite += snprintf(cbuf +  5*item, 1 + item, i1format, s->isfixedv);
    nwrite += snprintf(cbuf +  6*item, 1 + item, i1format, s->isfixedw);
    nwrite += snprintf(cbuf +  7*item, 1 + item, i1format, s->isfixeds);
    nwrite += snprintf(cbuf +  8*item, 1 + item, i1format, s->type);
    nwrite += snprintf(cbuf +  9*item, 1 + item, i1format, s->bond[0]);
    nwrite += snprintf(cbuf + 10*item, 1 + item, i1format, s->bond[1]);
    nwrite += snprintf(cbuf + 11*item, 1 + item, i1format, s->rng);
    nwrite += snprintf(cbuf + 12*item, 1 + item, i1format, s->isfixedrxyz[0]);
    nwrite += snprintf(cbuf + 13*item, 1 + item, i1format, s->isfixedrxyz[1]);
    nwrite += snprintf(cbuf + 14*item, 1 + item, i1format, s->isfixedrxyz[2]);
    nwrite += snprintf(cbuf + 15*item, 1 + item, i1format, s->isfixedvxyz[0]);
    nwrite += snprintf(cbuf + 16*item, 1 + item, i1format, s->isfixedvxyz[1]);
    nwrite += snprintf(cbuf + 17*item, 1 + item, i1format, s->isfixedvxyz[2]);
    nwrite += snprintf(cbuf + 18*item, 1 + item, i1format, s->inter_type);
    /* This is the i/o version; we ignore s->ioversion: */
    nwrite += snprintf(cbuf + 19*item, 1 + item, i1format, IO_VERSION);
    nwrite += snprintf(cbuf + 20*item, 1 + item, i1format, s->bc);
    nwrite += snprintf(cbuf + 21*item, 1 + item, i1format, s->shape);
    nwrite += snprintf(cbuf + 22*item, 1 + item, i1format, s->active);
    nwrite += snprintf(cbuf + 23*item, 1 + item, i1format, s->magnetic);
    nwrite += snprintf(cbuf + 24*item, 1 + item, i1format, s->attr);
    nwrite += snprintf(cbuf + 25*item, 1 + item, i1format, s->intpad[0]);
    nwrite += snprintf(cbuf + 26*item, 1 + item, i1format, s->intpad[1]);
    nwrite += snprintf(cbuf + 27*item, 1 + item, i1format, s->intpad[2]);
    nwrite += snprintf(cbuf + 28*item, 1 + item, i1format, s->intpad[3]);
    nwrite += snprintf(cbuf + 29*item, 1 + item, i1format, s->intpad[4]);
    nwrite += snprintf(cbuf + 30*item, 1 + item, i1format, s->intpad[5]);
    nwrite += snprintf(cbuf + 31*item, 1 + item, i1format, s->intpad[6]);

    /* Integers */
    if (nwrite != 32*item) ifail = -1;

    /* Doubles */
    /* For historical reasons, vectors r, v, etc appear on one line */

    nwrite += snprintf(cbuf + 32*item, 1 + item, d1format, s->a0);
    nwrite += snprintf(cbuf + 33*item, 1 + item, d1format, s->ah);
    nwrite += snprintf(cbuf + 34*item, 1 + item, d3format, s->r[0]);
    nwrite += snprintf(cbuf + 35*item, 1 + item, d3format, s->r[1]);
    nwrite += snprintf(cbuf + 36*item, 1 + item, d1format, s->r[2]);
    nwrite += snprintf(cbuf + 37*item, 1 + item, d3format, s->v[0]);
    nwrite += snprintf(cbuf + 38*item, 1 + item, d3format, s->v[1]);
    nwrite += snprintf(cbuf + 39*item, 1 + item, d1format, s->v[2]);
    nwrite += snprintf(cbuf + 40*item, 1 + item, d3format, s->w[0]);
    nwrite += snprintf(cbuf + 41*item, 1 + item, d3format, s->w[1]);
    nwrite += snprintf(cbuf + 42*item, 1 + item, d1format, s->w[2]);
    nwrite += snprintf(cbuf + 43*item, 1 + item, d3format, s->s[0]);
    nwrite += snprintf(cbuf + 44*item, 1 + item, d3format, s->s[1]);
    nwrite += snprintf(cbuf + 45*item, 1 + item, d1format, s->s[2]);
    nwrite += snprintf(cbuf + 46*item, 1 + item, d3format, s->m[0]);
    nwrite += snprintf(cbuf + 47*item, 1 + item, d3format, s->m[1]);
    nwrite += snprintf(cbuf + 48*item, 1 + item, d1format, s->m[2]);
    nwrite += snprintf(cbuf + 49*item, 1 + item, d1format, s->b1);
    nwrite += snprintf(cbuf + 50*item, 1 + item, d1format, s->b2);
    nwrite += snprintf(cbuf + 51*item, 1 + item, d1format, s->c);
    nwrite += snprintf(cbuf + 52*item, 1 + item, d1format, s->h);
    nwrite += snprintf(cbuf + 53*item, 1 + item, d3format, s->dr[0]);
    nwrite += snprintf(cbuf + 54*item, 1 + item, d3format, s->dr[1]);
    nwrite += snprintf(cbuf + 55*item, 1 + item, d1format, s->dr[2]);
    nwrite += snprintf(cbuf + 56*item, 1 + item, d1format, s->deltaphi);
    nwrite += snprintf(cbuf + 57*item, 1 + item, d1format, s->q0);
    nwrite += snprintf(cbuf + 58*item, 1 + item, d1format, s->q1);
    nwrite += snprintf(cbuf + 59*item, 1 + item, d1format, s->epsilon);
    nwrite += snprintf(cbuf + 60*item, 1 + item, d1format, s->deltaq0);
    nwrite += snprintf(cbuf + 61*item, 1 + item, d1format, s->deltaq1);
    nwrite += snprintf(cbuf + 62*item, 1 + item, d1format, s->sa);
    nwrite += snprintf(cbuf + 63*item, 1 + item, d1format, s->saf);
    nwrite += snprintf(cbuf + 64*item, 1 + item, d1format, s->al);
    nwrite += snprintf(cbuf + 65*item, 1 + item, d1format, s->elabc[0]);
    nwrite += snprintf(cbuf + 66*item, 1 + item, d1format, s->elabc[1]);
    nwrite += snprintf(cbuf + 67*item, 1 + item, d1format, s->elabc[2]);
    nwrite += snprintf(cbuf + 68*item, 1 + item, d1format, s->quat[0]);
    nwrite += snprintf(cbuf + 69*item, 1 + item, d1format, s->quat[1]);
    nwrite += snprintf(cbuf + 70*item, 1 + item, d1format, s->quat[2]);
    nwrite += snprintf(cbuf + 71*item, 1 + item, d1format, s->quat[3]);
    nwrite += snprintf(cbuf + 72*item, 1 + item, d1format, s->quatold[0]);
    nwrite += snprintf(cbuf + 73*item, 1 + item, d1format, s->quatold[1]);
    nwrite += snprintf(cbuf + 74*item, 1 + item, d1format, s->quatold[2]);
    nwrite += snprintf(cbuf + 75*item, 1 + item, d1format, s->quatold[3]);
    nwrite += snprintf(cbuf + 76*item, 1 + item, d1format, s->dpad[0]);
    nwrite += snprintf(cbuf + 77*item, 1 + item, d1format, s->dpad[1]);
    nwrite += snprintf(cbuf + 78*item, 1 + item, d1format, s->dpad[2]);
    nwrite += snprintf(cbuf + 79*item, 1 + item, d1format, s->dpad[3]);

    if (nwrite != NTOT_VAR*item) ifail = -2;

    /* Finally */
    memcpy(buf, cbuf, NTOT_VAR*item*sizeof(char));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_state_io_read_buf
 *
 *****************************************************************************/

int colloid_state_io_read_buf(colloid_state_t * s, const char * buf) {

  int ifail = 0;

  if (s == NULL || buf == NULL) {
    ifail = -1;
  }
  else {
    memcpy(s, buf, sizeof(colloid_state_t));
  }

  return ifail;
}

/*****************************************************************************
 *
 *  colloid_state_io_read_buf_ascii
 *
 *****************************************************************************/

int colloid_state_io_read_buf_ascii(colloid_state_t * s, const char * buf) {

  int ifail = 0;

  if (s == NULL || buf == NULL) {
    ifail = -1;
  }
  else {
    /* Make sure there is a \0 before we get to sscanf, hence the memcpy() */
    int nr = 0;                       /* number of char read */
    int sz = 25*sizeof(char);
    char tmp[BUFSIZ] = {0};

    memcpy(tmp, buf +  0*sz, sz); nr += sscanf(tmp, "%d", &s->index);
    memcpy(tmp, buf +  1*sz, sz); nr += sscanf(tmp, "%d", &s->rebuild);
    memcpy(tmp, buf +  2*sz, sz); nr += sscanf(tmp, "%d", &s->nbonds);
    memcpy(tmp, buf +  3*sz, sz); nr += sscanf(tmp, "%d", &s->nangles);
    memcpy(tmp, buf +  4*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedr);
    memcpy(tmp, buf +  5*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedv);
    memcpy(tmp, buf +  6*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedw);
    memcpy(tmp, buf +  7*sz, sz); nr += sscanf(tmp, "%d", &s->isfixeds);
    memcpy(tmp, buf +  8*sz, sz); nr += sscanf(tmp, "%d", &s->type);
    memcpy(tmp, buf +  9*sz, sz); nr += sscanf(tmp, "%d", &s->bond[0]);
    memcpy(tmp, buf + 10*sz, sz); nr += sscanf(tmp, "%d", &s->bond[1]);
    memcpy(tmp, buf + 11*sz, sz); nr += sscanf(tmp, "%d", &s->rng);
    memcpy(tmp, buf + 12*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedrxyz[0]);
    memcpy(tmp, buf + 13*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedrxyz[1]);
    memcpy(tmp, buf + 14*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedrxyz[2]);
    memcpy(tmp, buf + 15*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedvxyz[0]);
    memcpy(tmp, buf + 16*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedvxyz[1]);
    memcpy(tmp, buf + 17*sz, sz); nr += sscanf(tmp, "%d", &s->isfixedvxyz[2]);

    memcpy(tmp, buf + 18*sz, sz); nr += sscanf(tmp, "%d", &s->inter_type);
    memcpy(tmp, buf + 19*sz, sz); nr += sscanf(tmp, "%d", &s->ioversion);
    memcpy(tmp, buf + 20*sz, sz); nr += sscanf(tmp, "%d", &s->bc);
    memcpy(tmp, buf + 21*sz, sz); nr += sscanf(tmp, "%d", &s->shape);
    memcpy(tmp, buf + 22*sz, sz); nr += sscanf(tmp, "%d", &s->active);
    memcpy(tmp, buf + 23*sz, sz); nr += sscanf(tmp, "%d", &s->magnetic);
    memcpy(tmp, buf + 24*sz, sz); nr += sscanf(tmp, "%d", &s->attr);
    memcpy(tmp, buf + 25*sz, sz); nr += sscanf(tmp, "%d", &s->intpad[0]);
    memcpy(tmp, buf + 26*sz, sz); nr += sscanf(tmp, "%d", &s->intpad[1]);
    memcpy(tmp, buf + 27*sz, sz); nr += sscanf(tmp, "%d", &s->intpad[2]);
    memcpy(tmp, buf + 28*sz, sz); nr += sscanf(tmp, "%d", &s->intpad[3]);
    memcpy(tmp, buf + 29*sz, sz); nr += sscanf(tmp, "%d", &s->intpad[4]);
    memcpy(tmp, buf + 30*sz, sz); nr += sscanf(tmp, "%d", &s->intpad[5]);
    memcpy(tmp, buf + 31*sz, sz); nr += sscanf(tmp, "%d", &s->intpad[6]);

    if (nr != 32*sz) ifail = -1;

    /* Doubles */
    memcpy(tmp, buf + 32*sz, sz); nr += sscanf(tmp, "%le", &s->a0);
    memcpy(tmp, buf + 33*sz, sz); nr += sscanf(tmp, "%le", &s->ah);
    memcpy(tmp, buf + 34*sz, sz); nr += sscanf(tmp, "%le", &s->r[0]);
    memcpy(tmp, buf + 35*sz, sz); nr += sscanf(tmp, "%le", &s->r[1]);
    memcpy(tmp, buf + 36*sz, sz); nr += sscanf(tmp, "%le", &s->r[2]);
    memcpy(tmp, buf + 37*sz, sz); nr += sscanf(tmp, "%le", &s->v[0]);
    memcpy(tmp, buf + 38*sz, sz); nr += sscanf(tmp, "%le", &s->v[1]);
    memcpy(tmp, buf + 39*sz, sz); nr += sscanf(tmp, "%le", &s->v[2]);
    memcpy(tmp, buf + 40*sz, sz); nr += sscanf(tmp, "%le", &s->w[0]);
    memcpy(tmp, buf + 41*sz, sz); nr += sscanf(tmp, "%le", &s->w[1]);
    memcpy(tmp, buf + 42*sz, sz); nr += sscanf(tmp, "%le", &s->w[2]);
    memcpy(tmp, buf + 43*sz, sz); nr += sscanf(tmp, "%le", &s->s[0]);
    memcpy(tmp, buf + 44*sz, sz); nr += sscanf(tmp, "%le", &s->s[1]);
    memcpy(tmp, buf + 45*sz, sz); nr += sscanf(tmp, "%le", &s->s[2]);
    memcpy(tmp, buf + 46*sz, sz); nr += sscanf(tmp, "%le", &s->m[0]);
    memcpy(tmp, buf + 47*sz, sz); nr += sscanf(tmp, "%le", &s->m[1]);
    memcpy(tmp, buf + 48*sz, sz); nr += sscanf(tmp, "%le", &s->m[2]);
    memcpy(tmp, buf + 49*sz, sz); nr += sscanf(tmp, "%le", &s->b1);
    memcpy(tmp, buf + 50*sz, sz); nr += sscanf(tmp, "%le", &s->b2);
    memcpy(tmp, buf + 51*sz, sz); nr += sscanf(tmp, "%le", &s->c);
    memcpy(tmp, buf + 52*sz, sz); nr += sscanf(tmp, "%le", &s->h);
    memcpy(tmp, buf + 53*sz, sz); nr += sscanf(tmp, "%le", &s->dr[0]);
    memcpy(tmp, buf + 54*sz, sz); nr += sscanf(tmp, "%le", &s->dr[1]);
    memcpy(tmp, buf + 55*sz, sz); nr += sscanf(tmp, "%le", &s->dr[2]);
    memcpy(tmp, buf + 56*sz, sz); nr += sscanf(tmp, "%le", &s->deltaphi);
    memcpy(tmp, buf + 57*sz, sz); nr += sscanf(tmp, "%le", &s->q0);
    memcpy(tmp, buf + 58*sz, sz); nr += sscanf(tmp, "%le", &s->q1);
    memcpy(tmp, buf + 59*sz, sz); nr += sscanf(tmp, "%le", &s->epsilon);
    memcpy(tmp, buf + 60*sz, sz); nr += sscanf(tmp, "%le", &s->deltaq0);
    memcpy(tmp, buf + 61*sz, sz); nr += sscanf(tmp, "%le", &s->deltaq1);
    memcpy(tmp, buf + 62*sz, sz); nr += sscanf(tmp, "%le", &s->sa);
    memcpy(tmp, buf + 63*sz, sz); nr += sscanf(tmp, "%le", &s->saf);
    memcpy(tmp, buf + 64*sz, sz); nr += sscanf(tmp, "%le", &s->al);
    memcpy(tmp, buf + 65*sz, sz); nr += sscanf(tmp, "%le", &s->elabc[0]);
    memcpy(tmp, buf + 66*sz, sz); nr += sscanf(tmp, "%le", &s->elabc[1]);
    memcpy(tmp, buf + 67*sz, sz); nr += sscanf(tmp, "%le", &s->elabc[2]);

    memcpy(tmp, buf + 68*sz, sz); nr += sscanf(tmp, "%le", &s->quat[0]);
    memcpy(tmp, buf + 69*sz, sz); nr += sscanf(tmp, "%le", &s->quat[1]);
    memcpy(tmp, buf + 70*sz, sz); nr += sscanf(tmp, "%le", &s->quat[2]);
    memcpy(tmp, buf + 71*sz, sz); nr += sscanf(tmp, "%le", &s->quat[3]);

    memcpy(tmp, buf + 72*sz, sz); nr += sscanf(tmp, "%le", &s->quatold[0]);
    memcpy(tmp, buf + 73*sz, sz); nr += sscanf(tmp, "%le", &s->quatold[1]);
    memcpy(tmp, buf + 74*sz, sz); nr += sscanf(tmp, "%le", &s->quatold[2]);
    memcpy(tmp, buf + 75*sz, sz); nr += sscanf(tmp, "%le", &s->quatold[3]);

    memcpy(tmp, buf + 76*sz, sz); nr += sscanf(tmp, "%le", &s->dpad[0]);
    memcpy(tmp, buf + 77*sz, sz); nr += sscanf(tmp, "%le", &s->dpad[1]);
    memcpy(tmp, buf + 78*sz, sz); nr += sscanf(tmp, "%le", &s->dpad[2]);
    memcpy(tmp, buf + 79*sz, sz); nr += sscanf(tmp, "%le", &s->dpad[3]);

    if (nr != NTOT_VAR*sz) ifail = -2;
  }

  return ifail;
}
