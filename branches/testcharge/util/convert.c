/*****************************************************************************
 *
 *  Convert old colloid (pre-electrokinetics) -> new colloid type
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "src/colloid.h"

typedef struct old_type old_t;

struct old_type {

  int    index;         /* Unique global index for colloid */
  int    rebuild;       /* Rebuild flag */
  double a0;            /* Input radius (lattice units) */
  double ah;            /* Hydrodynamic radius (from calibration) */
  double r[3];          /* Position */
  double v[3];          /* Velocity */
  double w[3];          /* Angular velocity omega */
  double s[3];          /* Magnetic dipole, or spin */
  double m[3];          /* Currect direction of motion vector (squirmer) */
  double b1;	        /* squirmer active parameter b1 */
  double b2;            /* squirmer active parameter b2 */
  double c;             /* Wetting free energy parameter C */
  double h;             /* Wetting free energy parameter H */
  double dr[3];         /* r update (pending refactor of move/build process) */
  double deltaphi;      /* order parameter bbl net; required to restart */
  double spare1;        /* spare scalar */
  double spare2[3];     /* spare vector */

};

int binary = 0;
int old_read_ascii(old_t * ps, FILE * fp);
int old_read_binary(old_t * ps, FILE * fp);

/*****************************************************************************
 *
 *  main
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int n;
  int ncolloid;

  old_t colloid_old;
  colloid_state_t colloid_new;

  FILE * fp_old = NULL;
  FILE * fp_new = NULL;

  fp_old = fopen("test_yukawa_cds.001-001", "r");
  fp_new = fopen("new-file.001-001", "w");

  if (fp_old == NULL) {
    printf("Failed to open old\n");
    exit(-1);
  }
  if (fp_new == NULL) {
    printf("Failed to open new\n");
    exit(-1);
  }

  /* read / write header */

  if (binary) {
    fread(&ncolloid, sizeof(int), 1, fp_old);
    fwrite(&ncolloid, sizeof(int), 1, fp_new);
  }
  else {
    fscanf(fp_old, "%22d\n",  &ncolloid);
    printf("Collods: %d\n", ncolloid);
    fprintf(fp_new, "%22d\n", ncolloid);    
  }

  /* read /write colloids */

  for (n = 0; n < ncolloid; n++) {
    if (binary) {
      old_read_binary(&colloid_old, fp_old);
      convert_old_new(colloid_old, &colloid_new); 
      colloid_state_write_binary(colloid_new, fp_new);
    }
    else {
      old_read_ascii(&colloid_old, fp_old);
      convert_old_new(colloid_old, &colloid_new);
      colloid_state_write_ascii(colloid_new, fp_new);
    }
  }

  fclose(fp_new);
  fclose(fp_old);

  return 0;
}

/*****************************************************************************
 *
 *  convert_old_new
 *
 ****************************************************************************/

int convert_old_new(old_t old, colloid_state_t * coll) {

  int n;

  coll->index = old.index;
  coll->rebuild = 1;
  coll->nbonds = 0;
  coll->nangles = 0;
  coll->isfixedr = 0;
  coll->isfixedv = 0;
  coll->isfixedw = 0;
  coll->isfixeds = 0;
  for (n = 0; n < NPAD_INT; n++) {
    coll->intpad[n] = 0;
  }
  coll->a0 = old.a0;
  coll->ah = old.ah;
  coll->r[0] = old.r[0];
  coll->r[1] = old.r[1];
  coll->r[2] = old.r[2];
  coll->v[0] = old.v[0];
  coll->v[1] = old.v[1];
  coll->v[2] = old.v[2];
  coll->w[0] = old.w[0];
  coll->w[1] = old.w[1];
  coll->w[2] = old.w[2];
  coll->s[0] = old.s[0];
  coll->s[1] = old.s[1];
  coll->s[2] = old.s[2];
  coll->m[0] = old.m[0];
  coll->m[1] = old.m[1];
  coll->m[2] = old.m[2];
  coll->b1 = old.b1;
  coll->b2 = old.b2;
  coll->c = old.c;
  coll->h = old.h;
  coll->dr[0] = old.dr[0];
  coll->dr[1] = old.dr[1];
  coll->dr[2] = old.dr[2];
  coll->deltaphi = old.deltaphi;

  coll->q0 = 0.0;
  coll->q1 = 0.0;
  coll->epsilon = 0.0;

  for (n = 0; n < NPAD_DBL; n++) {
    coll->dpad[n] = 0.0;
  }

  return 0;
}

/*****************************************************************************
 *
 *  old_read_ascii
 *
 *****************************************************************************/

int old_read_ascii(old_t * ps, FILE * fp) {

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
 *  old_read_binary
 *
 *  Returns the number of complete structures read (0 or 1)
 *
 *****************************************************************************/

int old_read_binary(old_t * ps, FILE * fp) {

  int nread;

  assert(ps);
  assert(fp);

  nread = fread(ps, sizeof(old_t), 1, fp);

  /* Always set the rebuild flag (even if file has zero) */

  ps->rebuild = 1;

  return nread;
}
