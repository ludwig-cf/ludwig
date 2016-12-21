/*****************************************************************************
 *
 *  colloid_init.c
 *
 *  Produce a file of colloid information suitable for reading into
 *  the main code.
 *
 *  Colloid positions are initialised at random to provide a requested
 *  volume fraction. Flat boundary walls may be included. A 'grace'
 *  distance dh may be specified to prevent the initial colloid
 *  positions being too close together, or too close to the wall.
 *
 *  For compilation instructions see the Makefile.
 *
 *  $ make colloid_init
 *
 *  $ ./a.out
 *  should produce a file colloid-00000000.001-001 in the specified format.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2016 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "../src/colloid.h"
#include "../src/pe.h"
#include "../src/coords.h"
#include "../src/util.h"
#include "../src/ran.h"

enum format {ASCII, BINARY};
#define NTRYMAX 10000

int  colloid_init_vf_n(const double ah, const double vf);
int  colloid_init_random(const int nc, colloid_state_t * state, double dh);
void colloid_init_trial(double r[3], double dh);
void colloid_init_write_file(const int nc, const colloid_state_t * pc,
			     const int form);

/*****************************************************************************
 *
 *  main
 *
 *  You need to set the system parameters found directly below.
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int ntotal[3] = {128.0, 128.0, 128.0};  /* Total system size (cf. input) */
  int periodic[3] = {1, 1, 1};           /* 0 = wall, 1 = periodic */
  int file_format = BINARY;

  int n;
  int index;
  int nrequest;
  int nactual;

  double a0 = 3.5;                      /* Input radius */
  double ah = 3.5;                      /* Hydrodynamic radius */ 
  double vf = 0.1;                      /* Volume fraction */
  double dh = 1.0;                      /* "grace' distance */
  double q0 = 0.0; 
  double q1 = 0.0;

  colloid_state_t * state;

  MPI_Init(&argc, &argv);
  pe_init();
  ran_init();

  /* This program is intended to be serial */
  assert(pe_size() == 1);

  coords_ntotal_set(ntotal);
  coords_periodicity_set(periodic);
  coords_init();

  /* Allocate required number of state objects, and set state
     to zero; initialise indices (position set later) */

  nrequest = colloid_init_vf_n(ah, vf);
  printf("Volume fraction %7.3f gives %d colloids\n", vf, nrequest);

  state = (colloid_state_t *) calloc(nrequest, sizeof(colloid_state_t));
  assert(state != NULL);

  for (n = 0; n < nrequest; n++) {
    state[n].index = 1 + n;
    state[n].rebuild = 1;
    state[n].a0 = a0;
    state[n].ah = ah;
    state[n].q0 = q0;
    state[n].q1 = q1;
  }

  /* Set positions and write out */

  nactual = colloid_init_random(nrequest, state, dh);
  colloid_init_write_file(nactual, state, file_format);

  free(state);

  coords_finish();
  pe_finalise();
  MPI_Finalize();

  return 0;
}

/****************************************************************************
 *
 *  colloid_vf_n
 *
 *  How many colloids of given ah make up a volume fraction of vf?
 *
 ****************************************************************************/

int colloid_init_vf_n(const double ah, const double vf) {

  int n;
  double volume;

  volume = L(X)*L(Y)*L(Z);
  n = vf*volume/(4.0*pi_*ah*ah*ah/3.0);

  return n;
}

/****************************************************************************
 *
 *  colloid_init_random
 *
 *  Place nc colloids at random positions subject to constraints of
 *  overlap with each other, and walls. The 'grace' distance is dh.
 *
 *  This will make NTRYMAX attempts per colloid requested. As there
 *  is no attempt to do anything with overlaps (except reject them)
 *  the number you get may be less then the number requested.
 *
 ****************************************************************************/

int colloid_init_random(const int nc, colloid_state_t * state, double dh) {

  int n;              /* Current number of positions successfully set */
  int ok;
  int ncheck;
  int nattempt;
  int nmaxattempt;

  double rtrial[3];
  double rsep[3];

  n = 0;
  nmaxattempt = nc*NTRYMAX;

  for (nattempt = 0; nattempt < nmaxattempt; nattempt++) {

    colloid_init_trial(rtrial, state[n].ah + dh);

    ok = 1;

    for (ncheck = 0; ncheck < n; ncheck++) {
      coords_minimum_distance(rtrial, state[ncheck].r, rsep);
      if (modulus(rsep) <= state[ncheck].ah + state[n].ah + dh) ok = 0;
    }

    if (ok) {
      state[n].r[X] = rtrial[X];
      state[n].r[Y] = rtrial[Y];
      state[n].r[Z] = rtrial[Z];
      n += 1;
    }

    if (n == nc) break;
  }

  printf("Ramdonly placed %d colloids in %d attempts\n", n, nattempt+1);

  return n;
}

/****************************************************************************
 *
 *  colloid_init_trial
 *
 *  Produce a random trial position based on system size and presence
 *  of walls.
 *
 ****************************************************************************/

void colloid_init_trial(double r[3], double dh) {

  int ia;
  double lmin, lmax;

  for (ia = 0; ia < 3; ia++) {
    lmin = Lmin(ia);
    lmax = Lmin(ia) +  L(ia);
    if (is_periodic(ia) == 0) {
      lmin += dh;
      lmax -= dh;
    }
    assert(lmax >= lmin);
    r[ia] = lmin + (lmax - lmin)*ran_serial_uniform();
  }

}

/****************************************************************************
 *
 *  colloid_init_write_file
 *
 ****************************************************************************/

void colloid_init_write_file(const int nc, const colloid_state_t * pc,
			     const int form) {
  int n;
  const char * filename = "config.cds.init.001-001";
  FILE * fp;

  fp = fopen(filename, "w");
  if (fp == NULL) {
    printf("Could not open %s\n", filename);
    exit(0);
  }

  if (form == BINARY) {
    fwrite(&nc, sizeof(int), 1, fp);
  }
  else {
    fprintf(fp, "%22d\n", nc);
  }

  for (n = 0; n < nc; n++) {
    if (form == BINARY) {
      colloid_state_write_binary(pc[n], fp);
    }
    else {
      colloid_state_write_ascii(pc[n], fp);
    }
  }

  if (ferror(fp)) {
    perror("perror: ");
    printf("Error reported on write to %s\n", filename);
  }

  fclose(fp);

  return;
}
