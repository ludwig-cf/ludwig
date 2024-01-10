/*****************************************************************************
 *
 *  coll_squ_subgrid_init.c
 *
 *  Produce a file of colloid/squirmer/subgrid particle information suitable 
 *  for reading into the main code.
 *
 *  Particle positions are initialised at random to provide a requested
 *  volume fraction. Flat boundary walls may be included. A 'grace'
 *  distance dh may be specified to prevent the initial particle 
 *  positions being too close together, or too close to the wall.
 *
 *  For compilation instructions see the Makefile.
 *
 *  $ make
 *
 *  $ ./coll_squ_subgrid_init
 *
 *  should produce a file config.cds.init.001-001 in the specified format.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kai Qi (kai.qi@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2021 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/pe.h"
#include "../src/coords.h"
#include "../src/colloid.h"
#include "../src/util.h"
#include "../src/util_fopen.h"

enum format {ASCII, BINARY};
#define NTRYMAX 10000

int  colloid_init_vf_n(cs_t * cs, const double ah, const double vf);
void colloid_init_trial(cs_t * cs, int * lcg, double r[3], double dh);
int  colloid_init_random(cs_t * cs, int * lcg, int nc,
			 colloid_state_t * state, double dh);
int  colloid_init_mc(cs_t * cs, int * lcg, int nc, int mc,
		     colloid_state_t * state, double dh);
void colloid_init_write_file(const int nc, const colloid_state_t * pc,
			     const int form);
double v_lj(double r, double rc);

/*****************************************************************************
 *
 *  main
 *
 *  You need to set the system parameters found directly below.
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int ntotal[3] = {18, 18, 18};        /* Total system size (cf. input) */
  int periodic[3] = {1, 1, 1};         /* 0 = wall, 1 = periodic */
  int file_format = ASCII;

  int n;
  int nrequest;
  int nactual;
  int optind;

  const double vf_default   = 0.015; /* Solid volume fraction */
  const int    mc_default   = 0;     /* Monte Carlo moves */
  const int    lcg_default  = 12345; /* Random number initial state */ 

  double a0 = 0.178;   /* Input radius */
  double ah = 0.2;     /* Hydrodynamic radius */ 
  double al = 1.58;    /* Offset parameter for subgrid particle */
  double dh = 0.03;    /* "grace' distance */
  double q0 = 0.0;     /* positive charge */ 
  double q1 = 0.0;     /* negative charge */
  double b1 = 0.00;
  double b2 = 0.00;

  int bc = COLLOID_BC_SUBGRID;
  int shape = COLLOID_SHAPE_SPHERE;

  colloid_state_t * state;
  pe_t * pe;
  cs_t * cs;

  double vf  = vf_default;
  int    mc  = mc_default;
  int    lcg = lcg_default;

  MPI_Init(&argc, &argv);

  if ((argc-1) % 2 != 0) {
    printf("Usage: %s [-n Monte-Carlo-moves] [-v colume-fraction]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  for (optind = 1; optind < argc && argv[optind][0] == '-'; optind += 2) {
    switch (argv[optind][1]) {
    case 'n':
      mc = atoi(argv[optind+1]);
      printf("%s: option -n sets mc = %d\n", argv[0], mc);
      break;
    case 'v':
      vf = atof(argv[optind+1]);
      printf("%s: option -v sets vf = %f\n", argv[0], vf);
      break;
    default:
      fprintf(stderr, "Unrecognised option: %s\n", argv[optind]);
      fprintf(stderr, "Usage: %s [-nv]\n", argv[0]);
      exit(EXIT_FAILURE);
    }   
  }

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* This program is intended to be serial */
  assert(pe_mpi_size(pe) == 1);

  cs_create(pe,&cs);
  cs_ntotal_set(cs, ntotal);
  cs_periodicity_set(cs, periodic);
  cs_init(cs);

  /* Allocate required number of state objects, and set state
     to zero; initialise indices (position set later) */

  nrequest = colloid_init_vf_n(cs, ah, vf);
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
    state[n].b1 = b1;
    state[n].b2 = b2;
    state[n].m[X] = 1.0;
    state[n].m[Y] = 0.0;
    state[n].m[Z] = 0.0;
    state[n].bc = bc;
    state[n].shape = shape;
    if (bc == COLLOID_BC_SUBGRID) state[n].al = al;
    state[n].rng = 1 + n;
  }

  if (vf < 0.35) {
    /* Set positions through random insertion */
    nactual = colloid_init_random(cs, &lcg, nrequest, state, dh);
  }
  else {
    /* Set positions through lattice and MC */
    nactual = colloid_init_mc(cs, &lcg, nrequest, mc, state, dh);
  }

  /* Write out */
  colloid_init_write_file(nactual, state, file_format);

  free(state);

  cs_free(cs);
  pe_free(pe);
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

int colloid_init_vf_n(cs_t * cs, const double ah, const double vf) {

  int n, ntotal[3];
  double volume;
  PI_DOUBLE(pi);

  assert(cs);

  cs_ntotal(cs, ntotal);

  volume = ntotal[X]*ntotal[Y]*ntotal[Z];
  n = vf*volume/(4.0*pi*ah*ah*ah/3.0);

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

void colloid_init_trial(cs_t * cs, int * lcg, double r[3], double dh) {

  int ia, ntotal[3];
  int periodic[3];
  double lmin, lmax, l[3];
  double u01;

  assert(cs);
  assert(lcg);

  cs_lmin(cs, l);
  cs_ntotal(cs, ntotal);
  cs_periodic(cs, periodic);

  for (ia = 0; ia < 3; ia++) {
    lmin = l[ia];
    lmax = l[ia] +  ntotal[ia];
    if (periodic[ia] == 0) {
      lmin += dh;
      lmax -= dh;
    }
    assert(lmax >= lmin);
    util_ranlcg_reap_uniform(lcg, &u01);
    r[ia] = lmin + (lmax - lmin)*u01;
  }

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

int colloid_init_random(cs_t * cs, int * lcg, int nc, colloid_state_t * state,
			double dh) {

  int n;              /* Current number of positions successfully set */
  int ok;
  int ncheck;
  int nattempt;
  int nmaxattempt;

  double rtrial[3];
  double rsep[3];

  assert(cs);

  n = 0;
  nmaxattempt = nc*NTRYMAX;

  for (nattempt = 0; nattempt < nmaxattempt; nattempt++) {

    colloid_init_trial(cs, lcg, rtrial, state[n].ah + dh);

    ok = 1;

    for (ncheck = 0; ncheck < n; ncheck++) {
      cs_minimum_distance(cs, rtrial, state[ncheck].r, rsep);
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

  printf("Randomly placed %d colloids in %d attempts\n", n, nattempt+1);

  return n;
}

/****************************************************************************
 *
 *  colloid_init_mc
 *
 *  Place nc colloids on a 3D lattice and perform NMC Monte Carlo moves,
 *  looping over all colloids. The moves consist of independent 
 *  displacements with uniformly distributed stepwith. 
 * 
 ****************************************************************************/

int colloid_init_mc(cs_t * cs, int * lcg, int nc, int nmc,
		    colloid_state_t * state, double dh) {

  int ix, iy, iz;
  int ii, ij, ik, im;
  int n, ncheck;
  int nbcc, nactual;

  int nx, ny, nz;
  int nxe, nye, nze;
  int ntotal[3];
  int ok;

  double ah_ref;
  double d_ci, d_cc, d_bndry, rsqrt3 = 1.0/sqrt(3.0);
  double ** rbcc;
  int * r, * s;

  double rtrial[3], rsep[3], dr;
  double eno, enn, boltzfac, ran;
  double delta;

  assert(cs);
  cs_ntotal(cs, ntotal);

  /* assuming there is one colloid, take radius */
  ah_ref = state[0].ah;

  d_ci = 2.0*ah_ref + dh;    /* distance corner site - intertitial site */
  d_cc = 2.0*rsqrt3 * d_ci;  /* distance corner site - corner site */

  d_bndry = dh + ah_ref;     /* position of first corner site */

  /* number of complete unit cells along each dimension */
  nx =  floor((ntotal[X]- 2.0*d_bndry)/d_cc);
  ny =  floor((ntotal[Y]- 2.0*d_bndry)/d_cc);
  nz =  floor((ntotal[Z]- 2.0*d_bndry)/d_cc);

  /* can we squeeze in another interstitial site? */
  nxe = floor((ntotal[X] - 2.0*d_bndry)/d_cc - nx + 0.5);
  nye = floor((ntotal[Y] - 2.0*d_bndry)/d_cc - ny + 0.5);
  nze = floor((ntotal[Z] - 2.0*d_bndry)/d_cc - nz + 0.5);

  nbcc = (nx+1)*(ny+1)*(nz+1);  /* corner sites in all ucs       */
  nbcc += nx * ny * nz;         /* interstitial sites in all ucs */
  nbcc += nxe * ny * nz;        /* additional interstitial sites */
  nbcc += nye * nx * nz;
  nbcc += nze * nx * ny;

  /* position of bcc sites */
  rbcc = (double **) calloc(nbcc, sizeof(double *));
  for (n = 0; n <nbcc; n++) {
    rbcc[n] = (double *) calloc(3, sizeof(double));
  }

  /* reservoir sampling */
  r = (int *) calloc(nc, sizeof(int));
  s = (int *) calloc(nbcc, sizeof(int));

  n = 0;

  /* position of corner sites */
  for (ix = 0; ix <= nx; ix++) {
    for (iy = 0; iy <= ny; iy++) {
      for (iz = 0; iz <= nz; iz++) {

	rbcc[n][0] = d_bndry + d_cc * ix;
	rbcc[n][1] = d_bndry + d_cc * iy;
	rbcc[n][2] = d_bndry + d_cc * iz;
	n++;

      }
    }
  }

  /* position of interstitial sites */
  for (ix = 0; ix < nx; ix++) {
    for (iy = 0; iy < ny; iy++) {
      for (iz = 0; iz < nz; iz++) {
 
	rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc * ix;
	rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc * iy;
	rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc * iz;
	n++;

      }
    }
  }

  /* extra interstitial sites */
  for (ix = nx; ix < nx+nxe ; ix++) {
    for (iy = 0; iy < ny; iy++) {
      for (iz = 0; iz < nz; iz++) {
 
	rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc * ix;
	rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc * iy;
	rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc * iz;
	n++;

      }
    }
  }

  /* extra interstitial sites */
  for (ix = 0; ix < nx; ix++) {
    for (iy = ny; iy < ny+nye; iy++) {
      for (iz = 0; iz < nz; iz++) {
 
	rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc * ix;
	rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc * iy;
	rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc * iz;
	n++;

      }
    }
  }

  /* extra interstitial sites */
  for (ix = 0; ix < nx; ix++) {
    for (iy = 0; iy < ny; iy++) {
      for (iz = nz; iz < nz+nze; iz++) {
 
	rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc * ix;
	rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc * iy;
	rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc * iz;
	n++;

      }
    }
  }

  n = 0;

  /* reservoir sampling if nc < nbcc */

  for (ii = 0; ii < nbcc; ii++) s[ii] = ii; 
  for (ij = 0; ij < nc; ij++) r[ij] = s[ij]; 

  for (ii = nc; ii < nbcc; ii++) {
    util_ranlcg_reap_uniform(lcg, &ran);
      ik = (*lcg) % ii;
    if (ik < nc) r[ik] = s[ii]; 
  }

  for (ik = 0; ik < nc; ik++) {
    state[ik].r[X] = rbcc[r[ik]][0];
    state[ik].r[Y] = rbcc[r[ik]][1];
    state[ik].r[Z] = rbcc[r[ik]][2];
    n++;
  }

  nactual = n;

  if (nactual != nc ) 
    printf("Mismatch between number of assigned and requested sites\n");

  ok = 1;

  /* now check for overlap */
  for (n=0; n<nc; n++){

    for (ncheck=0; ncheck<nc; ncheck++) {
      if (ncheck == n) continue;
      cs_minimum_distance(cs, state[n].r, state[ncheck].r, rsep);

      if (modulus(rsep) <= state[n].ah + state[ncheck].ah) {
	ok = 0;
	break;
      }

    }

  }

  if (ok) {
    printf("Placed %d colloids on bcc lattice\n", nactual);
  }
  else {
    printf("Overlap between %d colloids on bcc lattice. Aborting ...\n", nactual);
    exit(1);
  }

  delta = 0.01*ah_ref;

  for (im = 0; im < nmc; im++) {

    if (im%1000 == 0) {
      printf("MC move #%d\r", im); 
      fflush(stdout); 
    }

    eno = 0.0;
    enn = 0.0;

    util_ranlcg_reap_uniform(lcg, &ran);
    ii = (*lcg) % nc;

    rtrial[X] = state[ii].r[X];
    rtrial[Y] = state[ii].r[Y];
    rtrial[Z] = state[ii].r[Z];

    for (ij = 0; ij < nc; ij++) {

	if (ij == ii) continue;

	cs_minimum_distance(cs, rtrial, state[ij].r, rsep);
	dr = modulus(rsep) - state[ii].ah - state[ij].ah;

	eno += v_lj(dr, dh); 

    }

    util_ranlcg_reap_uniform(lcg, &ran);
    rtrial[X] += delta * (ran-0.5);
    util_ranlcg_reap_uniform(lcg, &ran);
    rtrial[Y] += delta * (ran-0.5);
    util_ranlcg_reap_uniform(lcg, &ran);
    rtrial[Z] += delta * (ran-0.5);

    for (ij = 0; ij < nc; ij++) {

	if (ij == ii) continue;

	cs_minimum_distance(cs, rtrial, state[ij].r, rsep);
	dr = modulus(rsep) - state[ii].ah - state[ij].ah;

	enn += v_lj(dr, dh); 

    }

    boltzfac = exp(-(enn-eno));
    util_ranlcg_reap_uniform(lcg, &ran);

    if (ran < boltzfac) {
      state[ii].r[X] = rtrial[X]; 
      state[ii].r[Y] = rtrial[Y];
      state[ii].r[Z] = rtrial[Z];
    }

  }

  ok = 1;

  // now check for overlap again
  for (n=0; n<nc; n++){

    for (ncheck=0; ncheck<nc; ncheck++) {
      if (ncheck == n) continue;
      cs_minimum_distance(cs, state[n].r, state[ncheck].r, rsep);

      if (modulus(rsep) <= state[n].ah + state[ncheck].ah) {
	ok = 0;
	break;
      }

    }

  }

  if (ok) {
    printf("Applied %d MC moves\n", nmc);
  }
  else {
    printf("MC moves created overlap. Aborting ...\n");
    exit(1);
  }

  return nactual;

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

  fp = util_fopen(filename, "w");
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
      colloid_state_write_binary(pc+n, fp);
    }
    else {
      colloid_state_write_ascii(pc+n, fp);
    }
  }

  if (ferror(fp)) {
    perror("perror: ");
    printf("Error reported on write to %s\n", filename);
  }

  fclose(fp);

  return;
}

/****************************************************************************
 *
 *  v_lj
 *
 *  Truncated Lennard-Jones interaction modelling soft sphere 
 *  repulsion between colloids in MC routine 
 *
 ****************************************************************************/

double v_lj(double dr, double rc) {

  double eps = 100.0, sig = 0.05;
  double v, v_rc;

  if (dr <= rc) {
    v_rc = 4*eps*(pow(sig/rc,12) - pow(sig/rc,6)); 
    v = 4*eps*(pow(sig/dr,12) - pow(sig/dr,6)) - v_rc; 
  }
  else {
    v = 0.0;
  }

  return v;
}

