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
 *  $ ./colloid_init [-a a0] [-h ah] [-v volume_fraction] [-X -Y -Z]
 *  should produce a file config.cds.init.001-001 in the specified format.
 *
 *  Options:
 *    -a sets a0 the input radius
 *    -d sets the grace distance dr between colloids (default 0.5)
 *    -h sets ah the hydrodynamic radius
 *    -n sets the number of colloids (cannot be specified with -v)
 *    -v sets the volume fraction 0 < vf < 1.0 (usually < 0.5)
 *
 *    -X sets total X system size (integer) > 0 [default 64)
 *    -Y sets total Y system size (integer) > 0 (default 64)
 *    -Z sers total Z system size (integer) > 0 (default 64)
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  (c) 2012-2026 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../src/colloid.h"
#include "../src/colloid_io_impl.h"
#include "../src/colloid_options.h"
#include "../src/pe.h"
#include "../src/coords.h"
#include "../src/util.h"
#include "../src/util_fopen.h"
#include "../src/ran.h"

/* Various limits on random steps */
#define A0_MIN  1.0
#define NTRYMAX 10000
#define NMC     10000000

const char * filename_ = "colloids-000000000.dat"; /* output file name */

int colloid_init_vf_n(cs_t * cs, const double ah, const double vf);
void colloid_init_trial(cs_t * cs, double r[3], double dh);
int colloid_init_random(cs_t * cs, int nc, colloid_state_t * state, double dh);
int colloid_init_mc(cs_t * cs, int nc, colloid_state_t * state, double dh);
int colloid_init_write_file_mpio(pe_t * pe, cs_t * cs, int nc,
                                 colloid_state_t *       state,
                                 io_record_format_enum_t ioformat);
double v_lj(double r, double rc);

/*****************************************************************************
 *
 *  main
 *
 *  Some of the parameters are still hardwired.
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int ntotal[3]   = {64, 64, 64}; /* Total system size (cf. input) */
  int periodic[3] = {1, 1, 1};    /* 0 = wall, 1 = periodic */

  int n;
  int nuser    = 0;
  int nrequest = 0;
  int nactual;
  int optind;

  io_record_format_enum_t file_format = IO_RECORD_ASCII;

  /* Default values */
  /* The -ve ah is used to signal whether the option has been specified
   * on the command line. It is adjusted accordingly. */

  const double a0_default =  2.3;  /* Input radius */
  const double ah_default = -2.3;  /* Hydrodynamic radius */
  const double vf_default = 0.02;  /* Volume fraction */
  const double dh_default =  0.5;  /* "grace' distance */
  const double q0_default =  0.0;  /* positive charge */
  const double q1_default =  0.0;  /* negative charge */

  /* At the moment, three quantities can come from the command line */

  double a0 = a0_default;
  double ah = ah_default;
  double vf = vf_default;
  double dh = dh_default;
  double q0 = q0_default;
  double q1 = q1_default;

  const double pi = 4.0*atan(1.0);
  colloid_state_t * state;

  pe_t * pe = NULL;
  cs_t * cs = NULL;

  MPI_Init(&argc, &argv);

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  ran_init(pe);

  /* Check the command line, then parse the meta data information,
   * and sort out the data file name  */

  if ((argc - 1) % 2 != 0) {
    printf("Usage: %s [-a a0] [-h ah] [-v volume-fraction]\n", argv[0]);
    exit(EXIT_FAILURE);
  }

  for (optind = 1; optind < argc && argv[optind][0] == '-'; optind += 2) {
    switch (argv[optind][1]) {
    case 'a':
      a0 = atof(argv[optind + 1]);
      printf("%s: option -a sets a0 = %f\n", argv[0], a0);
      break;
    case 'd':
      dh = atof(argv[optind + 1]);
      printf("%s: option -d sets dh = %f\n", argv[0], dh);
      break;
    case 'h':
      ah = atof(argv[optind + 1]);
      printf("%s: option -h sets ah = %f\n", argv[0], ah);
      break;
    case 'n':
      nuser = atoi(argv[optind + 1]);
      printf("%s: option -n sets no. of particles = %d\n", argv[0], nuser);
      break;
    case 'v':
      vf = atof(argv[optind + 1]);
      printf("%s: option -v sets vf = %f\n", argv[0], vf);
      break;
    case 'X': {
      size_t nx = 0;
      if (1 != sscanf(argv[optind + 1], "%zu", &nx)) {
        printf("-X: cannot interpret %s as integer\n", argv[optind + 1]);
        exit(-1);
      }
      if (1 <= nx && nx <= INT_MAX) {
        ntotal[X] = nx;
      }
      else {
        printf("-X option must be system size > 0 and < INT_MAX\n");
      }
    } break;
    case 'Y': {
      size_t ny = 0;
      if (1 != sscanf(argv[optind + 1], "%zu", &ny)) {
        printf("-Y: cannot interpret %s as integer\n", argv[optind + 1]);
        exit(-1);
      }
      if (1 <= ny && ny <= INT_MAX) {
        ntotal[Y] = ny;
      }
      else {
        printf("-Y option must be system size > 0 and < INT_MAX\n");
      }
    } break;
    case 'Z': {
      size_t nz = 0;
      if (1 != sscanf(argv[optind + 1], "%zu", &nz)) {
        printf("-Z: cannot interpret %s as integer\n", argv[optind + 1]);
        exit(-1);
      }
      if (1 <= nz && nz <= INT_MAX) {
        ntotal[Z] = nz;
      }
      else {
        printf("-Z option must be system size > 0 and < INT_MAX\n");
      }
    } break;
    default:
      fprintf(stderr, "Unrecognised option: %s\n", argv[optind]);
      fprintf(stderr, "Usage: %s [-ahv]\n", argv[0]);
      exit(EXIT_FAILURE);
    }
  }

  /* This program is intended to be serial */
  assert(pe_mpi_size(pe) == 1);

  cs_create(pe, &cs);
  cs_ntotal_set(cs, ntotal);
  cs_periodicity_set(cs, periodic);
  cs_init(cs);

  /* Check a0 and ah are consistent. If no ah has been set, ah = a0. */
  /* In any case, ah >= a0 so that no hard-sphere overlaps can occur. */

  if (a0 < A0_MIN) {
    printf("You have set a0 =  %5.2f\n", a0);
    printf("Please use   a0 >= %5.2f\n", A0_MIN);
    exit(-1);
  }

  if (ah < 0.0) {
    /* If the user has set a -ve value (!), this message won't quite make
       sense ... */
    printf("%s: option -h not set: using ah = a0\n", argv[0]);
    ah = a0;
  }
  if (ah < a0) {
    /* If the user has requested this ... */
    printf("%s: ah must be >= a0\n", argv[0]);
    exit(-1);
  }

  /* Allocate required number of state objects, and set state
     to zero; initialise indices (position set later) */
  /* The volume fraction will be computed on the basis of
   * an effective radius max(a0, ah), but does not include the
   * 'grace distance'. */

  if (nuser > 0) {
    double aeff = ah; /* ah >= a0 */

    vf = (4.0/3.0)*pi*aeff*aeff*aeff*nuser/(ntotal[X]*ntotal[Y]*ntotal[Z]);
    printf("The system size is: X: %8d\n", ntotal[X]);
    printf("                    Y: %8d\n", ntotal[Y]);
    printf("                    Z: %8d\n", ntotal[Z]);
    printf("Particles requested:   %8d\n", nuser);
    printf("Effective radius:      %5.2f\n", aeff);
    printf("Volume fraction:       %10.3e\n", vf);
    if (0.0 < vf && vf < 1.0) {
      nrequest = nuser;
    }
    else {
      printf("Requested volume fraction cannot be met: %f\n\n", vf);
      exit(-1);
    }
  }
  else {
    nrequest = colloid_init_vf_n(cs, ah, vf);
    printf("The system size is: X: %8d\n", ntotal[X]);
    printf("                    Y: %8d\n", ntotal[Y]);
    printf("                    Z: %8d\n", ntotal[Z]);
    printf("Effective radius:      %5.2f\n", ah);
    printf("Volume fraction %7.3f gives %d colloids\n", vf, nrequest);
  }

  state = (colloid_state_t *) calloc(nrequest, sizeof(colloid_state_t));
  assert(state != NULL);

  for (n = 0; n < nrequest; n++) {
    state[n].index   = 1 + n;
    state[n].rebuild = 1;
    state[n].a0      = a0;
    state[n].ah      = ah;
    state[n].q0      = q0;
    state[n].q1      = q1;
    state[n].rng     = 1 + n;
    state[n].bc      = COLLOID_BC_BBL;
    state[n].shape   = COLLOID_SHAPE_SPHERE;
  }

  if (vf < 0.35) {
    /* Set positions through random insertion */
    nactual = colloid_init_random(cs, nrequest, state, dh);
  }
  else {
    /* Set positions through lattice and MC */
    nactual = colloid_init_mc(cs, nrequest, state, dh);
  }

  /* Write out */
  {
    int ifail =
        colloid_init_write_file_mpio(pe, cs, nactual, state, file_format);
    if (ifail != 0) {
      printf("Problem writing colloid file. Please check the output.\n");
    }
  }

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

  int    n, ntotal[3];
  double volume;
  PI_DOUBLE(pi);

  assert(cs);

  cs_ntotal(cs, ntotal);

  volume = ntotal[X] * ntotal[Y] * ntotal[Z];
  n      = vf*volume/(4.0*pi*ah*ah*ah/3.0);

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

void colloid_init_trial(cs_t * cs, double r[3], double dh) {

  int    ia, ntotal[3];
  int    periodic[3];
  double lmin, lmax, l[3];

  cs_lmin(cs, l);
  cs_ntotal(cs, ntotal);
  cs_periodic(cs, periodic);

  for (ia = 0; ia < 3; ia++) {
    lmin = l[ia];
    lmax = l[ia] + ntotal[ia];
    if (periodic[ia] == 0) {
      lmin += dh;
      lmax -= dh;
    }
    assert(lmax >= lmin);
    r[ia] = lmin + (lmax - lmin) * ran_serial_uniform();
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

int colloid_init_random(cs_t * cs, int nc, colloid_state_t * state,
                        double dh) {

  int n = 0; /* Current number of positions successfully set */
  int ok;
  int ncheck;
  int nattempt;
  int nmaxattempt;

  double rtrial[3];
  double rsep[3];

  assert(cs);

  nmaxattempt = nc*NTRYMAX;

  for (nattempt = 0; nattempt < nmaxattempt; nattempt++) {

    colloid_init_trial(cs, rtrial, state[n].ah + dh);

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

  printf("Randomly placed %d colloids in %d attempts\n", n, nattempt + 1);

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

int colloid_init_mc(cs_t * cs, int nc, colloid_state_t * state, double dh) {

  int ix, iy, iz;
  int ii, ij, ik, im;
  int n, ncheck;
  int nbcc, nactual;

  int nx, ny, nz;
  int nxe, nye, nze;
  int ntotal[3];
  int ok;

  double    ah_ref;
  double    d_ci, d_cc, d_bndry, rsqrt3 = 1.0 / sqrt(3.0);
  double ** rbcc;
  int *     r, *s;

  double rtrial[3], rsep[3], dr;
  double eno, enn, boltzfac, ran;
  double delta;

  assert(cs);
  cs_ntotal(cs, ntotal);

  /* assuming there is one colloid, take radius */
  ah_ref = state[0].ah;

  d_ci = 2.0*ah_ref + dh; // distance corner site - intertitial site
  d_cc = 2.0*rsqrt3*d_ci; // distance corner site - corner site

  d_bndry = dh + ah_ref; // position of first corner site

  // number of complete unit cells along each dimension
  nx = floor((ntotal[X] - 2.0*d_bndry) / d_cc);
  ny = floor((ntotal[Y] - 2.0*d_bndry) / d_cc);
  nz = floor((ntotal[Z] - 2.0*d_bndry) / d_cc);

  // can we squeeze in another interstitial site?
  nxe = floor((ntotal[X] - 2.0*d_bndry) / d_cc - nx + 0.5);
  nye = floor((ntotal[Y] - 2.0*d_bndry) / d_cc - ny + 0.5);
  nze = floor((ntotal[Z] - 2.0*d_bndry) / d_cc - nz + 0.5);

  nbcc = (nx + 1)*(ny + 1)*(nz + 1); // corner sites in all ucs
  nbcc += nx*ny*nz;                  // interstitial sites in all ucs
  nbcc += nxe*ny*nz;                 // additional interstitial sites
  nbcc += nye*nx*nz;
  nbcc += nze*nx*ny;

  // position of bcc sites
  rbcc = (double **) calloc(nbcc, sizeof(double *));
  for (n = 0; n < nbcc; n++) {
    rbcc[n] = (double *) calloc(3, sizeof(double));
  }

  // reservoir sampling
  r = (int *) calloc(nc, sizeof(int));
  s = (int *) calloc(nbcc, sizeof(int));

  n = 0;

  // position of corner sites
  for (ix = 0; ix <= nx; ix++) {
    for (iy = 0; iy <= ny; iy++) {
      for (iz = 0; iz <= nz; iz++) {

        rbcc[n][0] = d_bndry + d_cc*ix;
        rbcc[n][1] = d_bndry + d_cc*iy;
        rbcc[n][2] = d_bndry + d_cc*iz;
        n++;
      }
    }
  }

  // position of interstitial sites
  for (ix = 0; ix < nx; ix++) {
    for (iy = 0; iy < ny; iy++) {
      for (iz = 0; iz < nz; iz++) {

        rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc*ix;
        rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc*iy;
        rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc*iz;
        n++;
      }
    }
  }

  // extra interstitial sites
  for (ix = nx; ix < nx + nxe; ix++) {
    for (iy = 0; iy < ny; iy++) {
      for (iz = 0; iz < nz; iz++) {

        rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc*ix;
        rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc*iy;
        rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc*iz;
        n++;
      }
    }
  }

  // extra interstitial sites
  for (ix = 0; ix < nx; ix++) {
    for (iy = ny; iy < ny + nye; iy++) {
      for (iz = 0; iz < nz; iz++) {

        rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc*ix;
        rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc*iy;
        rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc*iz;
        n++;
      }
    }
  }

  // extra interstitial sites
  for (ix = 0; ix < nx; ix++) {
    for (iy = 0; iy < ny; iy++) {
      for (iz = nz; iz < nz + nze; iz++) {

        rbcc[n][0] = d_bndry + d_ci*rsqrt3 + d_cc*ix;
        rbcc[n][1] = d_bndry + d_ci*rsqrt3 + d_cc*iy;
        rbcc[n][2] = d_bndry + d_ci*rsqrt3 + d_cc*iz;
        n++;
      }
    }
  }

  n = 0;

  // reservoir sampling if nc < nbcc
  for (ii = 0; ii < nbcc; ii++) s[ii] = ii;
  for (ij = 0; ij < nc; ij++)   r[ij] = s[ij];

  for (ii = nc; ii < nbcc; ii++) {
    ik = rand() % ii;
    if (ik < nc) r[ik] = s[ii];
  }

  for (ik = 0; ik < nc; ik++) {
    state[ik].r[X] = rbcc[r[ik]][0];
    state[ik].r[Y] = rbcc[r[ik]][1];
    state[ik].r[Z] = rbcc[r[ik]][2];
    n++;
  }

  nactual = n;

  if (nactual != nc) {
    printf("Mismatch between number of assigned and requested sites\n");
  }

  ok = 1;

  // now check for overlap
  for (n = 0; n < nc; n++) {

    for (ncheck = 0; ncheck < nc; ncheck++) {
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
    printf("Overlap between %d colloids on bcc lattice. Aborting ...\n",
           nactual);
    exit(1);
  }

  delta = 0.01 * ah_ref;

  for (im = 0; im < NMC; im++) {

    if (im % 1000 == 0) {
      printf("MC move #%d\r", im);
      fflush(stdout);
    }

    eno = 0.0;
    enn = 0.0;

    ii = rand() % nc;

    rtrial[X] = state[ii].r[X];
    rtrial[Y] = state[ii].r[Y];
    rtrial[Z] = state[ii].r[Z];

    for (ij = 0; ij < nc; ij++) {

      if (ij == ii) continue;

      cs_minimum_distance(cs, rtrial, state[ij].r, rsep);
      dr = modulus(rsep) - state[ii].ah - state[ij].ah;

      eno += v_lj(dr, dh);
    }

    rtrial[X] += delta * (ran_serial_uniform() - 0.5);
    rtrial[Y] += delta * (ran_serial_uniform() - 0.5);
    rtrial[Z] += delta * (ran_serial_uniform() - 0.5);

    for (ij = 0; ij < nc; ij++) {

      if (ij == ii) continue;

      cs_minimum_distance(cs, rtrial, state[ij].r, rsep);
      dr = modulus(rsep) - state[ii].ah - state[ij].ah;

      enn += v_lj(dr, dh);
    }

    boltzfac = exp(-(enn - eno));
    ran      = ran_serial_uniform();

    if (ran < boltzfac) {
      state[ii].r[X] = rtrial[X];
      state[ii].r[Y] = rtrial[Y];
      state[ii].r[Z] = rtrial[Z];
    }
  }

  ok = 1;

  // now check for overlap again
  for (n = 0; n < nc; n++) {

    for (ncheck = 0; ncheck < nc; ncheck++) {
      if (ncheck == n) continue;

      cs_minimum_distance(cs, state[n].r, state[ncheck].r, rsep);

      if (modulus(rsep) <= state[n].ah + state[ncheck].ah) {
        ok = 0;
        break;
      }
    }
  }

  if (ok) {
    printf("Applied %d MC moves\n", NMC);
  }
  else {
    printf("MC moves created overlap. Aborting ...\n");
    exit(1);
  }

  return nactual;
}

/*****************************************************************************
 *
 *  colloid_init_write_file_mpio
 *
 *****************************************************************************/

int colloid_init_write_file_mpio(pe_t * pe, cs_t * cs, int nc,
                                 colloid_state_t *       state,
                                 io_record_format_enum_t ioformat) {
  int               ifail    = 0;
  int               ncell[3] = {8, 8, 8};
  colloid_options_t opts     = colloid_options_ncell(ncell);
  colloids_info_t   info     = {};

  opts.output.mode      = COLLOID_IO_MODE_MPIIO;
  opts.output.iorformat = ioformat;

  colloids_info_initialise(pe, cs, &opts, &info);

  /* Insert the colloids into the cell list ... */
  for (int ic = 0; ic < nc; ic++) {
    colloid_t * pc = NULL;
    ifail = colloids_info_add_local(&info, state + ic, &pc);
    assert(ifail == 0);
  }
  colloids_info_ntotal_set(&info);

  /* Write output to file */
  {
    colloid_io_impl_t * io = NULL;

    colloid_io_impl_output(&info, &io);
    printf("Write to: %s\n", filename_);
    ifail = io->impl->write(io, filename_);
    io->impl->free(&io);
  }

  /* Finish */

  colloids_info_finalise(&info);

  return ifail;
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
    v_rc = 4.0*eps*(pow(sig / rc, 12) - pow(sig / rc, 6));
    v    = 4.0*eps*(pow(sig / dr, 12) - pow(sig / dr, 6)) - v_rc;
  }
  else {
    v = 0.0;
  }

  return v;
}
