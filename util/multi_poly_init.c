/*****************************************************************************
 *
 *  multi_poly_init.c
 *
 *  Produce a file of polymers information suitable for reading into
 *  the main code.
 *
 *  Polymer positions are initialised at random. Multiple polymers can 
 *  be generated. This code is suitable for dilute or intermediate polymer
 *  solutions.
 *  It should not be used if the system is dense. Boundary walls can not 
 *  be included. 
 *
 *  A 'grace' distance dh may be specified to prevent the initial monomer 
 *  positions being too close together.
 *
 *  For compilation instructions see the Makefile.
 *
 *  $ make multi_poly_init
 *
 *  $ ./multi_poly_init
 *
 *  should produce a file config.cds.init.001-001 in the specified format.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Contributing authors
 *  Kai Qi (kai.qi@epfl.ch)
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2021 The University of Edinburgh
 *  (c) 2020- Swiss Federal Institute of Technology Lausanne
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../src/colloid.h"
#include "../src/pe.h"
#include "../src/coords.h"
#include "../src/util.h"

enum format {ASCII, BINARY};

void colloid_init_trial(cs_t * cs, int * lcgstate, double r[3], double dh);
void colloid_init_write_file(const int nc, const colloid_state_t * pc,
			     const int form);
void poly_init_random(cs_t * cs, int * lcgstate, int nc,
		      colloid_state_t * state, double dh, int Lpoly,
		      int Npoly, double lbond);

void grow_one_monomer(cs_t * cs, int * lcgstate, double r1[3], double r2[3],
		      double dh,double lbond);

/*****************************************************************************
 *
 *  main
 *
 *  You need to set the system parameters found directly below.
 *
 *****************************************************************************/

int main(int argc, char ** argv) {
  
  int from_file = 1;
  int without_bonds = 1;
  int random_positions = 0;

  int ntotal[3] = {32, 32, 32};        /* Total system size (cf. input) */
  int periodic[3] = {1, 1, 1};         /* 0 = wall, 1 = periodic */
  int file_format = ASCII;

  int n;
  int nrequest;


  double a0 = 0.2;    /* Input radius */
  double ah = 0.2;      /* Hydrodynamic radius */ 
  double al = 1.25;     /* Offset parameter for subgrid particle */
  double dh = 0.50;     /* "grace' distance */
  double q0 = 0.0;      /* positive charge */ 
  double q1 = 0.0;      /* negative charge */
  double b1 = 0.00;
  double b2 = 0.00;

  /* COLLOID_TYPE_DEFAULT  fully resolved standard;
   * COLLOID_TYPE_ACTIVE   squirmer;
   * COLLOID_TYPE_SUBGRID  subgrid. For polymers, must be subgrid. */

  int type  = COLLOID_TYPE_SUBGRID;

  int Npoly = 4;        /* number of polymers */
  int Lpoly = 20;       /* length of a polymer */
  double lbond = 1.0;   /* bond length */
  //CHANGE1
  int inter_type=0; /* interaction type: 0,1,2,3...; put 0 if only one type of interaction is used*/

  colloid_state_t * state;
  pe_t * pe;
  cs_t * cs;

  int lcg = 12345;      /* Random number generator initial state */


  MPI_Init(&argc, &argv);

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);

  /* This program is intended to be serial */
  assert(pe_mpi_size(pe) == 1);

  cs_create(pe,&cs);
  cs_ntotal_set(cs, ntotal);
  cs_periodicity_set(cs, periodic);
  cs_init(cs);

  /* Allocate required number of state objects, and set state
     to zero; initialise indices (position set later) */

  if (from_file) {
    /* Must know number of colloids in advance */
    nrequest = 61;
  }
  else {
    nrequest=Npoly*Lpoly;
  }
  state = (colloid_state_t *) calloc(nrequest, sizeof(colloid_state_t));
  assert(state != NULL);
  for (int n = 0; n < nrequest; n++) {
    state[n].isfixedr = 0;
    state[n].isfixedv = 0;
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
    state[n].type = type;
    if (type == COLLOID_TYPE_SUBGRID) {
      state[n].al= al;
      /* Needs a_L */
      state[n].u0 = 0.00001;
      state[n].delta = 5.0;
      state[n].cutoff = 5.0;
    }
    state[n].rng = 1 + n;
    //CHANGE1
    state[n].inter_type=inter_type;
  }

  if (from_file) {
    int numcol;
    float pos[3], phi_production;
    int ni[3];
    char data[256];
    char line[256];
    int iscentre;
    int indexcentre;

    FILE* file;
    file = fopen("latticeFullerene.txt", "r");

    while (fgets(line, sizeof(line), file)) {
      strcpy(data,line);
      sscanf(data, "%d %f %f %f %d %d %d %d %d %f", &numcol, &pos[0], &pos[1], &pos[2], &ni[0], &ni[1], &ni[2], &iscentre, &indexcentre, &phi_production);
      numcol = numcol - 1;
      state[numcol].r[X] = pos[X];
      state[numcol].r[Y] = pos[Y];
      state[numcol].r[Z] = pos[Z];
      state[numcol].phi_production = phi_production;
      if (without_bonds) {
        state[numcol].nbonds = 0;
        state[numcol].bond[0] = 0;
        state[numcol].bond[1] = 0;
        state[numcol].bond[2] = 0;
      }
      else {
        state[numcol].nbonds = 3;
	state[numcol].nangles = 1;
        state[numcol].bond[0] = ni[X];
        state[numcol].bond[1] = ni[Y];
        state[numcol].bond[2] = ni[Z];
        state[numcol].iscentre = iscentre;
        state[numcol].indexcentre = indexcentre;
	if (iscentre == 1) {
	  state[numcol].nbonds = 0;
	  state[numcol].u0 = 0;
	  state[numcol].nangles = 0;
        }
      }
    }
  fclose(file);
  }

  if (random_positions) {
    poly_init_random(cs, &lcg, nrequest, state, dh, Lpoly, Npoly, lbond);
  }

  colloid_init_write_file(nrequest, state, file_format);
  free(state);

  cs_free(cs);
  pe_free(pe);
  MPI_Finalize();

  return 0;
}

/****************************************************************************
 *
 *  colloid_init_trial
 *
 *  Produce a random trial position based on system size.
 *
 ****************************************************************************/

void colloid_init_trial(cs_t * cs, int * lcgstate, double r[3], double dh) {

  int ia, ntotal[3];
  int periodic[3];
  double lmin, lmax, l[3];
  double u01;                /* uniform random draw on [0,1] */

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
    util_ranlcg_reap_uniform(lcgstate, &u01);
    r[ia] = lmin + (lmax - lmin)*u01;
  }

}

/*****************************************************************************
 *
 * grow_one_monomer
 *
 * Grow a single monomer. 
 *
 *****************************************************************************/

void grow_one_monomer(cs_t * cs, int * lcgstate, double r1[3], double r2[3],
		      double dh,double lbond) {
  int ia, ntotal[3];
  int periodic[3];
  double bd_min[3], bd_max[3], l[3],rand_vec[3];
  int exceed;

  cs_lmin(cs, l);
  cs_ntotal(cs, ntotal);
  cs_periodic(cs, periodic);

  for (ia = 0; ia < 3; ia++) {
    bd_min[ia] = l[ia];
    bd_max[ia] = l[ia] +  ntotal[ia];
    if (periodic[ia] == 0) {
      bd_min[ia] += dh;
      bd_max[ia] -= dh;
    }
    assert(bd_max[ia] >= bd_min[ia]);
  }

  do {
    util_random_unit_vector(lcgstate, rand_vec);

    exceed=0;
    for (ia = 0; ia < 3; ia++) {
      r2[ia] = r1[ia] + lbond*rand_vec[ia];
      if (r2[ia] <= bd_min[ia] ||  r2[ia] >= bd_max[ia]) {exceed=1;break;}
    }
  } while(exceed);
}

/*****************************************************************************
 *
 * poly_init_random
 *
 * We grow each polymer from a single monomer. Initially, the first monomer is
 * randomly placed in the simulation box. Then, the second monomer is grown on 
 * a sphere with radius equal to the bond length. The first monmoer locates at 
 * the center of the sphere. The code will repeat this procedure until a full
 * length polymer is generated.  
 *
 *****************************************************************************/

void poly_init_random(cs_t * cs, int * lcgstate, int nc,
		      colloid_state_t * state, double dh, int Lpoly, int Npoly,
		      double lbond) {

  double rtrial[3];
  int mon1,mon2,monl;
  int Nmon=0;           /* count how many monomers have been set  */
  int monc;             /* check monomer */
  int overlap;
  double rsep[3];


  for (int pl = 0; pl < Npoly; pl++) {
    
    mon1 = pl*Lpoly;

    do {
      colloid_init_trial(cs, lcgstate, rtrial, state[mon1].ah + dh);

        overlap=0;
        monc=0;
        while (monc < Nmon) {
	  cs_minimum_distance(cs, rtrial, state[monc].r, rsep);
	  if (modulus(rsep) <= state[mon1].ah + state[monc].ah + dh) {
	    overlap = 1;
	    break;
	  }
	  monc++;
        }
    } while(overlap);

    state[mon1].r[X] = rtrial[X];
    state[mon1].r[Y] = rtrial[Y];
    state[mon1].r[Z] = rtrial[Z];
    state[mon1].nbonds=1;
    state[mon1].bond[0]=mon1+2;
    Nmon++;


    for (monl = 1; monl < Lpoly; monl++) {
        mon2=pl*Lpoly+monl;
        mon1=mon2-1;

        do {
	  grow_one_monomer(cs, lcgstate, state[mon1].r,rtrial,
			   state[mon2].ah+dh, lbond);
	  overlap=0;
	  monc=0;
	  while (monc < Nmon) {
	    cs_minimum_distance(cs, rtrial, state[monc].r, rsep);
	    if (modulus(rsep) <= state[mon2].ah + state[monc].ah + dh) {
	      overlap = 1;
	      break;
	    }
	    monc++;
	  }
        } while(overlap);

        state[mon2].r[X] = rtrial[X];
        state[mon2].r[Y] = rtrial[Y];
        state[mon2].r[Z] = rtrial[Z];
        if (monl < (Lpoly-1)) {
            state[mon2].nbonds=2;
            state[mon2].bond[0]=mon2;
            state[mon2].bond[1]=mon2+2;
        }
        else {
            state[mon2].nbonds=1;
            state[mon2].bond[0]=mon2;
        }
        Nmon++;
    }

  }

  assert(Nmon==Npoly*Lpoly);
  
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
