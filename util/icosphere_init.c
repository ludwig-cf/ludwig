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
  
  int from_file = 1;
  int random_positions = 0;
  int central_links = 1;
  int two_vesicles = 0;

  int ntotal[3] = {32, 32, 32};        /* Total system size (cf. input) */
  int periodic[3] = {1, 1, 1};         /* 0 = wall, 1 = periodic */
  int file_format = ASCII;

  int n;
  int nrequest;


  double a0 = 0.2;    /* Input radius */
  double ah = 0.2;      /* Hydrodynamic radius */ 
  double al = 1.25;     /* Offset parameter for subgrid particle */
  double dh = 0.50;     /* grace distance */
  double q0 = 0.0;      /* positive charge */ 
  double q1 = 0.0;      /* negative charge */
  double b1 = 0.00;
  double b2 = 0.00;

  /* COLLOID_TYPE_DEFAULT  fully resolved standard;
   * COLLOID_TYPE_ACTIVE   squirmer;
   * COLLOID_TYPE_SUBGRID  subgrid. For polymers, must be subgrid. */

  int type  = COLLOID_TYPE_SUBGRID;

  int Npoly = 1;        /* number of polymers */
  int Lpoly = 1;       /* length of a polymer */
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
    nrequest = 43;
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
    /*
    state[n].m[X] = XXXmxXXX;
    state[n].m[Y] = XXXmyXXX;
    state[n].m[Z] = XXXmzXXX;
    state[n].n[X] = XXXnxXXX;
    state[n].n[Y] = XXXnyXXX;
    state[n].n[Z] = XXXnzXXX;
    */
    state[n].type = type;
    if (type == COLLOID_TYPE_SUBGRID) {
      state[n].al= al;
      /* Needs a_L */
      state[n].u0 = 1e-3;
      state[n].delta = 3;
      state[n].cutoff = 3.0;
    }
    state[n].rng = 1 + n;
    //CHANGE1
    state[n].inter_type=inter_type;
  }

  if (from_file) {
    int numcol;
    float pos[3];
    int mi[7];
    float r0[7];
    int nConnec;
    char data[256];
    char line[256];
    int iscentre, ishole, indexcentre;

    FILE* file;
    file = fopen("latticeIcosphere.txt", "r");

    state[0].nbonds_mesh = 42;
    for (int ind = 0; ind < 42; ind++) {
      state[0].bond_mesh[ind] = ind + 2;
    }

    while (fgets(line, sizeof(line), file)) {
      strcpy(data,line);

      sscanf(data, "%d %f %f %f %d %d %d %d %d %d %d %d %f %f %f %f %f %f %f %d %d %d",
		 &numcol, &pos[0], &pos[1], &pos[2], &nConnec, &mi[0], &mi[1], &mi[2], &mi[3], &mi[4], &mi[5], &mi[6], &r0[0], &r0[1], &r0[2], &r0[3], &r0[4], &r0[5], &r0[6], &iscentre, &ishole, &indexcentre);


      numcol = numcol - 1;

      state[numcol].r[X] = pos[X];
      state[numcol].r[Y] = pos[Y];
      state[numcol].r[Z] = pos[Z];

      state[numcol].nbonds = 0;
      state[numcol].nbonds2 = 0;
      state[numcol].nbonds3 = 0;
      
      state[numcol].bond[0] = 0;
      state[numcol].bond[1] = 0;
      state[numcol].bond[2] = 0;

      state[numcol].bond2[0] = 0;
      state[numcol].bond2[1] = 0;
      state[numcol].bond2[2] = 0;

      state[numcol].bond3[0] = 0;
      state[numcol].bond3[1] = 0;
      state[numcol].bond3[2] = 0;

      if (numcol >= 1) {

        state[numcol].tuple.indices[0] = mi[0];
        state[numcol].tuple.indices[1] = mi[1];
        state[numcol].tuple.indices[2] = mi[2];
        state[numcol].tuple.indices[3] = mi[3];
        state[numcol].tuple.indices[4] = mi[4];
        state[numcol].tuple.indices[5] = mi[5];
        state[numcol].tuple.indices[6] = mi[6];

        state[numcol].tuple.r0s[0] = r0[0];
        state[numcol].tuple.r0s[1] = r0[1];
        state[numcol].tuple.r0s[2] = r0[2];
        state[numcol].tuple.r0s[3] = r0[3];
        state[numcol].tuple.r0s[4] = r0[4];
        state[numcol].tuple.r0s[5] = r0[5];
        state[numcol].tuple.r0s[6] = r0[6];

        state[numcol].nbonds_mesh = nConnec;
        state[numcol].bond_mesh[0] = mi[0];
        state[numcol].bond_mesh[1] = mi[1];
        state[numcol].bond_mesh[2] = mi[2];
        state[numcol].bond_mesh[3] = mi[3];
        state[numcol].bond_mesh[4] = mi[4];
        state[numcol].bond_mesh[5] = mi[5];
        state[numcol].bond_mesh[6] = mi[6];

        for (int numbond = 7; numbond < 42; numbond++) {
          state[numcol].bond_mesh[numbond] = 0;
        }  
      }

      state[numcol].iscentre = iscentre;
      state[numcol].ishole = ishole;
      state[numcol].indexcentre = indexcentre;
      
      if (iscentre == 1) state[numcol].nangles = 0;
      else state[numcol].nangles = 1;

      if (iscentre == 1) {
        state[numcol].u0 = 0;
      }
    }
  fclose(file);
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
