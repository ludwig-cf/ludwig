/*****************************************************************************
 *
 *  multi_squ_poly_init.c
 *
 *  Produce a file of squirmer and polymers information suitable for reading into
 *  the main code.
 *
 *  Polymer/squirmer positions are initialised at random. Multiple polymers and 
 *  squirmers can be generated simultaneously. This code is suitable for dilute 
 *  or intermediate polymer solution. It should not be used if the system is dense. 
 *  Boundary walls can not be included. 
 *
 *  A 'grace' distance dh may be specified to prevent the initial particle 
 *  positions being too close together.
 *
 *  For compilation instructions see the Makefile.
 *
 *  $ make multi_squ_poly_init
 *
 *  $ ./a.out
 *  should produce a file config.cds.init.001-001 in the specified format.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2012-2019 The University of Edinburgh
 *
 *  Kai Qi (kai.qi@epfl.ch)
 *  (c) 2020- Swiss Federal Institute of Technology Lausanne
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>


#include "../src/colloid.h"
#include "../src/pe.h"
#include "../src/coords.h"
#include "../src/util.h"
#include "../src/ran.h"

enum format {ASCII, BINARY};
#define NTRYMAX 10000

void colloid_init_trial(cs_t * cs, double r[3], double dh);
void colloid_init_random(cs_t * cs, int nc, colloid_state_t * state,double dh);
void colloid_init_write_file(const int nc, const colloid_state_t * pc,const int form);
void poly_init_random(cs_t * cs, int N_sq, colloid_state_t * state,double dh,int Lpoly,int Npoly,double lbond);

double r2();
void random_unit_vector(double *rand_vec);
void grow_one_monomer(cs_t * cs,double r1[3], double r2[3], double dh,double lbond);

/*****************************************************************************
 *
 *  main
 *
 *  You need to set the system parameters found directly below.
 *
 *****************************************************************************/

int main(int argc, char ** argv) {

  int ntotal[3] = {36, 36, 36};        /* Total system size (cf. input) */
  int periodic[3] = {1, 1, 1};         /* 0 = wall, 1 = periodic */
  int file_format = ASCII;

  int n;
  int nrequest;
  int nactual;


  double a0_pl = 0.179;   /* Input radius */
  double ah_pl = 0.2;   /* Hydrodynamic radius */ 
  double al_pl = 1.66; /* Offset parameter for subgrid particle */
  double dh_pl = 0.50;   /* "grace' distance */
  double q0_pl = 0.0;   /* positive charge */ 
  double q1_pl = 0.0;   /* negative charge */
  double b1_pl = 0.00;
  double b2_pl = 0.00;
  int type_pl=2; /* 0 colloid, 1 squirmer, 2 subgrid; For polymers, you can only put 2. */
  int Npoly=8; /* number of polymers */
  int Lpoly=120; /* length of a polymer */
  double lbond=1.0; /* bond length */
  int inter_type_pl=0; /* interaction type: 0,1,2,3... */

  double a0_sq = 3.0;   /* Input radius */
  double ah_sq = 3.0;   /* Hydrodynamic radius */ 
  double al_sq = 0.0; /* Offset parameter for subgrid particle */
  double dh_sq = 0.5;   /* "grace' distance */
  double q0_sq = 0.0;   /* positive charge */ 
  double q1_sq = 0.0;   /* negative charge */
  double b1_sq = 0.01;
  double b2_sq = -0.05;
  int type_sq=1; /* 0 colloid; 1 squirmer; 2 subgrid */
  int N_sq = 1;  /* number of squirmers */
  int inter_type_sq=1; /* interaction type: 0,1,2,3... */

  colloid_state_t * state;
  pe_t * pe;
  cs_t * cs;

  MPI_Init(&argc, &argv);

  pe_create(MPI_COMM_WORLD, PE_QUIET, &pe);
  ran_init(pe);

  /* This program is intended to be serial */
  assert(pe_mpi_size(pe) == 1);

  cs_create(pe,&cs);
  cs_ntotal_set(cs, ntotal);
  cs_periodicity_set(cs, periodic);
  cs_init(cs);

  /* Allocate required number of state objects, and set state
     to zero; initialise indices (position set later) */

  nrequest=Npoly*Lpoly+N_sq;

  state = (colloid_state_t *) calloc(nrequest, sizeof(colloid_state_t));
  assert(state != NULL);

  srand(time(NULL));

  double rand_vec[3];
  for (n = 0; n < N_sq; n++) {
    state[n].index = 1 + n;
    state[n].rebuild = 1;
    state[n].a0 = a0_sq;
    state[n].ah = ah_sq;
    state[n].q0 = q0_sq;
    state[n].q1 = q1_sq;
    state[n].b1 = b1_sq;
    state[n].b2 = b2_sq;
    random_unit_vector(rand_vec);
    state[n].m[X] = rand_vec[0];
    state[n].m[Y] = rand_vec[1];
    state[n].m[Z] = rand_vec[2];
    state[n].type=type_sq;
    if (type_sq==1)
        state[n].al= al_sq;
    state[n].rng = 1 + n;
    state[n].inter_type=inter_type_sq;
  }

  for (n = N_sq; n < nrequest; n++) {
    state[n].index = 1 + n;
    state[n].rebuild = 1;
    state[n].a0 = a0_pl;
    state[n].ah = ah_pl;
    state[n].q0 = q0_pl;
    state[n].q1 = q1_pl;
    state[n].b1 = b1_pl;
    state[n].b2 = b2_pl;
    state[n].m[X] = 1.0;
    state[n].m[Y] = 0.0;
    state[n].m[Z] = 0.0;
    state[n].type=type_pl;
    if (type_pl==2)
        state[n].al= al_pl;
    state[n].rng = 1 + n;
    state[n].inter_type=inter_type_pl;
  }

  colloid_init_random(cs, N_sq, state, dh_sq);

  poly_init_random(cs,N_sq,state,dh_pl,Lpoly,Npoly,lbond);

  /* Write out */
  colloid_init_write_file(nrequest, state, file_format);

  free(state);

  cs_free(cs);
  pe_free(pe);
  MPI_Finalize();

  return 0;
}

double r2()
{
    return (double)rand() / (double)RAND_MAX ;
}

/****************************************************************************
 *
 *  colloid_init_trial
 *
 *  Produce a random trial position based on system size.
 *
 ****************************************************************************/

void colloid_init_trial(cs_t * cs, double r[3], double dh) {

  int ia, ntotal[3];
  int periodic[3];
  double lmin, lmax, l[3];

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
    r[ia] = lmin + (lmax - lmin)*r2();
    //r[ia] = lmin + (lmax - lmin)*ran_serial_uniform();
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

void colloid_init_random(cs_t * cs, int nc, colloid_state_t * state,double dh) {

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

  printf("Randomly placed %d squimrers in %d attempts\n", n, nattempt+1);
}
/*****************************************************************************
 *
 *random_unit_vector
 *
 *Randomly generate an unit vector.  
 *
 *****************************************************************************/

void random_unit_vector(double *rand_vec) {

    double norm=0;
    for (int i=0;i<3;i++) {
        rand_vec[i]=r2()-0.5;
        //rand_vec[i]=ran_serial_uniform()-0.5;
        norm+=rand_vec[i]*rand_vec[i];
    }

    norm=sqrt(norm);

    for (int i=0;i<3;i++) 
        rand_vec[i]/=norm;
}

/*****************************************************************************
 *
 * grow_one_monomer
 *
 * Grow a single monomer. 
 *
 *****************************************************************************/

void grow_one_monomer(cs_t * cs,double r1[3], double r2[3], double dh,double lbond) {
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

  do{
    random_unit_vector(rand_vec);

    exceed=0;
    for (ia = 0; ia < 3; ia++) {
        r2[ia]=r1[ia]+lbond*rand_vec[ia];
        if(r2[ia]<=bd_min[ia] ||  r2[ia]>=bd_max[ia]) {exceed=1;break;}
    }
  }while(exceed);
}

/*****************************************************************************
 *
 * poly_init_random
 *
 * We grow each polymer from a single monomer. Initially, the first monomer is
 * randomly placed in the simulation box. Then, the second monomer is grown on 
 * a sphere with radius equal to the bond length. The first monmoer locates at 
 * the center of the sphere. The code will repeat this procedure until a full length
 * polymer is generated.  
 *
 *****************************************************************************/

void poly_init_random(cs_t * cs, int N_sq, colloid_state_t * state,double dh,int Lpoly,int Npoly,double lbond) {

  double rtrial[3];
  int mon1,mon2,monl;
  int Nmon=0; //count how many monomers have been set 
  int parc; //check particle 
  int overlap;
  double rsep[3];


  for (int pl=0;pl<Npoly;pl++) {
    
    mon1=pl*Lpoly+N_sq;

    do{
        colloid_init_trial(cs, rtrial, state[mon1].ah + dh);

        overlap=0;
        parc=0;
        while(parc<(Nmon+N_sq)) {
            cs_minimum_distance(cs, rtrial, state[parc].r, rsep);
            if (modulus(rsep) <= state[mon1].ah + state[parc].ah + dh) {overlap = 1;break;}
            parc++;
        }
    }while(overlap);

    state[mon1].r[X] = rtrial[X];
    state[mon1].r[Y] = rtrial[Y];
    state[mon1].r[Z] = rtrial[Z];
    state[mon1].nbonds=1;
    state[mon1].bond[0]=mon1+2;
    Nmon++;


    for (monl=1;monl<Lpoly;monl++) {
        mon2=pl*Lpoly+monl+N_sq;
        mon1=mon2-1;

        do{
            grow_one_monomer(cs,state[mon1].r,rtrial,state[mon2].ah+dh,lbond);
            overlap=0;
            parc=0;
            while(parc<(Nmon+N_sq)) {
                cs_minimum_distance(cs, rtrial, state[parc].r, rsep);
                if (modulus(rsep) <= state[mon2].ah + state[parc].ah + dh) {overlap = 1;break;}
                parc++;
            }
        }while(overlap);

        state[mon2].r[X] = rtrial[X];
        state[mon2].r[Y] = rtrial[Y];
        state[mon2].r[Z] = rtrial[Z];
        if(monl<(Lpoly-1)) {
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

  //for (monl=0;monl<Lpoly;monl++) {
  //  for(int pl=0;pl<Npoly;pl++) {
  //    int mon=pl*Lpoly+monl+N_sq;
  //    printf("%lf   %lf   %lf   ",state[mon].r[X],state[mon].r[Y],state[mon].r[Z]);
  //  }
  //  printf("\n");
  //}
      
  
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
