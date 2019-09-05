/*****************************************************************************
 *
 *  field_psi_init_rt.c
 *
 *  Initialisation of surfactant "psi" order parameter field.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2019 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "util.h"
#include "noise.h"
#include "io_harness.h"
#include "field_s.h"
#include "field_psi_init_rt.h"

#define  DEFAULT_SEED       25
#define  DEFAULT_NOISE      0.1
#define  DEFAULT_PATCH_SIZE 1
#define  DEFAULT_PATCH_VOL  0.3
#define  DEFAULT_PATCH_VOL2  0.6

int field_init_uniform(field_t * psi, double psi0);
int field_init_ternary_X(field_t * field);
int field_init_ternary_XY(field_t * field);
int field_init_ternary_bbb(field_t * field);
int field_init_ternary_ggg(field_t * field);
//int field_init_ternary_spinodal(field_t * field, int seed1, double psi0, double amp);
int field_init_patches(field_t * field, int seed1, int patch1,
                       double volminus11,double volminus22);
/*****************************************************************************
 *
 *  field_psi_init_rt
 *
 *  Read the initial coice from the input.
 *
 *  Any additional information must be supplied via the structure
 *  field_psi_info_t parameters.
 *
 *****************************************************************************/

int field_psi_init_rt(pe_t * pe, rt_t * rt, field_psi_info_t param,
		      field_t * psi) {

  int p;
    char value[BUFSIZ];
  char filestub[FILENAME_MAX];
  io_info_t * iohandler = NULL;
    
    assert(pe);
    assert(rt);
    assert(psi);

  p = rt_string_parameter(rt, "psi_initialisation", value, BUFSIZ);

  if (p == 0) pe_fatal(pe, "Please specify psi_initialisation in input\n");

  if (strcmp(value, "uniform") == 0) {
    double psi0;

    pe_info(pe, "Initialising psi to a uniform value psi0\n");

    p = rt_double_parameter(rt, "psi_initialisation_psi0", &psi0);
    if (p == 0) pe_fatal(pe, "Please specify psi0 in input\n");

    pe_info(pe, "Initial value psi0: %14.7e\n", psi0);
    field_init_uniform(psi, psi0);
    return 0;
  }
    if (strcmp(value, "ternary_X") == 0) {
        
        
        pe_info(pe, "Initialising psi to a ternary_X psi0\n");
        
        //p = rt_double_parameter(rt, "psi_initialisation_psi0", &psi0);
        if (p == 0) pe_fatal(pe, "Please specify psi0 in input\n");
        
//pe_info(pe, "Initial value psi0: %14.7e\n", psi0);
        field_init_ternary_X(psi);
        return 0;
    }
    if (strcmp(value, "ternary_XY") == 0) {
        
        
        pe_info(pe, "Initialising psi to a ternary_XY psi0\n");
        
        //p = rt_double_parameter(rt, "psi_initialisation_psi0", &psi0);
        if (p == 0) pe_fatal(pe, "Please specify psi0 in input\n");
        
        //pe_info(pe, "Initial value psi0: %14.7e\n", psi0);
        field_init_ternary_XY(psi);
        return 0;
    }
    if (strcmp(value, "ternary_bbb") == 0) {
        
        
        pe_info(pe, "Initialising psi to a ternary_XY psi0\n");
        
        //p = rt_double_parameter(rt, "psi_initialisation_psi0", &psi0);
        if (p == 0) pe_fatal(pe, "Please specify psi0 in input\n");
        
        //pe_info(pe, "Initial value psi0: %14.7e\n", psi0);
        field_init_ternary_bbb(psi);
        return 0;
    }
    if (strcmp(value, "ternary_ggg") == 0) {
        
        
        pe_info(pe, "Initialising psi to a ternary_XY psi0\n");
        
        //p = rt_double_parameter(rt, "psi_initialisation_psi0", &psi0);
        if (p == 0) pe_fatal(pe, "Please specify psi0 in input\n");
        
        //pe_info(pe, "Initial value psi0: %14.7e\n", psi0);
        field_init_ternary_ggg(psi);
        return 0;
    }
    
    if ( strcmp(value, "patches") == 0) {
        int seed1 = DEFAULT_SEED;
        int patch1 = DEFAULT_PATCH_SIZE;
        double volminus11 = DEFAULT_PATCH_VOL;
        double volminus22 = DEFAULT_PATCH_VOL2;
        pe_info(pe, "Initialising psi in patches\n");
        printf("ini");
        rt_int_parameter(rt, "random_seed", &seed1);
        printf("random_seed");
        rt_int_parameter(rt, "psi_init_patch_size", &patch1);
         printf("patch");
        rt_double_parameter(rt, "psi_init_patch_vol", &volminus11);
        printf("volum1");
        rt_double_parameter(rt, "psi_init_patch_vol2", &volminus22);
        printf("volum2");
        field_init_patches(psi, seed1, patch1, volminus11,volminus22);
        printf("end");
        return 0;
    }
    if (strcmp(value, "from_file") == 0) {
        pe_info(pe, "Initial order parameter requested from file\n");
        strcpy(filestub, "psi.init"); /* A default */
        rt_string_parameter(rt, "psi_file_stub", filestub, FILENAME_MAX);
        pe_info(pe, "Attempting to read psi from file: %s\n", filestub);
        
        field_io_info(psi, &iohandler);
        io_read_data(iohandler, filestub, psi);
        return 0;
    }

  pe_fatal(pe, "Initial psi choice not recognised: %s\n", value);

  return 0;
}

/*****************************************************************************
 *
 *  field_init_uniform
 *
 *****************************************************************************/

int field_init_uniform(field_t * field, double value) {

  int nlocal[3];
  int ic, jc, kc, index;

  assert(field);

  cs_nlocal(field->cs, nlocal);

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(field->cs, ic, jc, kc);
	field_scalar_set(field, index, value);

      }
    }
  }

  return 0;
}
/*****************************************************************************
 *
 *  field_init_ternary_X
 *
 *  Initialise one block of chosen thickness at central position on X
 *
 *****************************************************************************/

int field_init_ternary_X(field_t * field) {
    
    int nlocal[3];
    int noffset[3];
    int ic, jc, kc, index;
    double x;
    double psi0;
    double len[3];
    
    cs_nlocal(field->cs, nlocal);
    cs_nlocal_offset(field->cs, noffset);
    cs_ltot(field->cs, len);
    
    // x1 = 0.5*(len[X] - xwidth);
    // x2 = 0.5*(len[X] + xwidth);
    
    for (ic = 1; ic <= nlocal[X]; ic++) {
        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {
                
                index = cs_index(field->cs, ic, jc, kc);
                x = noffset[X] + ic;
                
                if (x < 0.3*len[X]) {
                    psi0 = 0.0;
                }
                else if (x > 0.6*len[X]) {
                    psi0 = 0.0;
                }
                else
                {
                    psi0 = 1.0;
                }
                
                field_scalar_set(field, index, psi0);
            }
        }
    }
    
    return 0;
}
/*****************************************************************************
 *
 *  field_init_ternary_XY
 *
 *  Initialise one block of chosen thickness at central position on X
 *
 *****************************************************************************/

int field_init_ternary_XY(field_t * field) {
    
    int nlocal[3];
    int noffset[3];
    int ic, jc, kc, index;
    double x, y;
    double psi0;
    double len[3];
    
    cs_nlocal(field->cs, nlocal);
    cs_nlocal_offset(field->cs, noffset);
    cs_ltot(field->cs, len);
    
    // x1 = 0.5*(len[X] - xwidth);
    // x2 = 0.5*(len[X] + xwidth);
    
    for (ic = 1; ic <= nlocal[X]; ic++) {
        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {
                
                index = cs_index(field->cs, ic, jc, kc);
                x = noffset[X] + ic;
                y = noffset[X] + jc;
                
                if (x < 0.5*len[X]  && x > 0.2*len[X] && y > 0.3*len[Y]  && y < 0.7*len[Y]) {
                    psi0 = 0.0;
                }
                else if (x < 0.8*len[X]  && x > 0.5*len[X] && y > 0.3*len[Y]  && y < 0.7*len[Y]) {
                    psi0 = 0.0;
                }
                else
                {
                    psi0 = 1.0;
                }
                
                field_scalar_set(field, index, psi0);
            }
        }
    }
    
    return 0;
}
/*****************************************************************************
 *
 *  field_init_ternary_bb
 *
 *  Initialise one block of chosen thickness at central position on X
 *
 *****************************************************************************/

int field_init_ternary_bbb(field_t * field) {
    
    int nlocal[3];
    int noffset[3];
    int ic, jc, kc, index;
    double x, y;
    double psi0;
    double len[3];
    
    cs_nlocal(field->cs, nlocal);
    cs_nlocal_offset(field->cs, noffset);
    cs_ltot(field->cs, len);
    
    
    for (ic = 1; ic <= nlocal[X]; ic++) {
        for (jc = 1; jc <= 0.5*nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {
                
                index = cs_index(field->cs, ic, jc, kc);
                x = noffset[X] + ic;
                y = noffset[X] + jc;
                
                
                if ((x-0.5*len[X])*(x-0.5*len[X])+(y-0.5*len[Y])*(y-0.5*len[Y]) <= 0.25*len[X]*0.25*len[X]) {
                    psi0 = 1.0;
                }
                else {
                    psi0 = 0.0;
                }
                
                field_scalar_set(field, index, psi0);
            }
        }
    }
    for (ic = 1; ic <= nlocal[X]; ic++) {
        for (jc = 0.5*nlocal[Y]; jc <= 1.0*nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {
                
                index = cs_index(field->cs, ic, jc, kc);
                x = noffset[X] + ic;
                y = noffset[X] + jc;
                
                if ((x-0.5*len[X])*(x-0.5*len[X])+(y-0.5*len[Y])*(y-0.5*len[Y]) <= 0.25*len[X]*0.25*len[X] ) {
                    psi0 = 1.0;
                }
                else {
                    psi0 = 0.0;
                }
                
                
                field_scalar_set(field, index, psi0);
            }
        }
    }
    
    return 0;
}
/*****************************************************************************
 *
 *  field_init_ternary_ggg
 *
 *  Initialise one block of chosen thickness at central position on X
 *
 *****************************************************************************/

int field_init_ternary_ggg(field_t * field) {
    
    int nlocal[3];
    int noffset[3];
    int ic, jc, kc, index;
    double x, y;
    double psi0;
    double len[3];
    
    cs_nlocal(field->cs, nlocal);
    cs_nlocal_offset(field->cs, noffset);
    cs_ltot(field->cs, len);
    
    
    for (ic = 1; ic <= nlocal[X]; ic++) {
        for (jc = 1; jc <= nlocal[Y]; jc++) {
            for (kc = 1; kc <= nlocal[Z]; kc++) {
                
                index = cs_index(field->cs, ic, jc, kc);
                x = noffset[X] + ic;
                y = noffset[X] + jc;
                
                
                if ((x-0.36*len[X])*(x-0.36*len[X])+(y-0.5*len[Y])*(y-0.5*len[Y]) <= 0.15*len[X]*0.15*len[X]) {
                    psi0 = 0.0;
                }
                else if ((x-0.64*len[X])*(x-0.64*len[X])+(y-0.5*len[Y])*(y-0.5*len[Y]) <= 0.15*len[X]*0.15*len[X]){
                    psi0 = 0.0;
                }
                else
                {
                    psi0 = 1.0;
                }
                field_scalar_set(field, index, psi0);
            }
        }
    }
    
    return 0;
}
/*****************************************************************************
 *
 *  field_init_patches
 *
 *  Also for spinodal, but a slightly different strategy of patches,
 *  which is better for large composition ratios.
 *
 *  Generally, the further away from 50:50 one moves, the larger
 *  the patch size must be to prevent diffusion (via the mobility)
 *  washing out the spinodal decomposition.
 *
 *  Composition is initialised with phi = +1 or phi = -1
 *  Patch is the patch size in lattice sites
 *  volminus1 is the overall fraction of the phi = -1 phase.
 *
 *****************************************************************************/
int field_init_patches(field_t * field, int seed1, int patch1,
                       double volminus11,double volminus22) {
    
    int ic, jc, kc, index;
    int ip, jp, kp;
    int nlocal[3];
    int ipatch1, jpatch1, kpatch1;
    int count = 0;
    
    double psi1;
    double ran_uniform;
    
    noise_t * rng = NULL;
    
    assert(field);
    
    cs_nlocal(field->cs, nlocal);
    
    noise_create(field->pe, field->cs, &rng);
    noise_init(rng, seed1);
    
    for (ic = 1; ic <= nlocal[X]; ic += patch1) {
        for (jc = 1; jc <= nlocal[Y]; jc += patch1) {
            for (kc = 1; kc <= nlocal[Z]; kc += patch1) {
                
                index = cs_index(field->cs, ic, jc, kc);
                
                /* Uniform patch */
                psi1 = 1.0;
                noise_uniform_double_reap(rng, index, &ran_uniform);
                
                if (ran_uniform < volminus11) {psi1 = 0.0;}
                else if (ran_uniform >= volminus11 && ran_uniform < volminus22){ psi1=1.0; }
                else {psi1 = 0.0;}
                
                
                ipatch1 = dmin(nlocal[X], ic + patch1 - 1);
                jpatch1 = dmin(nlocal[Y], jc + patch1 - 1);
                kpatch1 = dmin(nlocal[Z], kc + patch1 - 1);
                
                for (ip = ic; ip <= ipatch1; ip++) {
                    for (jp = jc; jp <= jpatch1; jp++) {
                        for (kp = kc; kp <= kpatch1; kp++) {
                            
                            index = cs_index(field->cs, ip, jp, kp);
                             field_scalar_set(field, index, psi1);
                            count += 1;
                        }
                    }
                }
                
                /* Next patch1 */
            }
        }
    }
    
    noise_free(rng);
    
    assert(count == nlocal[X]*nlocal[Y]*nlocal[Z]);
    
    return 0;
}
