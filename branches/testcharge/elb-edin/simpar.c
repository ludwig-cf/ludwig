#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <ctype.h>
#include "simpar.h"

/*
 init_param begins by creating a table of param_t objects, i.e.
 pairs of name values. It does this by scanning line by line the
 input file. In case of comment or blank line, it assigns BLANK to 
 borh name and value but does not add the param_t data to the table.
 In case two strings separated by blanks are found, param_t data is
 built.
 Before adding this element to the table of param_t, check_param checks
 if simulation parameter (first string) exists. If not, program dies.
 This is done looping over name_params structure, checking if there
 is an element of the structure with the same name before the LAST 
 element.
 If check_param ok, then param_t data is added to the param table.
 Lastly set_param is responsible for storing the param_t objects contained
 in the table into the sim_params structure.
 Note that sim_params->XYZ are filled in the same order of sim_params,
 THEREFORE ORDER OF ASSIGNMENT SUCH AS 
 sim_params->width     = atoi(  (char *) &p_value[9] );
 MUST RESPEOCT THE ORDER OF char * name_params[], i.e. width must
 be the 10th element in name_params.

 Therefore, to add parameter, simply add the parameter at the end of 
 name_params[].
 Then add it to struct sim_params in simpar.h, so that it can be assigned
 by set_params.
 So now you can add a line in set_params like this to assign it in set_params,
 at the bottom
 sim_params->NEW PARAM = (real) WAHTEVER( (char *) &p_value[NN] );
 careful with the right NN depending on position 
 in name_params.
 Laslty, add it whereever you want in the printf to output starting
 log of the code.
 */

char * name_params[] = {
  "numsteps", 
  "Nx",
  "Ny",
  "Nz",
  "seed",
  "wall_vel",
  "u_x",
  "u_y",
  "u_z",
  "width",
  "diffusion_acc",
  "num_obj",
  "r_eff",
  "rho0",
  "nu",
  "kbt",
  "ch_bjl",
  "ch_wall",
  "ch_lambda",
  "e_slope_x",
  "e_slope_y",
  "e_slope_z",
  "GreenKubo",
  "restart",
  "outfreq",
  "output",
  "ch_objects",
  "ch_distype",
  "multisteps",
  "D_therm",
  "D_plus",
  "D_minus",
  "f_accuracy",
  "poisson_acc",
  "max_eq_fluxes",
  "r_pair",
  "inf_colloids",
  "move",
  "LAST"
};

int init_param( FILE * cfg_file, sim_params_t * sim_params )
{
  param_t param;
  param_t table_params[TABSIZ];

  int i = 0;
  char l[LINESIZ];
  while( fgets( l, LINESIZ, cfg_file ) != NULL ){
    //printf("Read line  **%s", l);
    int c = 0;
    /* Avoid leading spaces */
    while( l[c] == ' ' || l[c] == '\t' ){ 
      c++; 
    }
    /* Check if it is a comment or a blank line. If so skip it */
    if( l[c] == '#' || l[c] == '\n'){
      strcpy( param.name,  BLANK);
      strcpy( param.value, BLANK);
      //printf("Found %c so Skipping line '%s'", l[c], l); 
    }else{
      /* Now scanning line for name and value */
      char name[PARSIZ];
      char value[PARSIZ];
      sscanf( (char *) l,   "%s %s",   name,   value);
      //printf("Valid line with name %s and value %s\n", (char *) name, (char *) value );
      strcpy( param.name,  (char *) name);
      strcpy( param.value, (char *) value);
      check_param( param, name_params );
      table_params[i] = param;
      //printf("TABLE el %d has name %s and value %s\n", i, table_params[i].name, table_params[i].value);
      i++;
    }
  }
  /* Store table params in paramters structure */
  set_params( sim_params, table_params, name_params );
  return 1;
}

int check_param( param_t param, char * name_params[] )
{
  char * name = (char *) &param.name;
  //char * value = (char *) &param.value;
  int i = 0;
  char * p_name;
  while( (p_name = (char *) name_params[i]) ){
    if( !strcmp(p_name, "LAST") ){ break; }
    //printf("Now checking name %s against name_params[%d] %s\n", (char *) &param.name, i, name_params[i]);
    //if( !strcmp(name, name_params[i]) ){ printf("%s --> %s\n", name, value); return 1; } 
    if( !strcmp(name, name_params[i]) ){ return 1; } 
    i++;
  }
  printf( "Simulation parameter %s does not exist!! Dying....\n", name );
  exit( 1 );
}

long int idum;
int set_params( sim_params_t * sim_params, param_t * table_params, char * name_params[] )
{
  int i = 0;
  char * p_name;
  char p_value[TABSIZ][PARSIZ]; 
  while( (p_name = (char *) name_params[i]) ){
    if( !strcmp( p_name, "LAST") ){ break; }
    lookup_param( p_name, table_params, (char *) p_value[i]  );
    i++;
  }
  sim_params->numsteps   = atoi(  (char *) p_value[0] );
  sim_params->Nx         = atoi(  (char *) p_value[1] );
  sim_params->Ny         = atoi(  (char *) p_value[2] );
  sim_params->Nz         = atoi(  (char *) p_value[3] );
  sim_params->seed       = (long) atoi(  (char *) p_value[4] );
  if( sim_params->seed > 0 )  idum = -sim_params->seed; // ran1 as from num recipes must be initializeh with negative int
  if( sim_params->seed == 0 ){
    idum = (long) -time( NULL );
    sim_params->seed = -idum;
  }
  sim_params->wall_vel   = (real) atof( (char *) p_value[5] );
  sim_params->u_x        = (real) atof( (char *) p_value[6] );
  sim_params->u_y        = (real) atof( (char *) p_value[7] );
  sim_params->u_z        = (real) atof( (char *) p_value[8] );
  sim_params->width      = atoi(  (char *) p_value[9] );
  sim_params->diffusion_acc  = (real) atof( (char *) p_value[10] );
  sim_params->num_obj    = atoi(  (char *) p_value[11] );
  sim_params->r_eff      = (real) atof( (char *) p_value[12]) ;
  sim_params->rho0       = (real) atof( (char *) p_value[13] );
  sim_params->nu         = (real) atof( (char *) p_value[14] );
  sim_params->kbt        = (real) atof( (char *) p_value[15] );
  sim_params->ch_bjl     = (real) atof( (char *) p_value[16] );
  sim_params->ch_wall    = (real) atof( (char *) p_value[17] );
  sim_params->ch_lambda  = (real) atof( (char *) p_value[18] );
  sim_params->e_slope_x  = (real) atof( (char *) p_value[19] );
  sim_params->e_slope_y  = (real) atof( (char *) p_value[20] );
  sim_params->e_slope_z  = (real) atof( (char *) p_value[21] );
  sim_params->GreenKubo   = atoi(  (char *) p_value[22] );
  sim_params->outfreq    = atoi(  (char *) p_value[24] );
  sim_params->ch_objects = (real) atof( (char *) p_value[26] );
  sim_params->multisteps = atoi( (char *) p_value[28] );
  sim_params->D_therm    = (real) atof( (char *) p_value[29] );
  sim_params->D_plus     = (real) atof( (char *) p_value[30] );
  sim_params->D_minus    = (real) atof( (char *) p_value[31] );
  sim_params->f_accuracy    = (real) atof( (char *) p_value[32] );
  sim_params->poisson_acc    = (real) atof( (char *) p_value[33] );
  sim_params->max_eq_fluxes    = (real) atof( (char *) p_value[34] );
  sim_params->r_pair    = (real) atof( (char *) p_value[35] );
  strcpy( sim_params->output, (char *) p_value[25] );
  strcpy( sim_params->restart, (char *) p_value[23] );
  strcpy( sim_params->ch_distype, (char *) p_value[27] );
  strcpy( sim_params->inf_colloids, (char *) p_value[36] );
  sim_params->move = 0;
  if( !strcmp(p_value[37], "yes") ) sim_params->move = 1;
  struct tm *ptr;
  time_t tm;
  tm = time(NULL);
  ptr = localtime( &tm );
  printf("\n\t\tSTARTING elb rv %s SIMULATION @ %s\n",  SVN_REV, asctime(ptr));
  printf("Simulation parameters:\n\n");
  printf("\tnumsteps   -->\t%d\n",    sim_params->numsteps );
  printf("\tNx         -->\t%d\n",    sim_params->Nx );
  printf("\tNy         -->\t%d\n",    sim_params->Ny );
  printf("\tNz         -->\t%d\n",    sim_params->Nz );
  printf("\n\tGreenKubo   -->\t%d\n",    sim_params->GreenKubo );
  printf("\n\tRestart     -->\t%s\n", sim_params->restart );
  if( sim_params->width >= 0 ){
    printf("\n\twall_vel   -->\t%2.2f\n", sim_params->wall_vel );
    printf("\twall width -->\t%d\n",    sim_params->width );
    printf("\tch_wall    -->\t%2.2f\n", sim_params->ch_wall );
  }

  printf("\n\tnum_obj    -->\t%d\n",    sim_params->num_obj );
  printf("\tr_eff      -->\t%2.2f\n", sim_params->r_eff );
  if( sim_params->num_obj == 2 ) printf("\tr_pair      -->\t%2.2f\n", sim_params->r_pair );
  printf("\tch_objects -->\t%2.2f\n", sim_params->ch_objects );
  printf("\tch_distype -->\t%c\n",    sim_params->ch_distype[0] );
  printf("\tseed       -->\t%ld\n",    sim_params->seed );
  if( sim_params->u_x + sim_params->u_y + sim_params->u_z > 0.0 ){
    printf("\tu_ampl_x   -->\t%2.2f\n", sim_params->u_x );
    printf("\tu_ampl_y   -->\t%2.2f\n", sim_params->u_y );
    printf("\tu_ampl_z   -->\t%2.2f\n", sim_params->u_z );
  }
  printf("\n\trho0       -->\t%2.2f\n", sim_params->rho0 );
  printf("\tnu         -->\t%2.4f\n", sim_params->nu );
  printf("\tkbt        -->\t%2.2f\n", sim_params->kbt );
  printf("\tBjerrun length     -->\t%2.2f\n", sim_params->ch_bjl );
  printf("\tDebye length       -->\t%2.2f\n\n", sim_params->ch_lambda );

  printf("\tD_therm    -->\t%e\n", sim_params->D_therm );
  printf("\tD_plus     -->\t%2.4f\n", sim_params->D_plus  );
  printf("\tD_minus    -->\t%2.4f\n", sim_params->D_minus );
  printf("\tMax %% fluxes equilibrating       -->\t%2.4f\n", sim_params->max_eq_fluxes );
  printf("\tPoisson accuracy (SOR)           -->\t%e\n", sim_params->poisson_acc );
  printf("\tDiffusion accuracy (Smolukoski)  -->\t%e\n",    sim_params->diffusion_acc );

  printf("\n\tMultisteps -->\t%d\n",    sim_params->multisteps );
  if( sim_params->e_slope_x > 0.0 ) printf("\n\te_slope_x  -->\t%2.2f\n\n", sim_params->e_slope_x );
  if( sim_params->e_slope_y > 0.0 ) printf("\n\te_slope_y  -->\t%2.2f\n\n", sim_params->e_slope_y );
  if( sim_params->e_slope_z > 0.0 ) printf("\n\te_slope_z  -->\t%2.2f\n\n", sim_params->e_slope_z );
  // printf("\tFields accuracy    -->\t%.8f\n", sim_params->f_accuracy );
  printf("\toutfreq            -->\t%d\n",    sim_params->outfreq );
  printf("\toutput             -->\t%s\n",    sim_params->output );
  printf("\tInfile Colloids    -->\t%s\n",    sim_params->inf_colloids );
  printf("\n\tMoving Colloids    -->\t%d\n",    sim_params->move );
  printf("\n");
  return 1;
}

void lookup_param( char * p_name, param_t * table_params, char * p_value )
{
  int i;
  char * t_name;
  char * t_value;
  for( i = 0; i < TABSIZ; i++ ){
    param_t param = table_params[i];
    t_name = (char *) &param.name;
    t_value = (char *) &param.value;
    //printf("Trying to compare t_name %s with p_name %s\n", t_name,  p_name);
    //if( !strcmp(p_name, t_name) ){ printf("%s ++> %s\n", t_name, t_value); strcpy(p_value, t_value); return; }
    if( !strcmp(p_name, t_name) ){ strcpy(p_value, t_value); return; } 
  }
  printf( "Not found simulation parameter %s....Dying\n", p_name );
  exit( 2 );
}


