/*****************************************************************************
 *
 *  regsteer.c
 *
 *  Routines to support RealityGrid steering framework.
 *  The exact details of what is required will change as
 *  required ...
 *
 *  IMPORTANT:
 *  Not support in transitional version, i.e., no -D_REGS_ please..
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/


#ifdef _REGS_

#include "ReG_Steer_Appside.h"


static Int  REGS_control(Int step);
static void REGS_temperature(void);


static char *  _changed_param_labels[REG_MAX_NUM_STR_PARAMS];
static char *  _recvd_cmd_params[REG_MAX_NUM_STR_CMDS];
static int     _iotype_handle[REG_INITIAL_NUM_IOTYPES];
static int     _chktype_handle[REG_INITIAL_NUM_IOTYPES];


static char   _application[REG_MAX_STRING_LENGTH];
static Float  _target;
static Float  _actual;
static Float  _vx2;
static Float  _vy2;
static Float  _vz2;
static Float  _rho2;
static Float  _ghost;
static Float  _temp;
static Int    _ghostmode;
static Float  _viscosity;


/*****************************************************************************
 *
 *  REGS_init
 *
 *  Initialise the steering library. This requires significant
 *  effort:
 *
 *  1) Declare parameters (monitored and steerable)
 *     - the appropriate variables should be available to this routine
 *       (probably global)
 *  2) Declare supported commands
 *  3) Declare checkpoints
 *
 *****************************************************************************/

void REGS_init() {

  int    numCommands = 4;
  int    commands[4];
  int    status;
  int    num_iotypes;
  int    num_chktypes;

  char * param_labels[REG_INITIAL_NUM_PARAMS];
  void * param_ptrs[REG_INITIAL_NUM_PARAMS];
  int    param_types[REG_INITIAL_NUM_PARAMS];
  int    param_strbl[REG_INITIAL_NUM_PARAMS];
  char * param_min[REG_INITIAL_NUM_PARAMS];
  char * param_max[REG_INITIAL_NUM_PARAMS];

  char * iotype_labels[REG_INITIAL_NUM_IOTYPES];
  int    iotype_dirn[REG_INITIAL_NUM_IOTYPES];
  int    iotype_frequency[REG_INITIAL_NUM_IOTYPES];

  int    i;

  /* Malloc steering-specific stuff */

  _changed_param_labels[0] = (char *) malloc(sizeof(char)*
     (REG_MAX_NUM_STR_CMDS + REG_MAX_NUM_STR_PARAMS)*REG_MAX_STRING_LENGTH);

  if (!_changed_param_labels[0]) {
    fprintf(stderr, "---> REGS_init\n");
    fprintf(stderr, "---> malloc failed for _changed_param_labels\n");
    exit(-1);
  }

  for (i = 1; i < REG_MAX_NUM_STR_PARAMS; i++) {
    _changed_param_labels[i] = _changed_param_labels[i-1] +
      REG_MAX_STRING_LENGTH;
  }

  _recvd_cmd_params[0] = _changed_param_labels[REG_MAX_NUM_STR_PARAMS-1]
    + REG_MAX_STRING_LENGTH;

  for(i = 1; i < REG_MAX_NUM_STR_CMDS; i++){
    _recvd_cmd_params[i] = _recvd_cmd_params[i-1] + REG_MAX_STRING_LENGTH;
  }

  /* Initialise steering library */
  Steering_enable(TRUE);


  /* User (and library-supplied) commands */

  commands[0] = REG_STR_STOP;
  commands[1] = REG_STR_PAUSE;
  commands[2] = REG_STR_RESUME;
  commands[3] = REG_STR_DETACH;

  status = Steering_initialize(numCommands, commands);

  if (status != REG_SUCCESS) {
    fprintf(stderr, "---> REGS_init:\n");
    fprintf(stderr, "---> steering initialize failed.\n");
    exit(-1);
  }


  sprintf(_application, "Fluctuating lattice Boltzmann");
  _target = gbl.temperature*gbl.temperature;
  _actual = 0.0;
  _vx2 = 0.0;
  _vy2 = 0.0;
  _vz2 = 0.0;
  _rho2 = 0.0;
  _ghost = 0.0;
  _temp = gbl.temperature;
  _ghostmode = 1;
  _viscosity = gbl.eta;

  param_labels[0] = "Application";
  param_ptrs[0]   = (void *)(_application);
  param_types[0]  = REG_CHAR;
  param_strbl[0]  = FALSE;
  param_min[0]    = "";
  param_max[0]    = "";

  param_labels[1] = "Target temperature";
  param_ptrs[1]   = (void *)(&_target);
  param_types[1]  = REG_DBL;
  param_strbl[1]  = FALSE;
  param_min[1]    = "";
  param_max[1]    = "";

  param_labels[2] = "Actual temperature";
  param_ptrs[2]   = (void *)(&_actual);
  param_types[2]  = REG_DBL;
  param_strbl[2]  = FALSE;
  param_min[2]    = "";
  param_max[2]    = "";

  param_labels[3] = "Mean v_x^2";
  param_ptrs[3]   = (void *)(&_vx2);
  param_types[3]  = REG_DBL;
  param_strbl[3]  = FALSE;
  param_min[3]    = "";
  param_max[3]    = "";

  param_labels[4] = "Mean v_y^2";
  param_ptrs[4]   = (void *)(&_vy2);
  param_types[4]  = REG_DBL;
  param_strbl[4]  = FALSE;
  param_min[4]    = "";
  param_max[4]    = "";

  param_labels[5] = "Mean v_z^2";
  param_ptrs[5]   = (void *)(&_vz2);
  param_types[5]  = REG_DBL;
  param_strbl[5]  = FALSE;
  param_min[5]    = "";
  param_max[5]    = "";

  param_labels[6] = "Mean rho^2";
  param_ptrs[6]   = (void *)(&_rho2);
  param_types[6]  = REG_DBL;
  param_strbl[6]  = FALSE;
  param_min[6]    = "";
  param_max[6]    = "";

  param_labels[7] = "Ghost temperature";
  param_ptrs[7]   = (void *)(&_ghost);
  param_types[7]  = REG_DBL;
  param_strbl[7]  = FALSE;
  param_min[7]    = "";
  param_max[7]    = "";

  param_labels[8] = "Input temperature";
  param_ptrs[8]   = (void *)(&_temp);
  param_types[8]  = REG_DBL;
  param_strbl[8]  = TRUE;
  param_min[8]    = "0.0";
  param_max[8]    = "1.0";

  param_labels[9] = "Ghost mode";
  param_ptrs[9]   = (void *)(&_ghostmode);
  param_types[9]  = REG_INT;
  param_strbl[9]  = TRUE;
  param_min[9]    = "0";
  param_max[9]    = "1";

  param_labels[10] = "Fluid viscosity";
  param_ptrs[10]   = (void *)(&_viscosity);
  param_types[10]  = REG_DBL;
  param_strbl[10]  = TRUE;
  param_min[10]    = "0.0";
  param_max[10]    = "0.16666666667"; /* Avoid over-relaxed regime for moment*/


  status = Register_params(11, param_labels, param_strbl, param_ptrs,
			   param_types, param_min, param_max);

  if (status != REG_SUCCESS) {
    fprintf(stderr, "---> REGS_init:\n");
    fprintf(stderr, "---> failed to register parameters.\n");
    exit(-1);
  }

  /* I/O definitions */

  iotype_labels[0] = "OUR TEST OUTPUT";
  iotype_dirn[0] = REG_IO_OUT;
  iotype_frequency[0] = 0;

  num_iotypes = 1;
  status = Register_IOTypes(num_iotypes, iotype_labels, iotype_dirn, 
                            iotype_frequency, _iotype_handle);

  if (status != REG_SUCCESS) {
    fprintf(stderr, "---> REGS_init:\n");
    fprintf(stderr, "failed to register IO types\n");
    exit(-1);
  }

  /* Register checkpoint emission */

  num_chktypes = 1;
  iotype_labels[0] = "MY_CHECKPOINT";
  iotype_dirn[0] = REG_IO_INOUT;
  iotype_frequency[0] = 0;

  status = Register_ChkTypes(num_chktypes, iotype_labels, iotype_dirn, 
                            iotype_frequency, _chktype_handle);

  if (status != REG_SUCCESS) {
    fprintf(stderr, "---> REGS_init:\n");
    fprintf(stderr, "---> failed to register checkpoint types\n");
    exit(-1);
  }

  return;
}

/*****************************************************************************
 *
 *  REGS_control
 *
 *****************************************************************************/

Int REGS_control(Int nstep) {

  Int  finished = 0; /* return value */

#ifdef _REGS_

  int  num_params_changed;
  int  recvd_cmds[REG_MAX_NUM_STR_CMDS];
  int  num_recvd_cmds;
  int  iohandle;
  int  data_type, data_count;
  int  status;
  int  i;

  char header[BUFSIZ];

  _target = gbl.temperature*gbl.temperature;
  _temp   = gbl.temperature;
  _viscosity = gbl.eta;
  REGS_temperature();

  status = Steering_control(nstep, &num_params_changed, _changed_param_labels,
			    &num_recvd_cmds, recvd_cmds, _recvd_cmd_params);


  if (status == REG_SUCCESS) {

    if (num_params_changed > 0) {
      gbl.temperature = _temp;
      gbl.eta         = _viscosity;
      RAND_reset_temperature(_ghostmode);
    }

    for (i = 0; i < num_recvd_cmds; i++){

      if (recvd_cmds[i] == _iotype_handle[0]) {
	/*
	  sprintf(filename,"steer_test_output-%d",step);
	  COM_write_site(filename,         MODEL_write_phi);
	*/

	if (Emit_start(_iotype_handle[0], step, &iohandle)
	    == REG_SUCCESS ) {

	  if(Make_vtk_header(header, "Some data", gbl.N.x+2, gbl.N.y+2,
			     gbl.N.z+2, 1, 
			     REG_DBL) != REG_SUCCESS) {continue;}


	  data_count = strlen(header);
	  data_type  = REG_CHAR;
	  status = Emit_data_slice(iohandle, data_type, data_count, 
                                       (void *)header);

	  if(status != REG_SUCCESS){
	    printf("Call to Emit_data_slice failed\n");
	    Emit_stop(&iohandle);
	    continue;
	  }

	  data_count = (gbl.N.x+2)*(gbl.N.y+2)*(gbl.N.z+2);
	  data_type  = REG_DBL;
	  status = Emit_data_slice(iohandle, data_type, data_count, 
				   (void *) phi_site);

	  Emit_stop(&iohandle);
	}
      }
      else if(recvd_cmds[i] == REG_STR_STOP){
	finished = 1;
      }
      else if (recvd_cmds[i] == _chktype_handle[0]) {
	/* Check recvd_cmd_params to find out whether
	 * this is input or output of a checkpoint */


	if (strstr(strtok(_recvd_cmd_params[i], " "), "IN")) {
	  /* It's input! */
	}
	else {
	  /* Default to output */
	  /* For example, on output, record the checkpoint. */
	  Record_Chkpt(_chktype_handle[0], "checkpoint1");
	  fprintf(stdout, "MAIN: produce checkpoint here ...\n");
	}
      }
      else if(recvd_cmds[i] == REG_STR_PAUSE){

	if(Steering_pause(&num_params_changed,
			  _changed_param_labels,
			  &num_recvd_cmds,
			  recvd_cmds,
			  _recvd_cmd_params) != REG_SUCCESS){

	  fprintf(stderr, "Steering_pause returned error\n");
	}

	/* Reset loop to parse commands received following the
	   resume/stop command that broke us out of pause */
	i = -1;
      }
    }
  
  }

#endif

  return finished;
}

/*****************************************************************************
 *
 *  REGS_finish
 *
 *****************************************************************************/

void REGS_finish() {


  /* Clean-up the steering library */
  Steering_finalize();

  /* Clean up steering-specific variables */
  free(_changed_param_labels[0]);

  return;
}

/*****************************************************************************
 *
 *  REGS_temperature
 *
 *****************************************************************************/

void REGS_temperature() {

  Float   uvar, uxvar, uyvar, uzvar;
  Float   rhovar, chi2var;
  Float   rfluid;
  Int     i, j, k, index, p;
  Int     xfac, yfac;
  Float   rho, ux, uy, uz, chi2;
  Float   partsum[7], g_sum[7];
  Float   *f;

  IVector N = gbl.N;

  static Float uvar_t   = 0.0;   /* Long term mean for temperature */
  static Float rhovar_t = 0.0;   /* Ditto for the density variance */ 
  static Float steps    = 0.0;   /* sample counter */

  yfac = (N.z + 2);
  xfac = (N.y + 2)*yfac;

  uvar    = 0.0;   /* Total u variance */
  uxvar   = 0.0;   /* u_x variance */
  uyvar   = 0.0;   /* u_y variance */
  uzvar   = 0.0;   /* u_z variance */
  rhovar  = 0.0;   /* (1 - rho)^2  */
  chi2var = 0.0;   /* chi2 ghost mode variance */
  rfluid  = 0.0;   /* Fluid volume */

  /* Single loop: variances are computed assuming the appropriate
   * means are well-behaved (i.e., mean of u is zero, mean of rho
   * is 1) */ 

#pragma omp parallel for default(none)                                                      shared(N, site, site_map, xfac, yfac, cv)                                       private(i, j, k, index, p, f, rho, ux, uy, uz, chi2)                            reduction(+: uvar, uxvar, uyvar, uzvar, rhovar, chi2var, rfluid)
  for (i = 1; i <= N.x; i++)
    for (j = 1; j <= N.y; j++)
      for (k = 1; k <= N.z; k++) {

	index = i*xfac + j*yfac + k;

	if (site_map[index] == FLUID) {
	  f = site[index].f;

	  rho  = f[0];
	  ux   = 0.0;
	  uy   = 0.0;
	  uz   = 0.0;
	  chi2 = 0.0;        /* Ghost mode temperature */

	  for (p = 1; p < NVEL; p++) {
	    rho  += f[p];
	    ux   += f[p]*cv[p][0];
	    uy   += f[p]*cv[p][1];
	    uz   += f[p]*cv[p][2];
	    chi2 += f[p]*cv[p][0]*cv[p][1]*cv[p][2];
	  }

	  ux = ux/rho;
	  uy = uy/rho;
	  uz = uz/rho;

	  uvar    += ux*ux + uy*uy + uz*uz;
	  uxvar   += ux*ux;
	  uyvar   += uy*uy;
	  uzvar   += uz*uz;
	  rhovar  += (1.0 - rho)*(1.0 - rho);
	  chi2var += 9.0*chi2*chi2;
	  rfluid  += 1.0;
	}
      }

#ifdef _MPI_
  partsum[0] = uvar;
  partsum[1] = uxvar;
  partsum[2] = uyvar;
  partsum[3] = uzvar;
  partsum[4] = rhovar;
  partsum[5] = chi2var;
  partsum[6] = rfluid;

  MPI_Reduce(partsum, g_sum, 7, DT_Float, MPI_SUM, 0, Grid_Comm);

  uvar    = g_sum[0];
  uxvar   = g_sum[1];
  uyvar   = g_sum[2];
  uzvar   = g_sum[3];
  rhovar  = g_sum[4];
  chi2var = g_sum[5];
  rfluid  = g_sum[6];
#endif

  rfluid  = 1.0/rfluid;

  uvar    = uvar*rfluid;
  uxvar   = uxvar*rfluid;
  uyvar   = uyvar*rfluid;
  uzvar   = uzvar*rfluid;
  rhovar  = rhovar*rfluid;
  chi2var = chi2var*rfluid;

  _actual = uvar;
  _vx2 = uxvar;
  _vy2 = uyvar;
  _vz2 = uzvar;
  _rho2 = rhovar;
  _ghost = chi2var;

  return;
}

#else /* Provide no-op stubs if no _REGS_ */

void REGS_init() {return;}
void REGS_finish() {return;}

#endif /* _REGS_ */
