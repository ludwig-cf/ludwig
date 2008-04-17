
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pe.h"
#include "timer.h"
#include "runtime.h"
#include "coords.h"

#include "utilities.h"
#include "lattice.h"
#include "model.h"
#include "communicate.h"

extern Site * site;
extern double * phi_site;

IO_Param     io_grp;      /* Parameters of current IO group */

int          input_format;     /* Default input format is ASCII */
int          output_format;    /* Default output format is binary */

char         input_config[256];
char         output_config[256];


/* Generic functions for output */

static void (*MODEL_write_rho)( FILE *, int, int );
static void (*MODEL_write_rho_phi)( FILE *, int, int );

/* Generic functions for input */
void (*MODEL_read_site)( FILE * );

void (*MODEL_write_site)( FILE *, int, int );
void (*MODEL_write_phi)( FILE *, int, int );
void (*MODEL_write_velocity)( FILE *, int, int );

static void    MODEL_read_site_asc( FILE * );
static void    MODEL_read_site_bin( FILE * );
static void    MODEL_write_site_asc( FILE *, int, int );
static void    MODEL_write_rho_asc( FILE *, int, int );
static void    MODEL_write_phi_asc( FILE *, int, int );
static void    MODEL_write_rho_phi_asc( FILE *, int, int );
static void    MODEL_write_site_bin( FILE *, int, int );
static void    MODEL_write_rho_bin( FILE *, int, int );
static void    MODEL_write_phi_bin( FILE *, int, int );
static void    MODEL_write_rho_phi_bin( FILE *, int, int );
static void    MODEL_write_velocity_asc( FILE *, int, int );
static void    MODEL_write_velocity_bin( FILE *, int, int );


#ifdef _MPI_

/* MPI-specific functions and variables */

static MPI_Datatype DT_Float_plane_XY;/* MPI Datatype: XY plane of Floats */
static MPI_Datatype DT_Float_plane_XZ;/* MPI Datatype: XZ plane of Floats */
static MPI_Datatype DT_Float_plane_YZ;/* MPI Datatype: YZ plane of Floats */

MPI_Comm     IO_Comm;     /* Communicator for parallel IO groups */
/* MPI tags */
enum{                            
  TAG_HALO_SWP_X_BWD = 100,
  TAG_HALO_SWP_X_FWD,
  TAG_HALO_SWP_Y_BWD,
  TAG_HALO_SWP_Y_FWD,
  TAG_HALO_SWP_Z_BWD,
  TAG_HALO_SWP_Z_FWD,
  TAG_IO
};

#endif /* _MPI_ */


/*---------------------------------------------------------------------------*\
 * void COM_halo_phi( void )                                                 *
 *                                                                           *
 * Halos phi_site to neighbouring PEs -- or if serial code, to its periodic  *
 * image                                                                     *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _MPI_, _TRACE_                                                   *
 *                                                                           *
 * Last Updated: 04/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_halo_phi( void )
{
  int i, j, k;
  int xfac, yfac;
  int N[3];
  
#ifdef _MPI_
  int nyz2, nxy2z2;

  MPI_Request req[4];
  MPI_Status status[4];
  
  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac = N[Z]+2;
  nyz2 = N[Y]*yfac;
  xfac = (N[Y]+2)*yfac;
  nxy2z2 = N[X]*xfac;
  
  /* Swap (YZ) plane: one contiguous block in memory */
  
  /* Particular case where source_PE == destination_PE (although MPI would */
  /* cope with it by itself) */
  if(cart_size(X) == 1)
    {
      for(j=0;j<=N[Y]+1;j++)
	for(k=0;k<=N[Z]+1;k++)
	  {
	    phi_site[             j*yfac+k] = phi_site[N[X]*xfac+j*yfac+k];
	    phi_site[(N[X]+1)*xfac+j*yfac+k] = phi_site[xfac    +j*yfac+k];
	  }
    }
  else
    {   
      MPI_Issend(&phi_site[xfac], 1, DT_Float_plane_YZ,
		 cart_neighb(BACKWARD,X),
		 TAG_HALO_SWP_X_BWD, cart_comm(), &req[0]);
      MPI_Irecv(&phi_site[(N[X]+1)*xfac], 1, DT_Float_plane_YZ,
		cart_neighb(FORWARD,X),
		TAG_HALO_SWP_X_BWD, cart_comm(), &req[1]);
      MPI_Issend(&phi_site[nxy2z2], 1, DT_Float_plane_YZ,
		 cart_neighb(FORWARD,X),
		 TAG_HALO_SWP_X_FWD, cart_comm(), &req[2]);
      MPI_Irecv(&phi_site[0], 1, DT_Float_plane_YZ, cart_neighb(BACKWARD,X),
		TAG_HALO_SWP_X_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status); /* Necessary to propagate corners */
    }
  
  /* Swap (XZ) plane: N[X]+2 blocks of N[Z]+2 sites (stride=(N[Y]+2)*(N[Z]+2)) */
  if(cart_size(Y) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(k=0;k<=N[Z]+1;k++)
	  {
	    phi_site[i*xfac              +k] = phi_site[i*xfac+N[Y]*yfac+k];
	    phi_site[i*xfac+(N[Y]+1)*yfac+k] = phi_site[i*xfac+    yfac+k];
	  }
    }
  else
    {
      MPI_Issend(&phi_site[yfac], 1, DT_Float_plane_XZ,
		 cart_neighb(BACKWARD,Y),
		 TAG_HALO_SWP_Y_BWD, cart_comm(), &req[0]);
      MPI_Irecv(&phi_site[(N[Y]+1)*yfac], 1, DT_Float_plane_XZ,
		cart_neighb(FORWARD,Y),
		TAG_HALO_SWP_Y_BWD, cart_comm(), &req[1]);
      MPI_Issend(&phi_site[nyz2], 1, DT_Float_plane_XZ, cart_neighb(FORWARD,Y),
		 TAG_HALO_SWP_Y_FWD, cart_comm(), &req[2]);
      MPI_Irecv(&phi_site[0], 1, DT_Float_plane_XZ, cart_neighb(BACKWARD,Y),
		TAG_HALO_SWP_Y_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status); /* Necessary to propagate corners */
    }
  
  /* Swap (XY) plane: (N[X]+2)*(N[Y]+2) blocks of 1 site (stride=N[Z]+2) */
  if(cart_size(Z) == 1)
    {
      for(i=0;i<=N[X]+1;i++)
	for(j=0;j<=N[Y]+1;j++)
	  {
	    phi_site[i*xfac+j*yfac      ] = phi_site[i*xfac+j*yfac+N[Z]];
	    phi_site[i*xfac+j*yfac+N[Z]+1] = phi_site[i*xfac+j*yfac+1  ];
	  }
    }
  else
    {
      MPI_Issend(&phi_site[1], 1, DT_Float_plane_XY, cart_neighb(BACKWARD,Z),
		 TAG_HALO_SWP_Z_BWD, cart_comm(), &req[0]);
      MPI_Irecv(&phi_site[N[Z]+1], 1, DT_Float_plane_XY,
		cart_neighb(FORWARD,Z),
		TAG_HALO_SWP_Z_BWD, cart_comm(), &req[1]);
      MPI_Issend(&phi_site[N[Z]], 1, DT_Float_plane_XY, cart_neighb(FORWARD,Z),
		 TAG_HALO_SWP_Z_FWD, cart_comm(), &req[2]);  
      MPI_Irecv(&phi_site[0], 1, DT_Float_plane_XY, cart_neighb(BACKWARD,Z),
		TAG_HALO_SWP_Z_FWD, cart_comm(), &req[3]);
      MPI_Waitall(4,req,status);
    }

#else /* Serial section (or OpenMP as appropriate) */

  TIMER_start(TIMER_HALO_LATTICE);

  get_N_local(N);
  yfac = N[Z]+2;
  xfac = (N[Y]+2)*yfac;
  
  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++) {
      phi_site[i*xfac + j*yfac           ] = phi_site[i*xfac + j*yfac + N[Z]];
      phi_site[i*xfac + j*yfac + N[Z] + 1] = phi_site[i*xfac + j*yfac + 1   ];
    }
    
  for(i=1;i<=N[X];i++)
    for(k=0;k<=N[Z]+1;k++) {
      phi_site[i*xfac                 + k] = phi_site[i*xfac + N[Y]*yfac + k];
      phi_site[i*xfac + (N[Y]+1)*yfac + k] = phi_site[i*xfac +      yfac + k];
    }
    
  for(j=0;j<=N[Y]+1;j++)
    for(k=0;k<=N[Z]+1;k++) {
      phi_site[                j*yfac + k] = phi_site[N[X]*xfac + j*yfac + k];
      phi_site[(N[X]+1)*xfac + j*yfac + k] = phi_site[xfac      + j*yfac + k];
    }
#endif /* _MPI_ */

  TIMER_stop(TIMER_HALO_LATTICE);

  return;
}

/*---------------------------------------------------------------------------*\
 * void COM_init( int argc, char **argv )                                    *
 *                                                                           *
 * Initialises communication routines                                        *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _MPI_, _TRACE_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - argc:    same as the main() routine's argc                              *
 * - argv:    same as the main() routine's argv                              *
 *                                                                           *
 * Last Updated: 06/01/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_init() {

  char tmp[256];

#ifdef _MPI_ /* Parallel (MPI) section */

  int nx2, ny2, nz2, nx2y2, ny2z2;
  int N_sites, colour;
  int N[3];

  get_N_local(N);

  /* Compute extents to define new datatypes */

  nx2 = N[X]+2;
  ny2 = N[Y]+2;
  nz2 = N[Z]+2;
  nx2y2 = nx2*ny2;
  ny2z2 = ny2*nz2;
  N_sites = (N[X]+2)*(N[Y]+2)*(N[Z]+2);


  /* Set-up parallel IO parameters (rank and root) */

  io_grp.n_io = 1; /* Default */
  RUN_get_int_parameter("n_io_nodes", &(io_grp.n_io));

  io_grp.size = pe_size() / io_grp.n_io;

  if((cart_rank()%io_grp.size) == 0) {
    io_grp.root = 1;
  }
  else {
    io_grp.root = 0;
  }

  /* Set-up filename suffix for each parallel IO file */

  io_grp.file_ext = (char *) malloc(16*sizeof(char));

  if (io_grp.file_ext == NULL) fatal("malloc(io_grp.file_ext) failed\n");

  io_grp.index = cart_rank()/io_grp.size + 1;   /* Start IO indexing at 1 */
  sprintf(io_grp.file_ext, ".%d-%d", io_grp.n_io, io_grp.index);

  /* Create communicator for each IO group, and get rank within IO group */
  MPI_Comm_split(cart_comm(), io_grp.index, cart_rank(), &IO_Comm);
  MPI_Comm_rank(IO_Comm, &io_grp.rank);

  /* Set-up datatypes for XY, XZ and YZ planes of Floats (including halos) */
  /* (XY) plane: (N[X]+2)*(N[Y]+2) blocks of 1 Float (stride=N[Z]+2) */
  MPI_Type_vector(nx2y2,1,nz2, MPI_DOUBLE, &DT_Float_plane_XY);
  MPI_Type_commit(&DT_Float_plane_XY);

  /* (XZ) plane: N[X]+2 blocks of N[Z]+2 Floats (stride=(N[Y]+2)*(N[Z]+2)) */
  MPI_Type_hvector(nx2,nz2,ny2z2*sizeof(double),MPI_DOUBLE,&DT_Float_plane_XZ);
  MPI_Type_commit(&DT_Float_plane_XZ);

  /* (YZ) plane: one contiguous block of (N[Y]+2)*(N[Z]+2) Floats */
  MPI_Type_contiguous(ny2z2, MPI_DOUBLE,&DT_Float_plane_YZ);
  MPI_Type_commit(&DT_Float_plane_YZ);

  MPI_Barrier(cart_comm());

#else /* Serial section */

  /* Serial definition of io_grp (used in cio) */
  io_grp.root  = 1;
  io_grp.n_io  = 1;
  io_grp.size  = 1;
  io_grp.index = 0;
  io_grp.rank  = 0;
  io_grp.file_ext = (char *) malloc(16*sizeof(char));
  if (io_grp.file_ext == NULL) fatal("malloc(io_grp.file_ext) failed\n");
  sprintf(io_grp.file_ext, ""); /* Nothing required in serial*/

#endif /* _MPI_ */

  /* Everybody */

  /* I/O */
  strcpy(input_config, "EMPTY");
  strcpy(output_config, "config.out");

  input_format = BINARY;
  output_format = BINARY;

  RUN_get_string_parameter("input_config", input_config, 256);
  RUN_get_string_parameter("output_config", output_config, 256);

  RUN_get_string_parameter("input_format", tmp, 256);
  if (strncmp("ASCII",  tmp, 5) == 0 ) input_format = ASCII;
  if (strncmp("ASCII_SERIAL",  tmp, 12) == 0 ) input_format = ASCII_SERIAL;
  if (strncmp("BINARY", tmp, 6) == 0 ) input_format = BINARY;

  RUN_get_string_parameter("output_format", tmp, 256);
  if (strncmp("ASCII",  tmp, 5) == 0 ) output_format = ASCII;
  if (strncmp("BINARY", tmp, 6) == 0 ) output_format = BINARY;

  /* Set input routines: point to ASCII/binary routine depending on current 
     settings */

  switch (input_format) {
  case BINARY:
    MODEL_read_site     = MODEL_read_site_bin;
    info("Input format is binary\n");
    break;
  case ASCII:
  case ASCII_SERIAL:
    MODEL_read_site     = MODEL_read_site_asc;
    info("Input format is ASCII\n");
    break;
  default:
    fatal("Incorrect input_format (%d)\n", input_format);
  }

  /* Set output routines: point to ASCII/binary routine depending on current 
     settings */

  switch (output_format) {
  case BINARY:
    info("Output format is binary\n");
    MODEL_write_site     = MODEL_write_site_bin;
    MODEL_write_rho      = MODEL_write_rho_bin;
    MODEL_write_phi      = MODEL_write_phi_bin;
    MODEL_write_rho_phi  = MODEL_write_rho_phi_bin;	
    MODEL_write_velocity = MODEL_write_velocity_bin;
    break;
  case ASCII:
    info("Output format is ASCII\n");
    MODEL_write_site     = MODEL_write_site_asc;
    MODEL_write_rho      = MODEL_write_rho_asc;
    MODEL_write_phi      = MODEL_write_phi_asc;
    MODEL_write_rho_phi  = MODEL_write_rho_phi_asc;	
    MODEL_write_velocity = MODEL_write_velocity_asc;
    break;
  default:
    fatal("Incorrect output format (%d)\n", output_format);
  }

}

/*---------------------------------------------------------------------------*\
 * void COM_read_site( char *filename, int (*func)( FILE * ) )               *
 *                                                                           *
 * Reads `something' into each lattice site from filename. The actual        *
 * reading is done by a user supplied function passed as argument            *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - filename:    file from which the sites will be read from                *
 * - func:        pointer to user-supplied routine which will perform the IO *
 *                                                                           *
 * Last Updated: 06/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_read_site( char *filename, void (*func)( FILE * ))
{
#ifdef _MPI_ /* Parallel (MPI) version */
  
  int     i,j,k,io_size,io_index;
  int     N[3], nloc[3];
  IVector N_read;
  char    io_filename[FILENAME_MAX];
  FILE    *fp;
  long    foffset;
  MPI_Status status;
  
  get_N_local(N);

  /* Parallel input:  io_grp.n_io concurrent inputs will take place */
  
  /* Set correct filename from function argument and local rank */
  sprintf(io_filename, "%s%s", filename,io_grp.file_ext);
  if(io_grp.root)
    {
      /* Open file for input */
      if( (fp = fopen(io_filename,"r")) == NULL )
	{
	  fatal("COM_read_site(): could not open data file %s\n", io_filename);
	}

      /* Ensure that simulation parameters are in the header */
      if( fscanf(fp,"%d %d %d %d %d %d %d %d\n\f",&nloc[X],&nloc[Y],&nloc[Z],
		 &N_read.x,&N_read.y,&N_read.z,&io_size,&io_index) != 8 )
	{
	  fatal("COM_read_site(): input file %s not standard\n",io_filename);
	}
      
      if( nloc[X] != N[X] || nloc[Y] != N[Y] || nloc[Z] != N[Z] ||
	  N_total(X) != N_read.x || N_total(Y) != N_read.y ||
	  N_total(Z) != N_read.z )
	  {
	    fatal("COM_read_site(): size of system mismatch\n");
	  }

      /* Check that files are compatible with current parallel IO settings */
      if( io_grp.size != io_size || io_grp.index != io_index )
	{
	  fatal("COM_read_link(): %s is not compatible current parallel IO\n",
		  io_filename);
	}
    } /* end of section for root IO PE */
    
  /* Each PE in the current IO group will read its own data in turn (by rank
     order). The use of blocking receives makes sure they don't `overtake' each
     other. Offset in file is passed using a ring communication over the local
     IO group. All IO groups access their own file concurrently (parallel IO) */
  
    /* Initiate ring communication / synchronisation within local IO group */
  else				 
    {
      /* Wait until previous PE in local group has completed its IO (use
	 blocking receives: similar mechanism to COM_read_link()) */
      MPI_Recv(&foffset, 1, MPI_LONG, io_grp.rank-1, TAG_IO, IO_Comm, &status);
      if((fp=fopen(io_filename,"r")) == NULL)
	{
	  fatal("COM_read_site(): could not open file %s\n", io_filename);
	}
      rewind(fp);

      /* Jumps to the correct location in file */
      fseek(fp, foffset, SEEK_SET);
    }

  /* Read each entry (simply scan all local sites) */
  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++)
      for(k=1;k<=N[Z];k++){
	func(fp);
      }

  foffset = ftell(fp);
  
  /* Pass on current offset (foffset) to next PE in local IO ring */
  if((io_grp.rank+1) != io_grp.size){
    MPI_Ssend(&foffset, 1, MPI_LONG, io_grp.rank+1, TAG_IO, IO_Comm);
  }

#else /* Serial version */

  int     i,j,k,io_size,io_index;
  int     N[3], nloc[3];
  IVector N_read;
  char    io_filename[FILENAME_MAX];
  FILE    *fp;

  get_N_local(N);

  strcpy(io_filename, filename);

  /* Open file for input */
  if( (fp = fopen(io_filename,"r")) == NULL )
    {
      fatal("COM_read_site(): could not open file %s\n", io_filename);
    }
    
  /* Ensure that simulation parameters are in the header */
  if( fscanf(fp,"%d %d %d %d %d %d %d %d\n\f",&nloc[X],&nloc[Y],&nloc[Z],
	     &N_read.x,&N_read.y,&N_read.z,&io_size,&io_index) != 8 )
    {
      fatal("COM_read_site(): input file %s not standard\n",io_filename);
    }

  if( nloc[X] != N[X] || nloc[Y] != N[Y] || nloc[Z] != N[Z] ||
      N_total(X) != N_read.x || N_total(Y) != N_read.y ||
      N_total(Z) != N_read.z )
    {
      fatal("COM_read_site(): size of system mismatch\n");
    }

  /* Read each entry (simply scan all local sites) */
  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++)
      for(k=1;k<=N[Z];k++){
	func(fp);
      }

  fclose(fp);

#endif /* _MPI_ */
}

/*---------------------------------------------------------------------------*\
 * void COM_write_site( char *filename, void (*func)( FILE *, int, int ))    *
 *                                                                           *
 * Writes `something' from each lattice site to filename. The actual         *
 * writing is done by a user supplied function passed as argument            *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: _TRACE_, _MPI_                                                   *
 *                                                                           *
 * Arguments                                                                 *
 * - filename:    file to which the sites will be written to                 *
 * - func:        pointer to user-supplied routine which will perform the IO *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

void COM_write_site( char *filename, void (*func)( FILE *, int, int ))
{
#ifdef _MPI_ /* Parallel (MPI) Section */
  
  int     i,j,k,ind,xfac,yfac,gxfac,gyfac,g_ind,io_size,io_index;
  int     N[3];
  int     offset[3];
  char    io_filename[FILENAME_MAX];
  FILE    *fp;
  int     msg=0;
  MPI_Status status;

  get_N_local(N);
  get_N_offset(offset);

  sprintf(io_filename, "%s%s", filename,io_grp.file_ext);
  io_size = io_grp.size;
  io_index = io_grp.index;
  if(io_grp.root)
    {
      /* Open file for output */
      if( (fp = fopen(io_filename,"w")) == NULL )
	{
	  fatal("COM_write_site(): could not open file %s\n", io_filename);
	}
      
      /* Print header information */
      fprintf(fp,"%d %d %d %d %d %d %d %d\n\f",N[X],N[Y],N[Z],
	      N_total(X),N_total(Y),N_total(Z),io_size,io_index);

    }  /* end of section for root IO PE */
  
  /* Each PE in the current IO group will write its own data in turn (by rank
     order). The use of blocking receives makes sure they don't `overtake' each
     other. All IO groups access their own file concurrently (parallel IO) */
  
  /* Initiate ring communication / synchronisation within local IO group */
  else				/* i.e. non IO root PE */
    {
      MPI_Recv(&msg, 1, MPI_INT, io_grp.rank-1, TAG_IO, IO_Comm, &status);
      if((fp=fopen(io_filename,"a+b")) == NULL)
	{
	  fatal("COM_write_site(): could not open file %s\n", io_filename);

	}
    }

  /* Write all local sites */

  xfac  = (N[Y]+2)*(N[Z]+2);
  yfac  = (N[Z]+2);

  gxfac = (N_total(Y) + 2)*(N_total(Z) + 2);
  gyfac = (N_total(Z) + 2);

  for(i=1;i<=N[X];i++)
    for(j=1;j<=N[Y];j++)
      for(k=1;k<=N[Z];k++)
	{
	  /* local index */
	  ind = i*xfac + j*yfac + k;
	  
	  /* global index */
	  g_ind = ((i + offset[X])*gxfac + (j + offset[Y])*gyfac + 
		   (k + offset[Z]));
	  func(fp,ind,g_ind);
	}
  fclose(fp);

  /* Inform next PE in local IO ring that it can now proceed with its own IO */
  if((io_grp.rank+1) != io_grp.size)
    MPI_Ssend(&msg, 1, MPI_INT, io_grp.rank+1, TAG_IO, IO_Comm);
  
#else /* Serial version */
  
  int     i,j,k,ind,xfac,yfac,io_size,io_index;
  int     N[3];
  char    io_filename[FILENAME_MAX];
  FILE    *fp;
  
  get_N_local(N);
  xfac = (N[Y]+2)*(N[Z]+2);
  yfac = (N[Z]+2);

  strcpy(io_filename, filename);
  io_size = 1; 
  io_index = 1;
  
  /* Open file for output */
  if( (fp = fopen(io_filename,"w")) == NULL )
    {
      fatal("COM_write_site(): could not open file %s\n", io_filename);
    }
  
  /* Print header information */

  fprintf(fp,"%d %d %d %d %d %d %d %d\n\f",N[X],N[Y],N[Z],N[X],N[Y],N[Z],
	  io_size, io_index);
  
  /* Write all sites */
  for(i=1; i<=N[X]; i++)
    for(j=1; j<=N[Y]; j++)
      for(k=1; k<=N[Z]; k++)
	{
	  /* local (and global) index */
	  ind = i*xfac + j*yfac + k;
	  
	  /* in the serial implementation, local index == global index */
	  func(fp,ind,ind);
	}
  
  fclose(fp);
  
#endif /* _MPI_ */
}

/*---------------------------------------------------------------------------*\
 * int COM_local_index( int g_ind )                                          *
 *                                                                           *
 * Translates a global index g_ind into a local index ind                    *
 * Returns the local index                                                   *
 *                                                                           *
 * Version: 2.0                                                              *
 * Options: none                                                             *
 *                                                                           *
 * Arguments                                                                 *
 * - index: local index                                                      *
 *                                                                           *
 * Last Updated: 15/07/2002 by JCD                                           *
\*---------------------------------------------------------------------------*/

int COM_local_index( int g_ind )
{
  int coordx, coordy, coordz;
  int N[3];
  int offset[3];
  int g_xfac, g_yfac, ind;

  get_N_local(N);
  get_N_offset(offset);

  g_yfac = (N_total(Z) + 2);
  g_xfac = (N_total(Y) + 2) * g_yfac;
  
  coordx =  g_ind / g_xfac;
  coordy = (g_ind % g_xfac) / g_yfac;
  coordz =  g_ind % g_yfac;
  
  coordx -= offset[X];
  coordy -= offset[Y];
  coordz -= offset[Z];
  
  ind = coordx*(N[Y]+2)*(N[Z]+2) + coordy*(N[Z]+2) + coordz;
  
  return(ind); 
}

/*----------------------------------------------------------------------------*/
/*!
 * Reads distribution function into site s from stream fp (ASCII input)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: FILE *fp: pointer for stream
 *- \c Returns:   void
 *- \c Buffers:   invalidates .halo, .rho, .phi, .grad_phi
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_read_site(), MODEL_read_site_bin(), COM_read_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_read_site_asc( FILE *fp )
{
  int i,ind,g_ind;
  
  if( fscanf(fp,"%d",&g_ind) != EOF )
    {
      /* Get local index */
      ind = COM_local_index( g_ind );

      for(i=0; i<NVEL; i++)
	{
	  if( fscanf(fp,"%lg %lg",&site[ind].f[i],&site[ind].g[i]) == EOF )
	    {
	      fatal("MODEL_read_site_asc(): read EOF\n");
	    }
	}
    }
  else
    {
      fatal("MODEL_read_site_asc(): read EOF\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Reads distribution function into site s from stream fp (binary input)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: FILE *fp: pointer for stream
 *- \c Returns:   void
 *- \c Buffers:   invalidates .halo, .rho, .phi, .grad_phi
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_read_site(), MODEL_read_site_asc(), COM_read_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_read_site_bin( FILE *fp )
{
  int ind,g_ind;
  
  if( fread(&g_ind,sizeof(int),1,fp) != 1 )
    {
      fatal("MODEL_read_site_bin(): couldn't read index\n");
    }
  
  /* Convert to local index */
  ind = COM_local_index( g_ind );
  
  if( fread(site+ind,sizeof(Site),1,fp) != 1 )
    {
      fatal("MODEL_read_site_bin(): couldn't read site\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes distribution function from site s with index ind to stream fp (ASCII
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_site(), MODEL_write_site_bin(), COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_site_asc( FILE *fp, int ind, int g_ind )
{
  int i;

  fprintf(fp,"%d ",g_ind);

  /* Write site information to stream */
  for(i=0; i<NVEL; i++)
    {
      if( fprintf(fp,"%g %g ",site[ind].f[i],site[ind].g[i]) < 0 )
	{
	  fatal("MODEL_write_site_asc(): couldn't write site data\n");
	}
    }
  fprintf(fp,"\n");
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) from site s with index ind to stream fp (ASCII output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho(), MODEL_write_rho_bin(), COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_asc( FILE *fp, int ind, int g_ind )
{
  double rho;    

  rho = get_rho_at_site(ind);

  /* Write density information to stream */
  if( fprintf(fp,"%d %lg\n",g_ind,rho) < 0 )
    {
      fatal("MODEL_write_rho_asc(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes composition (phi) from site s with index ind to stream fp (ASCII 
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_phi(), MODEL_write_phi_bin(), COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_phi_asc( FILE *fp, int ind, int g_ind )
{
  double phi;    

  phi = get_phi_at_site(ind);

  /* Write composition information to stream */
  if( fprintf(fp,"%d %lg\n",g_ind,phi) < 0 )
    {
      fatal("MODEL_write_phi_asc(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) and composition (phi) data from site s with index ind
 * to stream fp (ASCII output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho_phi(), MODEL_write_rho_phi_bin(), 
 *                COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_phi_asc( FILE *fp, int ind, int g_ind )
{
  double rho, phi;    

  rho = get_rho_at_site(ind);
  phi = get_phi_at_site(ind);

  /* Write density and composition information to stream */
  if( fprintf(fp,"%d %lg %lg\n",g_ind,rho,phi) < 0 )
    {
      fatal("MODEL_write_rho_phi_asc(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes distribution function from site s with index ind to stream fp (binary
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_site(), MODEL_write_site_asc(), COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_site_bin( FILE *fp, int ind, int g_ind )
{

  if( fwrite(&g_ind,sizeof(int),1,fp)    != 1  ||
      fwrite(site+ind,sizeof(Site),1,fp) != 1 )
    {
      fatal("MODEL_write_site_bin(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) from site s with index ind to stream fp (binary output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho(), MODEL_write_rho_asc(), COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_bin( FILE *fp, int ind, int g_ind )
{
  double rho;    

  rho = get_rho_at_site(ind);

  if( fwrite(&g_ind,sizeof(int),1,fp) != 1 ||
      fwrite(&rho,sizeof(double),1,fp) != 1 )
    {
      fatal("MODEL_write_rho_bin(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes composition (phi) from site s with index ind to stream fp (binary
 * output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_phi(), MODEL_write_phi_asc(), COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_phi_bin( FILE *fp, int ind, int g_ind )
{
  double phi;    

  phi = get_phi_at_site(ind);

  if( fwrite(&g_ind,sizeof(int),1,fp) != 1 ||
      fwrite(&phi,sizeof(double),1,fp) != 1 )
    {
      fatal("MODEL_write_phi_bin(): couldn't write data\n");
    }
}

/*----------------------------------------------------------------------------*/
/*!
 * Writes density (rho) and composition (phi) data from site s with index ind
 * to stream fp (binary output)
 *
 *- \c Options:   _TRACE_
 *- \c Arguments: 
 *  -# \c FILE \c *fp: pointer for stream
 *  -# \c int \c ind: local index of site s
 *  -# \c int \c g_ind: global index of site s
 *- \c Returns:   void
 *- \c Buffers:   no dependence
 *- \c Version:   2.0
 *- \c Last \c updated: 03/03/2002 by JCD
 *- \c Authors:   JC Desplat
 *- \c See \c also: MODEL_write_rho_phi(), MODEL_write_rho_phi_asc(), 
 *                COM_write_site()
 */
/*----------------------------------------------------------------------------*/

void MODEL_write_rho_phi_bin( FILE *fp, int ind, int g_ind )
{
  double rho,phi;    

  rho = get_rho_at_site(ind);
  phi = get_phi_at_site(ind);

  if( fwrite(&g_ind,sizeof(int),1,fp) != 1 ||
      fwrite(&rho,sizeof(double),1,fp) != 1 ||
      fwrite(&phi,sizeof(double),1,fp) != 1 )
    {
      fatal("MODEL_write_rho_phi_bin(): couldn't write data\n");
    }
}


/*****************************************************************************
 *
 *  MODEL_write_velocity_asc
 *
 *****************************************************************************/

void MODEL_write_velocity_asc( FILE *fp, int ind, int g_ind ) {

  double u[3];
  
  get_velocity_at_lattice(ind, u);

  fprintf(fp,"%d %lg %lg %lg\n", g_ind, u[X], u[Y], u[Z]);

}

/*****************************************************************************
 *
 *  MODEL_write_velocity_bin
 *
 *****************************************************************************/


void MODEL_write_velocity_bin( FILE *fp, int ind, int g_ind ) {

  double u[3];

  get_velocity_at_lattice(ind, u);

  fwrite(&g_ind, sizeof(int), 1, fp);
  fwrite(u + X, sizeof(double), 1, fp);
  fwrite(u + Y, sizeof(double), 1, fp);
  fwrite(u + Z, sizeof(double), 1, fp);

}


/*****************************************************************************
 *
 *  get_output_config_filename
 *
 *  Return conifguration file name stub for time "step"
 *
 *****************************************************************************/

void get_output_config_filename(char * stub, const int step) {

  sprintf(stub, "%s%8.8d", output_config, step);

  return;
}

/*****************************************************************************
 *
 *  get_input_config_filename
 *
 *  Return configuration file name (where for historical reasons,
 *  input_config holds the whole name). "step is ignored.
 *
 *****************************************************************************/

void get_input_config_filename(char * stub, const int step) {

  /* But use this... */
  sprintf(stub, "%s", input_config);

  return;
}

