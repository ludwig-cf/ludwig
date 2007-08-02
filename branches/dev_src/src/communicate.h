#ifndef _COM_H
#define _COM_H

enum io_type { BINARY, ASCII };

/* struct for parallel IO */
typedef struct{
  int root;                      /* Root PE of current I/O group */
  int n_io;                      /* Number of parallel IO group */
  int size;                      /* Size (in PEs) of each IO group */
  int index;                     /* Index of current IO group (start at 1) */
  int rank;                      /* Rank of PE in IO group */
  char *file_ext;                /* Filename suffix for parallel output */
} IO_Param;

void    COM_init( void );
void    COM_halo( void );
void    COM_halo_phi( void );
int     COM_local_index( int );
void    COM_read_site( char *, void (*func)( FILE * ));
void    COM_write_site( char *, void (*func)( FILE *, int, int ));

char *  get_input_config_filename(const int);
char *  get_output_config_filename(const int);

extern void (*MODEL_write_velocity)( FILE *, int, int );
extern void (*MODEL_write_site)( FILE *, int, int );
extern void (*MODEL_write_phi)( FILE *, int, int );
extern void (*MODEL_read_site)( FILE * );

#endif /* _COM_H */








