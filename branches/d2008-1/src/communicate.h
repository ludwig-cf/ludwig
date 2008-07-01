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

void  get_input_config_filename(char *, const int);
void  get_output_config_filename(char *, const int);

#endif /* _COM_H */








