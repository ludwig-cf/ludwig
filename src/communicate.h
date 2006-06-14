#ifndef _COM_H
#define _COM_H

void    COM_init( int, char ** );
void    COM_halo( void );
void    COM_halo_phi( void );
IVector COM_index2coord( int index );
int     COM_local_index( int );
void    COM_read_site( char *, void (*func)( FILE * ));
void    COM_write_site( char *, void (*func)( FILE *, int, int ));

char *  get_input_config_filename(const int);
char *  get_output_config_filename(const int);

extern void (*MODEL_write_site)( FILE *, int, int );
extern void (*MODEL_write_phi)( FILE *, int, int );
extern void (*MODEL_read_site)( FILE * );

#endif /* _COM_H */








