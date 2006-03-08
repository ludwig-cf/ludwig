/*****************************************************************************
 *
 *  cells.h
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef _CELL_H
#define _CELL_H

void      CELL_init_cells(void);
int       CELL_cell_index(int, int, int);
IVector   CELL_cell_coords(FVector);
void      CELL_destroy_list(int, int, int);
Colloid * CELL_get_head_of_list(int, int, int);
void      CELL_insert_at_head_of_list(Colloid *);
void      CELL_sort_list(int, int, int);
void      CELL_update_cell_lists(void);

#endif
