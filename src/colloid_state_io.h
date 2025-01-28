/*****************************************************************************
 *
 *  colloid_state_io.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2025 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_COLLOID_STATE_IO_H
#define LUDWIG_COLLOID_STATE_IO_H

#include "colloid.h"

#define COLLOID_BUFSZ (NTOT_VAR*25) /* format 25 char per item */

int colloid_state_io_write_buf(const colloid_state_t * s, char * buf);
int colloid_state_io_write_buf_ascii(const colloid_state_t * s, char * buf);
int colloid_state_io_read_buf(colloid_state_t * s, const char * buf);
int colloid_state_io_read_buf_ascii(colloid_state_t * s, const char * buf);

#endif
