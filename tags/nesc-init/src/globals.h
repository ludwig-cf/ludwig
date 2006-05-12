#ifndef _GLOBAL_H
#define _GLOBAL_H
/*-------------- All important stuff in these headers ---------------------*/

/* Common typedefs */
#include "utilities.h"

#include "leesedwards.h"

/* Model functions and definitions */
#include "model.h"

/* Colloids */
#include "colloids.h"
#include "interaction.h"
#include "test.h"
#include "wall.h"

/* Communication routines and definitions */
/* Must be included after model.h */
#include "communicate.h"

#include "lattice.h"

/* Random number routines */
#include "ran.h"

/* Graphics functions */
#include "regsteer.h"

/*--------------------- Definition of Global variables --------------------*/

extern Global     gbl;        /* Most global variables live here */
extern Site       *site;      /* The lattice */
extern char       *site_map;  /* Map of full and empty sites */

/*--------------------- End of header file --------------------------------*/
#endif /* _GLOBALS_H */
