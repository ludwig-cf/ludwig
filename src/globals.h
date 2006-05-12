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

#endif /* _GLOBALS_H */
