/*****************************************************************************
 *
 *  colloids_Q_tensor.c
 *
 *  Routines dealing with LC anchoring at surfaces.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Juho Lintuvuori (jlintuvu@ph.ed.ac.uk)
 *  (c) 2011 The University of Edinburgh
 *  
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "pe.h"
#include "build.h"
#include "coords.h"
#include "colloids.h"
#include "io_harness.h"
#include "util.h"
#include "model.h"
#include "blue_phase.h"
#include "colloids_Q_tensor.h"
#include "colloids_s.h"
#include "map_s.h"
#include "hydro_s.h"

/* OLD DEFAULTS 
static int anchoring_coll_ = ANCHORING_NORMAL;
static int anchoring_wall_ = ANCHORING_NORMAL;

static __targetConst__ int t_anchoring_coll_ = ANCHORING_NORMAL;
static __targetConst__ int t_anchoring_wall_ = ANCHORING_NORMAL;
*/
