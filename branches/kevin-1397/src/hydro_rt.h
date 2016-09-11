/*****************************************************************************
 *
 *  hydro_rt.h
 *
 *****************************************************************************/

#ifndef HYDRO_RT_H
#define HYDRO_RT_H

#include "pe.h"
#include "runtime.h"
#include "coords.h"
#include "leesedwards.h"
#include "hydro.h"

int hydro_rt(pe_t * pe, rt_t * rt, cs_t * cs, lees_edw_t * le,
	     hydro_t ** phydro);

#endif
