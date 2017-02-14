/*****************************************************************************
 *
 *  leesedwards_rt.c
 *
 *****************************************************************************/

#include <assert.h>
#include "leesedwards_rt.h"

/*****************************************************************************
 *
 *  lees_edw_init_rt
 *
 *****************************************************************************/

int lees_edw_init_rt(rt_t * rt, lees_edw_info_t * info) {

  int key;

  assert(rt);
  assert(info);

  info->nplanes = 0;
  info->type = LE_SHEAR_TYPE_STEADY;

  rt_int_parameter(rt, "N_LE_plane", &info->nplanes);
  rt_double_parameter(rt, "LE_plane_vel", &info->uy);

  key = rt_int_parameter(rt, "LE_oscillation_period", &info->period);
  if (key) info->type = LE_SHEAR_TYPE_OSCILLATORY;

  rt_int_parameter(rt, "LE_time_offset", &info->nt0);

  return 0;
}
