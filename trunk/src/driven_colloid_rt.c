
#include <assert.h>

#include "driven_colloid.h"
#include "driven_colloid_rt.h"

/*****************************************************************************
 *
 *  driven_colloid_runtime
 *
 *****************************************************************************/

int driven_colloid_runtime(pe_t * pe, rt_t * rt) {

  int n;
  double f0 = 0.0;

  assert(pe);
  assert(rt);

  n = rt_double_parameter(rt, "driving_force_magnitude", &f0);
  
  if (n == 1)driven_colloid_fmod_set(f0);

  if(is_driven()){
    f0 = driven_colloid_fmod_get();
    pe_info(pe, "\n");
    pe_info(pe, "Colloid driving force magnitude: %12.5e\n", f0);
  }
  return 0;
}
