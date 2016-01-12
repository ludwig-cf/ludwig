#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "driven_colloid.h"
#include "driven_colloid_rt.h"

/*****************************************************************************
 *
 *  driven_colloid_runtime
 *
 *****************************************************************************/

void driven_colloid_runtime(void) {

  int n;
  double f0 = 0.0;

  n = RUN_get_double_parameter("driving_force_magnitude", &f0);
  
  if (n == 1)driven_colloid_fmod_set(f0);

  if(is_driven()){
    f0 = driven_colloid_fmod_get();
    info("\n");
    info("Colloid driving force magnitude: %12.5e\n", f0);
  }
  return;
}
