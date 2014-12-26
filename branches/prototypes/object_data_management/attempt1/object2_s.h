/* Here we have an object maintaining
 *    (1) its own data (object2_data_t)
 *    (2) a reference to object1 (obj1)
 *    (3) its own target instance (object2_t * target) */

#ifndef OBJECT2_S_H
#define OBJECT2_S_H

#include "object2.h"

struct object2_data_s {
  int datum;
};

struct object2_s {
  object1_t * obj1;         /* Host reference to object1 */
  object2_data_t * data;    /* Own data (host) */
  object2_t * target;       /* Host pointer -> device memory instance */
};

#endif
