
#include <assert.h>
#include <stdio.h>

int main(int argc, char *argv[]) {

#ifdef __INTEL_COMPILER

  /* For a list of MACROs defined try e.g, "icc -E -dM - < /dev/null" */

  printf("__INTEL_COMPILER %d\n", __INTEL_COMPILER);
  printf("__VERSION__ %s\n", __VERSION__);
  printf("__INTEL_COMPILER_BUILD_DATE %d\n", __INTEL_COMPILER_BUILD_DATE);
  printf("__INTEL_COMPILER %d\n", __INTEL_COMPILER);
  printf("__ICC %d\n", __ICC);
  printf("__INTEL_COMPILER_UPDATE %d\n", __INTEL_COMPILER_UPDATE);

#endif

#ifdef __GNUC__

  /* Note icc also defines __GNUC__ */
  /* List of MACROs from command line, e.g., "gcc -E -dM - < /dev/null" */

  printf("__GNUC__ %d\n", __GNUC__);
  printf("__GNUC_MINOR__ %d\n", __GNUC_MINOR__);
  printf("__GNUC_PATCHLEVEL__ %d\n", __GNUC_PATCHLEVEL__);
  printf("__VERSION__ %s\n", __VERSION__);

#endif

#ifdef _CRAYC

  printf("_CRAYC %d\n", _CRAYC);
  printf("_RELEASE %d\n", _RELEASE);
  printf("_RELEASE_MINOR %d\n", _RELEASE_MINOR);
  printf("_RELEASE_STRING %s\n", _RELEASE_STRING);

#endif

  return 0;
}

