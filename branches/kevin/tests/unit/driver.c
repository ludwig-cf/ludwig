
#include "mpi.h"
#include "unit_control.h"

int main(int argc, char ** argv) {

  int ifail = 0;
  control_t * ctrl = NULL;

  int do_ut_lang(control_t * ctrl);
  int do_ut_util(control_t * ctrl);
  int do_ut_pe(control_t * ctrl);
  int do_ut_coords(control_t * ctrl);
  int do_ut_rti(control_t * ctrl);
  int do_ut_fe_symm(control_t * ctrl);
  int do_ut_fe_braz(control_t * ctrl);
  int do_ut_fe_surf(control_t * ctrl);
  int do_ut_fe_electro(control_t * ctrl);
  int do_ut_fe_es(control_t * ctrl);
  int do_ut_lees_edwards(control_t * ctrl);

  MPI_Init(&argc, &argv);

  ifail = control_create(MPI_COMM_WORLD, &ctrl);
  if (argc > 1) control_option_set(ctrl, CONTROL_VERBOSE);

  do_ut_lang(ctrl);
  do_ut_util(ctrl);
  do_ut_pe(ctrl);
  do_ut_coords(ctrl);
  do_ut_rti(ctrl);
  do_ut_lees_edwards(ctrl);
  do_ut_fe_electro(ctrl);
  do_ut_fe_es(ctrl);

  control_free(ctrl);

  MPI_Finalize();

  return 0;
}
