
struct decomp_comms {
  int *dest_proc;
  int  send_count;
  int *send_subsizes;
  MPI_Datatype *send_subarray;

  int *recv_proc;
  int recv_count;
  int *recv_subsizes;
  MPI_Datatype *recv_subarray;

};

void decomp_init();
void decomp_cart_to_pencil(double *in_array, double *out_array);
void decomp_pencil_to_cart(double *in_array, double *out_array);
int index_3d_f (int x, int y, int z, int size[]);
int index_3d_c (int x, int y, int z, int size[]);
void decomp_pencil_sizes(int size[3], int ip);
void decomp_pencil_starts(int start[3], int ip);
int decomp_fftarr_size();
void decomp_finish();

