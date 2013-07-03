
struct holder {
  int *dest_proc;
  int  send_count;
  int *send_subsizes;
  MPI_Datatype *send_subarray;

  int *recv_proc;
  int recv_count;
  int *recv_subsizes;
  MPI_Datatype *recv_subarray;

};

void initialise_decomposition_swap(int input_proc_dims[]);
void cart_to_pencil(double *send_array, double *recv_array);
void pencil_to_cart(double *end_array, double *final_array);
int index_3d_f (int x, int y, int z, int size[]);
int index_3d_c (int x, int y, int z, int size[]);
void pencil_sizess(int size[3]);
void pencil_starts(int start[3]);

