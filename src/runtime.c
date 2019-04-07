/******************************************************************************
 *
 *  runtime.c
 *
 *  Routines which read the input file at run time and provide access
 *  to the parameters specified therein.
 *
 *  The input file is parsed to create a list of key value pairs,
 *  which are stored as strings. The routines here remain completely
 *  agnostic about the meaning of the strings. The key / value pairs
 *  should be space separated, e.g., 
 *
 *  # The temperature of the fluid is
 *  temperature 0.01
 *
 *  Lines starting with #, and blank lines, are disregarded as comments.
 *
 *  The main code can get hold of the appropriate values by querying
 *  the "database" of keys to get their corresponding value. The
 *  relevant keys and types must clearly be known, e.g.,
 *
 *  rt_double_parameter(rt, "temperature", &value);
 *
 *  We demand that the keys are unique, i.e., they only appear
 *  once in the input file.
 *
 *  In parallel, the root process in pe_comm() is responsible
 *  for reading the input file, and the key value pair list is then
 *  broadcast to all other processes.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010-2018 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "runtime.h"

#define NKEY_LENGTH 128           /* Maximum key / value string length */

typedef struct key_pair_s key_pair_t;

struct key_pair_s {
  char key[NKEY_LENGTH];
  int  is_active;
  int  input_line_no;
  key_pair_t * next;
};

struct rt_s {
  pe_t * pe;                      /* Reference to parallel environment */
  int nkeys;                      /* Number of keys */
  char input_file[FILENAME_MAX];  /* Input file name */
  key_pair_t * keylist;           /* Key list */
};

static int rt_add_key_pair(rt_t * rt, const char *, int lineno);
static int rt_key_broadcast(rt_t * rt);
static int rt_is_valid_key_pair(rt_t * rt, const char *, int lineno);
static int rt_look_up_key(rt_t * rt, const char * key, char * value);
static int rt_free_keylist(key_pair_t * key);

/*****************************************************************************
 *
 *  rt_create
 *
 *****************************************************************************/

int rt_create(pe_t * pe, rt_t ** prt) {

  rt_t * rt = NULL;

  assert(pe);

  rt = (rt_t *) calloc(1, sizeof(rt_t));
  assert(rt);
  if (rt == NULL) pe_fatal(pe, "calloc(rt) failed\n");

  rt->pe = pe;
  pe_retain(pe);

  *prt = rt;

  return 0;
}

/*****************************************************************************
 *
 *  rt_free
 *
 *****************************************************************************/

int rt_free(rt_t * rt) {

  assert(rt);

  rt_free_keylist(rt->keylist);
  pe_free(rt->pe);
  free(rt);

  return 0;
}

/*****************************************************************************
 *
 *  rt_free_keylist
 *
 *****************************************************************************/

static int rt_free_keylist(key_pair_t * key) {

  if (key == NULL) return 0;

  if (key->next) {
    rt_free_keylist(key->next);
  }

  free(key);

  return 0;
}

/*****************************************************************************
 *
 *  rt_read_input_file
 *
 *  Read the input file, and construct the list of key value pairs.
 *
 *****************************************************************************/

int rt_read_input_file(rt_t * rt, const char * input_file_name) {

  FILE * fp_input;
  int    nline = 0;
  char   line[NKEY_LENGTH];

  assert(rt);

  /* Read the file and work out number of valid key lines */

  if (pe_mpi_rank(rt->pe) == 0) {

    fp_input = fopen(input_file_name, "r");

    if (fp_input == NULL) {
      pe_fatal(rt->pe, "Input file '%s' not found.\n", input_file_name);
    }
    else {

      while (fgets(line, NKEY_LENGTH, fp_input)) {
	nline += 1;
	/* Look at the line and add it if it's a key. */
	if (rt_is_valid_key_pair(rt, line, nline)) {
	  rt_add_key_pair(rt, line, nline);
	  rt->nkeys += 1;
	}
      }
    }

    fclose(fp_input);
  }

  strncpy(rt->input_file, input_file_name, FILENAME_MAX);

  rt_key_broadcast(rt);

  return 0;
}

/*****************************************************************************
 *
 *  rt_info
 *
 *****************************************************************************/

int rt_info(rt_t * rt) {

  assert(rt);

  pe_info(rt->pe, "Read %d user parameters from %s\n",
	  rt->nkeys, rt->input_file);

  return 0;
}

/*****************************************************************************
 *
 *  rt_key_broadcast
 *
 *  Make the keys available to all MPI processes. As the number of
 *  keys could be quite large, it's worth restricting this to one
 *  MPI_Bcast().
 *
 *****************************************************************************/

static int rt_key_broadcast(rt_t * rt) {

  char * packed_keys;
  int nk = 0;
  MPI_Comm comm;

  assert(rt);
  pe_mpi_comm(rt->pe, &comm);

  /* Broacdcast the number of keys and set up the message. */

  MPI_Bcast(&rt->nkeys, 1, MPI_INT, 0, comm);

  packed_keys = (char *) calloc(rt->nkeys*NKEY_LENGTH, sizeof(char));
  assert(packed_keys);
  if (packed_keys == NULL) pe_fatal(rt->pe, "malloc(packed_keys) failed\n");

  /* Pack message */

  if (pe_mpi_rank(rt->pe) == 0) {
    key_pair_t * key = rt->keylist;

    for ( ; key; key = key->next) {
      strncpy(packed_keys + nk*NKEY_LENGTH, key->key, NKEY_LENGTH);
      nk += 1;
    }
  }

  MPI_Bcast(packed_keys, rt->nkeys*NKEY_LENGTH, MPI_CHAR, 0, comm);

  /* Unpack message and set up the list */

  if (pe_mpi_rank(rt->pe) != 0) {
    for (nk = 0; nk < rt->nkeys; nk++) {
      rt_add_key_pair(rt, packed_keys + nk*NKEY_LENGTH, 0);
    }
  }

  free(packed_keys);

  return 0;
}

/*****************************************************************************
 *
 *  rt_double_parameter
 *
 *  Query the keys for a scalar double matching te given key.
 *
 *****************************************************************************/

int rt_double_parameter(rt_t * rt, const char * key, double * value) {

  int key_present = 0;
  char str_value[NKEY_LENGTH];

  assert(rt);

  key_present = rt_look_up_key(rt, key, str_value);
  if (key_present) *value = atof(str_value);

  return key_present;
}

/*****************************************************************************
 *
 *  rt_int_parameter
 *
 *  Query the keys for a scalar int.
 *
 *****************************************************************************/

int rt_int_parameter(rt_t * rt, const char * key, int * value) {

  int key_present = 0;
  char str_value[NKEY_LENGTH];

  assert(rt);

  key_present = rt_look_up_key(rt, key, str_value);
  if (key_present) *value = atoi(str_value);

  return key_present;
}

/*****************************************************************************
 *
 *  rt_double_parameter_vector
 *
 *  Query keys for a 3-vector of double.
 *
 *****************************************************************************/

int rt_double_parameter_vector(rt_t * rt, const char * key, double v[]) {

  int key_present = 0;
  char str_value[NKEY_LENGTH];

  assert(rt);

  key_present = rt_look_up_key(rt, key, str_value);

  if (key_present) {
    /* Parse the value as a 3-vector of double */
    if (sscanf(str_value, "%lf_%lf_%lf", &v[0], &v[1], &v[2]) != 3) {
      pe_fatal(rt->pe, "Could not parse input key %s as double[3]\n", key);
    }
  }

  return key_present;
}

/*****************************************************************************
 *
 *  run_get_int_parameter_vector
 *
 *  Query keys for a 3-vector of int.
 *
 *****************************************************************************/

int rt_int_parameter_vector(rt_t * rt, const char * key, int v[]) {

  int key_present = 0;
  char str_value[NKEY_LENGTH];

  assert(rt);

  key_present = rt_look_up_key(rt, key, str_value);

  if (key_present) {
    /* Parse the value as a 3-vector of ints */
    if (sscanf(str_value, "%d_%d_%d", &v[0], &v[1], &v[2]) != 3) {
      pe_fatal(rt->pe, "Could not parse input key %s as int[3]\n", key);
    }
  }

  return key_present;
}

/*****************************************************************************
 *
 *  rt_string_parameter
 *
 *  Query the key list for a string. Any truncation is treated as
 *  fatal to prevent problems down the line.
 *
 *****************************************************************************/

int rt_string_parameter(rt_t * rt, const char * key, char * value,
			unsigned int len) {

  int key_present = 0;
  char str_value[NKEY_LENGTH];

  assert(rt);

  key_present = rt_look_up_key(rt, key, str_value);

  if (key_present) {
    /* Just copy the string across */
    if (strlen(str_value) >= len) {
      pe_fatal(rt->pe, "Truncated input string %s\n", key);
    }
    strncpy(value, str_value, len);
  }

  return key_present;
}

/*****************************************************************************
 *
 *  rt_switch
 *
 *  Key value pairs to evaluate to "switch on"
 *
 *  switch  on
 *  switch  yes
 *  switch  1
 *  switch  true
 *
 *  If the key is not there, or it's anything else, it's "off" and
 *  return value is zero.
 *
 *****************************************************************************/

int rt_switch(rt_t * rt, const char * key) {

  int key_present = 0;
  int iswitch = 0;
  char str_value[NKEY_LENGTH];

  assert(rt);

  key_present = rt_look_up_key(rt, key, str_value);

  if (key_present) {
    if (strcmp(str_value, "Yes") == 0) iswitch = 1;
    if (strcmp(str_value, "yes") == 0) iswitch = 1;
    if (strcmp(str_value, "1") == 0) iswitch = 1;
    if (strcmp(str_value, "on") == 0) iswitch = 1;
    if (strcmp(str_value, "true") == 0) iswitch = 1;
  }

  return iswitch;
}

/*****************************************************************************
 *
 *  rt_active_keys
 *
 *  Count up the number of active key / value pairs.
 *
 *****************************************************************************/

int rt_active_keys(rt_t * rt, int * nactive) {

  key_pair_t * key = NULL;

  assert(rt);
  *nactive = 0;

  key = rt->keylist;

  for ( ; key; key = key->next) {
    if (key->is_active) *nactive += 1;
  }

  return 0;
}

/*****************************************************************************
 *
 *  rt_is_valid_key
 *
 *  Checks a line of the input file (one string) is a valid key.
 *
 *  Invalid strings, along with comments introduced via #, and blank
 *  lines return 0.
 *
 *****************************************************************************/

static int rt_is_valid_key_pair(rt_t * rt, const char * line, int lineno) {

  char a[NKEY_LENGTH];
  char b[NKEY_LENGTH];

  if (strncmp("#",  line, 1) == 0) return 0;
  if (strncmp("\n", line, 1) == 0) return 0;

  /* Minimal syntax checks. The user will need to sort these
   * out. */

  if (sscanf(line, "%s %s", a, b) != 2) {
    /* This does not look like a key value pair... */
    pe_fatal(rt->pe, "Please check input file syntax at line %d:\n %s\n",
	     lineno, line);
  }
  else {
    /* Check against existing keys for duplicate definitions. */

    key_pair_t * key = rt->keylist;

    while (key) {

      /* We must compare for exact equality against existing key. */
      sscanf(key->key, "%s ", b);

      if (strcmp(b, a) == 0) {
	pe_info(rt->pe, "At line %d: %s\n", lineno, line); 
	pe_fatal(rt->pe, "Duplication of parameters in input file: %s %s\n",
		 a, b);
      }

      key = key->next;
    }
  }

  return 1;
}

/*****************************************************************************
 *
 *  rt_add_key_pair
 *
 *  Put a new key on the list.
 *
 *****************************************************************************/

static int rt_add_key_pair(rt_t * rt, const char * key, int lineno) {

  key_pair_t * pnew = NULL;

  assert(rt);

  pnew = (key_pair_t *) calloc(1, sizeof(key_pair_t));
  assert(pnew);

  if (pnew == NULL) {
    pe_fatal(rt->pe, "calloc(key_pair) failed\n");
  }
  else {
    /* Put the new key at the head of the list. */

    strncpy(pnew->key, key, NKEY_LENGTH);
    pnew->is_active = 1;
    pnew->input_line_no = lineno;

    pnew->next = rt->keylist;
    rt->keylist = pnew;
  }

  return 0;
}

/*****************************************************************************
 *
 *  rt_look_up_key
 *
 *  Look through the list of keys to find one matching "key"
 *  and return the corrsponding value string.
 *
 *****************************************************************************/

static int rt_look_up_key(rt_t * rt, const char * key, char * value) {

  int key_present = 0;
  char a[NKEY_LENGTH];
  char b[NKEY_LENGTH];
  key_pair_t * pkey;

  assert(rt);

  pkey = rt->keylist;

  for ( ; pkey; pkey = pkey->next) {

    sscanf(pkey->key, "%s %s", a, b);

    if (strcmp(a, key) == 0) {
      pkey->is_active = 0;
      key_present = 1;
      strncpy(value, b, NKEY_LENGTH);
      break;
    }
  }

  return key_present;
}
