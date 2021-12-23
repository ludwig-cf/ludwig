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
 *  (c) 2010-2021 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "runtime.h"

#define NKEY_LENGTH 128           /* Maximum key / value string length */
#define NKEY_MAX    1024          /* Prevent buffer overflow in keys */

/* Relevant key/value separators (include \n to strip them out) */
static const char * sep_ = " \t\n";

typedef struct key_pair_s key_pair_t;

struct key_pair_s {
  char key[NKEY_LENGTH];
  char val[NKEY_LENGTH];
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
static int rt_is_valid_key_pair(rt_t * rt, const char * line, int lineno);
static int rt_look_up_key(rt_t * rt, const char * key, char * value);
static int rt_free_keylist(key_pair_t * key);
static int rt_vinfo(rt_t * rt, rt_enum_t lv, const char * fmt, ...);

static int rt_is_valid_token(const char * token);
static int rt_line_count_tokens(const char * line);
static int rt_key_value(const char * line, char ** key, char ** val);
static int rt_add_key_value_pair(rt_t * rt, const char * key,
				 const char * val, int lineno);

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
	}
	if (rt->nkeys > NKEY_MAX) {
	  pe_fatal(rt->pe, "Too many keys! Increase NKEY_MAX %d\n", NKEY_MAX);
	}
      }
    }

    fclose(fp_input);
  }

  strncpy(rt->input_file, input_file_name,
	  strnlen(input_file_name, FILENAME_MAX-1));

  rt_key_broadcast(rt);

  return 0;
}

/*****************************************************************************
 *
 *  rt_add_key_value
 *
 *  Add an extra key value pair (useful for testing)
 *
 *****************************************************************************/

int rt_add_key_value(rt_t * rt, const char * key, const char * value) {

  int nline = 0;
  int added = 0;

  assert(rt);
  assert(key);
  assert(value);

  /* Say the line is just the number of keys */
  nline = rt->nkeys + 1;

  if (rt_is_valid_token(key) && rt_is_valid_token(value)) {
    rt_add_key_value_pair(rt, key, value, nline);
    added = 1;
  }

  return added;
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

  int nkeys_at_root = 0;
  char * packed_keys = NULL;
  MPI_Comm comm;

  assert(rt);
  pe_mpi_comm(rt->pe, &comm);

  /* Broacdcast the number of keys and set up the message. */

  nkeys_at_root = rt->nkeys;
  MPI_Bcast(&nkeys_at_root, 1, MPI_INT, 0, comm);

  if (nkeys_at_root <= NKEY_MAX) {
    packed_keys = (char *) calloc(2*nkeys_at_root*NKEY_LENGTH, sizeof(char));
    assert(packed_keys);
  }

  if (packed_keys == NULL) pe_fatal(rt->pe, "malloc(packed_keys) failed\n");

  /* Pack message */

  if (pe_mpi_rank(rt->pe) == 0) {
    key_pair_t * key = rt->keylist;
    int nk = 0;

    for ( ; key; key = key->next) {
      strncpy(packed_keys + nk*NKEY_LENGTH, key->key, NKEY_LENGTH);
      nk += 1;
      strncpy(packed_keys + nk*NKEY_LENGTH, key->val, NKEY_LENGTH);
      nk += 1;
    }
    assert(nk == 2*rt->nkeys);
  }

  MPI_Bcast(packed_keys, 2*nkeys_at_root*NKEY_LENGTH, MPI_CHAR, 0, comm);

  /* Unpack message and set up the list */

  if (pe_mpi_rank(rt->pe) != 0) {
    for (int nk = 0; nk < nkeys_at_root; nk++) {
      char * key = packed_keys + (2*nk  )*NKEY_LENGTH;
      char * val = packed_keys + (2*nk+1)*NKEY_LENGTH;
      rt_add_key_value(rt, key, val);
    }
    assert(rt->nkeys == nkeys_at_root);
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

int rt_double_parameter_vector(rt_t * rt, const char * key, double v[3]) {

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

int rt_int_parameter_vector(rt_t * rt, const char * key, int v[3]) {

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
 *  rt_int_nvector
 *
 *  Vector of specified length with values 1_2_3_...
 *
 *  Return zero on key present and success.
 *
 *****************************************************************************/

int rt_int_nvector(rt_t * rt, const char * key, int nv, int * v,
		   rt_enum_t level) {

  int ierr = 0;
  int key_present = 0;
  char str_value[NKEY_LENGTH];

  assert(rt);
  assert(v);

  key_present = rt_look_up_key(rt, key, str_value);

  if (!key_present) {
    ierr = -1;
  }
  else {
    /* Tokenize */
    int iread = 0;
    char * token = strtok(str_value, "_");
    while (token) {
      if (sscanf(token, "%d", &v[iread]) != 1) {
	rt_vinfo(rt, level, "Key %s has bad value %s\n", key, token);
	break;
      }
      token = strtok(NULL, "_");
      iread += 1;
    }
    if (iread != nv) {
      rt_vinfo(rt, level, "Vector (key %s) has incorrect length\n", key);
      ierr = -2;
    }
  }

  return ierr;
}

/*****************************************************************************
 *
 *  rt_double_nvector
 *
 *  Vector of given length with values "1.0_2.0_3.0_4.0_..."
 *
 *  Return 0 on key present and success.
 *
 *****************************************************************************/

int rt_double_nvector(rt_t * rt, const char * key, int nv, double * v,
		      rt_enum_t level) {

  int ierr = 0;
  int key_present = 0;
  char str_value[NKEY_LENGTH];
  
  assert(rt);
  assert(v);

  key_present = rt_look_up_key(rt, key, str_value);

  if (!key_present) {
    ierr = -1;
  }
  else {
    /* Tokenize and read values ... */
    int iread = 0;
    char * token = strtok(str_value, "_");
    while (token) {
      if (sscanf(token, "%lf", &v[iread]) != 1) {
	rt_vinfo(rt, level, "Key %s has bad value %s\n", key, token);
	break;
      }
      token = strtok(NULL, "_");
      iread += 1;
    }
    if (iread != nv) {
      rt_vinfo(rt, level, "Vector (key %s) has incorrect length\n", key);
      ierr = -2;
    }
  }

  return ierr;
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
 *  rt_is_valid_token
 *
 *  Some checks on allowable tokens for either keys or values.
 *
 *****************************************************************************/

static int rt_is_valid_token(const char * token) {

  int isvalid = 1;

  assert(token);

  if (rt_line_count_tokens(token) != 1) isvalid = 0;
  if (strncmp(token, "#", 1)      == 0) isvalid = 0;     

  return isvalid;
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

  assert(rt);
  assert(line);

  /* Ignore comments and blank lines */

  if (strncmp("#",  line, 1) == 0) return 0;
  if (strncmp("\n", line, 1) == 0) return 0;

  /* Minimal syntax checks. The user will need to sort these
   * out. */

  if (rt_line_count_tokens(line) < 2) {
    /* This does not look like a key value pair... */
    pe_fatal(rt->pe, "Input file at line %d has %d token:\n %s\n",
	     lineno, rt_line_count_tokens(line), line);
  }
  else {
    /* Check against existing keys for duplicate definitions. */
    char * newkey = NULL;
    char * newval = NULL;

    rt_key_value(line, &newkey, &newval);

    if (strncmp(newval, "#", 1) == 0) {
      pe_info(rt->pe,  "Second token in line must not be a comment\n");
      pe_fatal(rt->pe, "Please check input line at line %d\n", lineno);
    }

    if (rt_key_present(rt, newkey)) {
      pe_info(rt->pe, "Key %s (line %d) is already present\n", newkey, lineno);
      pe_fatal(rt->pe, "Please check the input and remove one or other\n");
    }

    free(newkey);
    free(newval);
  }

  return 1;
}

/*****************************************************************************
 *
 *  rt_line_count_tokens
 *
 *  Count " " separated tokens in line.
 *
 *****************************************************************************/

static int rt_line_count_tokens(const char * line) {

  int ntok = 0;

  assert(line);

  {
    char * copy  = strndup(line, BUFSIZ);
    char * token = strtok(copy, sep_);

    while (token) {
      ntok += 1;
      token = strtok(NULL, sep_);
    }

    free(copy);
  }

  return ntok;
}

/*****************************************************************************
 *
 *  rt_key_value
 *
 *  Parse a single string as space-separated key value.
 *  A new copy of the key and value are returned.
 *
 *  Returns 0 on sucess.
 *
 *****************************************************************************/

static int rt_key_value(const char * line, char ** key, char ** val) {

  int ifail = 0;
  int ntok = rt_line_count_tokens(line);

  if (ntok < 2) {
    ifail = -1;
  }
  else {
    char * copy = strndup(line, BUFSIZ);
    *key = strdup(strtok(copy, sep_));
    *val = strdup(strtok(NULL, sep_));
    free(copy);
  }

  return ifail;
}

/*****************************************************************************
 *
 *  rt_add_key_pair
 *
 *  Put a new key on the list.
 *
 *****************************************************************************/

static int rt_add_key_pair(rt_t * rt, const char * line, int lineno) {

  int ifail = 0;
  char * newkey = NULL;
  char * newval = NULL;

  assert(rt);
  assert(line);

  rt_key_value(line, &newkey, &newval);

  ifail = rt_add_key_value_pair(rt, newkey, newval, lineno);
  
  free(newkey);
  free(newval);

  return ifail;
}

static int rt_add_key_value_pair(rt_t * rt, const char * key,
				 const char * val, int lineno) {
  key_pair_t * pnew = NULL;

  assert(rt);

  pnew = (key_pair_t *) calloc(1, sizeof(key_pair_t));
  assert(pnew);

  if (pnew == NULL) {
    pe_fatal(rt->pe, "calloc(key_pair) failed\n");
  }
  else {
    /* Put the new key at the head of the list. */
    rt->nkeys += 1;

    strncpy(pnew->key, key, NKEY_LENGTH - strnlen(key, NKEY_LENGTH) - 1);
    strncpy(pnew->val, val, NKEY_LENGTH - strnlen(val, NKEY_LENGTH) - 1);

    pnew->is_active = 1;
    pnew->input_line_no = lineno;

    pnew->next = rt->keylist;
    rt->keylist = pnew;
  }

  return 0;
}

/*****************************************************************************
 *
 *  rt_key_present
 *
 *  Is the key present in the list at all?
 *
 *****************************************************************************/

int rt_key_present(rt_t * rt, const char * key) {

  int present = 0;
  key_pair_t * pair = NULL;

  assert(rt);

  for ( ; pair; pair = pair->next) {
    if (strncmp(key, pair->key, NKEY_LENGTH) == 0) {
      present = 1;
      break;
    }
  }
  
  return present;
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
  key_pair_t * pkey;

  assert(rt);

  pkey = rt->keylist;

  for ( ; pkey; pkey = pkey->next) {

    if (strcmp(pkey->key, key) == 0) {
      int len = strnlen(pkey->val, NKEY_LENGTH);
      pkey->is_active = 0;
      key_present = 1;
      strncpy(value, pkey->val, NKEY_LENGTH - len - 1);

      break;
    }
  }

  return key_present;
}

/*****************************************************************************
 *
 *  rt_key_required
 *
 *  Convenience to declare that a given key should be present and
 *  take some action.
 *
 *  If not RT_FATAL, returns 0 if key is present, non-zero on "error".
 *
 *****************************************************************************/

int rt_key_required(rt_t * rt, const char * key, rt_enum_t level) {

  int ierr = 0;
  char value[NKEY_LENGTH] = {};

  assert(rt);
  assert(key);

  ierr = -1 + rt_look_up_key(rt, key, value);

  if (ierr == 0) {
    /* No problem */
  }
  else {
    strncpy(value, key, strnlen(key, NKEY_LENGTH-1));

    /* Information */
    if (level == RT_INFO) {
      pe_info(rt->pe, "The following input key is absent...\n");
      pe_info(rt->pe, "A default value will be used for: %s\n", value);
    }
    /* Fatal */
    if (level == RT_FATAL) {
      pe_info(rt->pe, "The following input key is missing...\n");
      pe_info(rt->pe, "Required key: %s\n", value);
      pe_fatal(rt->pe, "Please check the input and try again.\n");
    }
  }

  return ierr;
}

/*****************************************************************************
 *
 *  rt_info
 *
 *  A filter for messages.
 *
 *****************************************************************************/

static int rt_vinfo(rt_t * rt, rt_enum_t lv, const char * fmt, ...) {

  int rank;

  assert(rt);

  rank = pe_mpi_rank(rt->pe);

  if (lv >= RT_INFO && rank == 0) {

    va_list args;

    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);

    if (lv == RT_FATAL) {
      MPI_Comm comm = MPI_COMM_NULL;
      pe_mpi_comm(rt->pe, &comm);
      printf("Please check input and try again.\n");
      MPI_Abort(comm, 0);
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  rt_report_unused_keys
 *
 *****************************************************************************/

int rt_report_unused_keys(rt_t * rt, rt_enum_t level) {

  int n_unused = 0;

  assert(rt);

  {
    key_pair_t * key = rt->keylist;

    for (; key; key = key->next) {
      if (key->is_active) n_unused += 1;
      if (level == RT_INFO && key->is_active) {
	pe_info(rt->pe, "Warning: key/value present in input but not used:\n");
	pe_info(rt->pe, "(Line %d): %s\n", key->input_line_no, key->key);
      }
    }
  }

  return n_unused;
}
