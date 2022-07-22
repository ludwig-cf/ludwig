/*****************************************************************************
 *
 *  util_fopen.c
 *
 *  This is a version of fopen() which allows the specification of
 *  something other than the default UMASK for new files, which is
 *  not available via the ANSI C fopen().
 *
 *  UMASK 0644 leads to annoying security alerts from code scanning,
 *  so use 0600 instead.
 *
 *  The allowed modes are "rwa+b". The optional "x" and "e" do
 *  nothing.
 * 
 *  The Unix implementation needs to use open() then fdopen().
 *
 *  A fall-back option which is just a wrapper to standard fopen()
 *  is available.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2022 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>

#include "util_fopen.h"

/*****************************************************************************
 *
 *  util_fopen_default_umask
 *
 *****************************************************************************/

int util_fopen_default_umask(void) {

  return 0600;
}

#ifdef USE_STANDARD_FOPEN

/*****************************************************************************
 *
 *  util_fopen
 * 
 *****************************************************************************/

FILE * util_fopen(const char * path, const char * mode) {

  return fopen(path, mode);
}

#else

#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>

FILE * util_fopen_umask(const char * path, const char * mode, int umask);

/*****************************************************************************
 *
 *  util_fopen
 *
 *  Formally, the umask as argument to open() is of type mode_t (often
 *  size 16 bytes). Here we just say int.
 *
 *****************************************************************************/

FILE * util_fopen(const char * path, const char * mode) {

  int umask = util_fopen_default_umask();

  assert(sizeof(int) >= sizeof(mode_t));

  return util_fopen_umask(path, mode, umask);
}

/*****************************************************************************
 *
 *  util_fopen_umask
 *
 *  "r"   open for reading
 *  "w"   open for writing    [create new file] write at position 0
 *  "a"   open for appending  [create new file] write at end
 *  "r+"  reading and writing
 *  "w+"  reading and writing [create new file] write at position 0
 *  "a+"  reading and writing [create new file] read from start, write at end
 *
 *  Optionally, 'b' may appear, but has no effect.
 *
 *  Standard arguments 'x' and 'e' are not handled.
 *
 *****************************************************************************/

FILE * util_fopen_umask(const char * path, const char * mode, int umask) {

  int flags = 0;                 /* For open() */
  const char * fdmode = NULL;    /* For fopen() */
  FILE * fp = NULL;

  if (!path) goto err_einval;
  if (!mode) goto err_einval;

  /* Verify 'mode' consists of supported characters only. */
  /* 'b' may appear but doesn't have any consequences. */

  if (strlen(mode) != strspn(mode, "rwa+b")) goto err_einval;

  {
    /* Arguments may appear in any order ... */
    int have_r = strchr(mode, 'r') ? 1 : 0;
    int have_w = strchr(mode, 'w') ? 2 : 0;
    int have_a = strchr(mode, 'a') ? 4 : 0;
    int have_p = strchr(mode, '+') ? 8 : 0;
  
    switch (have_r + have_w + have_a + have_p) {
    case 1:
      flags = O_RDONLY;
      fdmode = "r";
      break;
    case 2:
      flags = O_WRONLY | O_CREAT | O_TRUNC;
      fdmode = "w";
      break;
    case 4:
      flags = O_WRONLY | O_CREAT | O_APPEND;
      fdmode = "a";
      break;
    case 9:
      flags = O_RDWR | O_CREAT;
      fdmode = "r+";
      break;
    case 10:
      flags = O_RDWR | O_CREAT | O_TRUNC;
      fdmode = "w+";
      break;
    case 12:
      flags = O_RDWR | O_CREAT | O_APPEND;
      fdmode = "a+";
      break;
    default:
      /* Invalid combination of 'r', 'w', 'a', and '+'. */
      goto err_einval;
    }
  }

  {
    int fd = -1;   /* open() file descriptor */

    if (flags & O_CREAT) {
      /* New file with a umask (of type mode_t) */
      mode_t umode = umask;
      fd = open(path, flags, umode);
    }
    else {
      /* Existing file */
      fd = open(path, flags);
    }

    /* If open() has failed, errno is set by open(). Fail here: */

    if (fd == -1) return NULL;

    fp = fdopen(fd, fdmode);

    if (fp) {
      /* Success */
      return fp;
    }
    else {
      /* Failed. Clean up from open(), but retain fdopen() errno. */

      int fdopen_errno = errno;

      if (flags & O_TRUNC) unlink(path);
      close(fd);
      
      errno = fdopen_errno;
    }
  }

  return NULL;

err_einval:
  errno = EINVAL;
  return NULL;
}

#endif
