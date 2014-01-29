#!/usr/bin/awk -f

##############################################################################
#
#  awk-fp-diff.sh
#
#  This is a 'diff' script which allows differences in floating
#  point numbers to within a tolerance.
#
#  This is to allow regression tests to pass if there is only
#  a round-off error (which they would not with normal 'diff').
#
#  Edinburgh Soft Matter and Statistical Physics Group and
#  Edinburgh Parallel Computeing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#
##############################################################################

BEGIN {

  # "When the music stops... begin!"

  if (ARGC != 3) {
    # note ARGV[0] will be /usr/bin/awk
    print "usage: awk-fp-diff.sh file1 file2"
    exit -1
  }

  # There a a number of global objects in use, including:
  #
  # TOLERANCE  the floating point tolerance
  # files1[]   lines of file 1 (1..nlines1 with file1[0] the filename)
  # files2[]   lines of file 2 (1..nlines2 with file2[0] the filename)
  # lcslen[,]  lowest common subsequence array for diff algorithm 

  TOLERANCE = 1.0e-12
  nlines1 = 0
  nlines2 = 0
  file1[0] = ARGV[1]
  file2[0] = ARGV[2]
}

{
  # FNR is the record counter for current file
  # NR is total number of records read

  if (FNR == NR) {
    # For first file, store the lines. Note the indexing is natural
    # numbers: first line is file1[1]
    file1[FNR] = ""
    file1[FNR] = $0
    nlines1 += 1
  }
  else {
    # Second file
    file2[FNR] = ""
    file2[FNR] = $0
    nlines2 += 1
  }
}

END {
  fp_lcs_compute()
  fp_print_diff(nlines1, nlines2)
}

###############################################################################
#
#  fp_lines_equal
#
#  If two lines of text are the same, 1 is returned. If they differ
#  zero is returned.
#
#  The lines are parsed to see if they have floating point
#  content. "The same" for such content is then related to TOLERANCE.
#
###############################################################################

function fp_lines_equal(line1, line2) {

  if (line1 == line2) return 1; 

  # OK, the lines don't match. Can this be attributed to a floating
  # point mismatch at the level of TOLERANCE?

  # Parse tokens in each line; must have same number or no match

  nt1 = split(line1, tokens1, " ")
  nt2 = split(line2, tokens2, " ")

  if (nt1 != nt2) return 0;

  # Check corresponding tokens

  for (it = 1; it <= nt1; it++) {

      if (tokens1[it] == tokens2[it]) continue
      
      fp1 = matches_floating_point(tokens1[it])
      fp2 = matches_floating_point(tokens2[it])

      if (fp1 && fp2) {
	  # two floating point numbers
	  if (fp_differ(tokens1[it], tokens2[it])) return 0
      }
      else {
	  # other combinations of strings
	  if (tokens1[it] != tokens2[it]) return 0
      }
  }

  return 1;
}

##############################################################################
#
#  See if a token matches floating point output
#  usually, but not necessarily, produced by C format "%14.7e"
#
#  See, e.g., http://www.regular-expressions.info/floatingpoint.html
#
##############################################################################

function matches_floating_point(string) {

    if (string ~ /[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?/) return 1

    return 0
}

##############################################################################
#
#  Return 0 if two floating point numbers are equal to within a tolerance
#         1 if they differ
#
##############################################################################

function fp_differ(sx, sy) {

    delta = sx - sy
    if (delta < 0.0) delta = -delta
    if (delta < TOLERANCE) return 0 

    return 1
}

##############################################################################
#
#  fp_lcs_compute
#
#  The lowest common subexpression (LCS) algorithm for diff.
#  The wikipidea page for LCS is the source of this!
#
#  The diff is based on fp_lines_differ()
#
##############################################################################

function fp_lcs_compute() {

    start = 1
    nfile1_end = nlines1
    nfile2_end = nlines2

    # Remove matching lines at beginning

    while (start <= nfile1_end && start <= nfile2_end \
	   && file1[start] == file2[start]) start += 1

    while (start <= nfile1_end && start <= nfile2_end \
	   && file1[nfile1_end] == file2[nfile2_end]) {
      nfile1_end -= 1
      nfile2_end -= 1
    }

    for (i = 0; i <= nfile1_end; i++) lcslen[i "," start-1] = 0
    for (j = 0; j <= nfile2_end; j++) lcslen[start-1 "," j] = 0

    for (i = start; i <= nfile1_end; i++) {
	for (j = start; j <= nfile2_end; j++) {
	    if (fp_lines_equal(file1[i], file2[j])) {
		lcslen[i "," j] = lcslen[i-1 "," j-1] + 1
	    }
	    else {
		# max function
		if (lcslen[i "," j-1] >= lcslen[i-1 "," j]) {
		    lcslen[i "," j] = lcslen[i "," j-1]
		}
		else {
		    lcslen[i "," j] = lcslen[i-1 "," j]
		}
	    }
	}
    }

    # We have the lcslen array
}

##############################################################################
#
#  fp_print_diff
#
#  A recursive function which examines the lcslen array to determine
#  the 'diff'-style output.
#
#  The routine should be invoked with i = nlines1 and j = nlines2.
#
#  Note on usual diff output format, which has three cases e.g.,
#   1. 8a12,15   => Append lines 12-15 of file 2 after line 8 of file 1
#   2. 5,7c8,10  => Change lines 5-7 of file 1 to read as 8-10 of file 2
#   3. 5.7d3     => Delete lines 5-7 of file 1
#
#  We haven't atempted these 'change commands' here.
#
###############################################################################

function fp_print_diff(i, j) {

    if (i > 0 && j > 0 && fp_lines_equal(file1[i], file2[j])) {
	fp_print_diff(i-1,j-1)
	# print shared lines here, if required
    }
    else if (j > 0 && (i == 0 || lcslen[i "," j-1] >= lcslen[i-1 "," j])) {
	fp_print_diff(i, j-1)
	print j, "> " file2[j]
    }
    else if (i > 0 && (j == 0 || lcslen[i "," j-1] < lcslen[i-1 "," j])) {
	fp_print_diff(i-1, j)
	print i, "< " file1[i]
    }
    else {
	# print "Start here, or, end the recursion"
    }

}
