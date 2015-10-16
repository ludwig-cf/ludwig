##############################################################################
#
#  nightly.sh
#
#  This is the script to run the nightly test, and is specific for
#  local set up.
#
#  Edinburgh Soft Matter and Statisitcal Physics Group and
#  Edinburgh Parallel Computing Centre
#
#  Kevin Stratford (kevin@epcc.ed.ac.uk)
#  (c) 2010-2014 The University of Edinburgh
#
##############################################################################
#!/bin/bash --login

# Despite --login we still need to get appropriate paths, etc:
. /etc/profile
eval `/usr/bin/modulecmd bash load PMPI`


# This is the local directory for the nightly stuff
thisdir=/home/w02/kevin/nightly
cd $thisdir

testdir=ludwig/trunk/tests
summary=$thisdir/ludwig-tests.log

# Log file

record=$thisdir/`date +%F-%T`.log

# Start a new summary file which will overwrite anything present
echo "Summary of $record" > $summary
echo "Test directory is  $testdir" >> $summary

# Checkout the SVN (and send the report to the record)

svn co --username stratford http://ccpforge.cse.rl.ac.uk/svn/ludwig &> $record

# start via bsub (indy0.epcc.ed.ac.uk)

cd $testdir
bsub -o $record -e $record -n 64 -W 600 -q normal -J test-all < test-all.sh

# Wait for the tests to finish, and clean up
# This includes a copy of the summary to a public location
# (the interactive job will wait for the main "test-all" job to
# finish before starting and itself will finish before the copy is attempted)

finish=$(cat <<EOF
 rm -rf ludwig
 awk '!/echo/ && /'TEST\|PASS\|FAIL\|SKIP/'' $record >> $summary
EOF
)

cd $thisdir
bsub -w "done(test-all)" -o $record -e $record -I -q interactive "$finish"

scp -p $summary kevin@garnet:~/html/.

