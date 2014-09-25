##############################################################################
#
#  nightly.sh
#
#  This is the script to run the nightly test.
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

testdir=ludwig/branches/testcharge/tests
summary=$thisdir/summary-testcharge.log

# Log file

record=$thisdir/`date +%F-%T`.log

# Start a new summary file which will overwrite anything present
echo "Summary of $record" > $summary
echo $SHELL >> $summary
echo $PATH >> $summary

# Checkout the SVN (and send the report to the record)

#svn co --username stratford http://ccpforge.cse.rl.ac.uk/svn/ludwig &> $record

# start via bsub (indy0.epcc.ed.ac.uk)

cd $testdir
bsub -o $record -e $record -n 64 -W 1:00 -q normal -J test-all < test-all.sh

# Wait for the tests to finish, and clean up

cd $thisdir
#bsub -w "done(test-all)" -o $record -e $record -n 1 -q normal "rm -rf ludwig"

