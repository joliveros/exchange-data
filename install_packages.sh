#
#  Original solution via StackOverflow:
#    http://stackoverflow.com/questions/35802939/install-only-available-packages-using-conda-install-yes-file-requirements-t
#

while read requirement; do conda install --yes $requirement; done < requirements-conda.txt
