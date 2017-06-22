#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -e grid_output
#$ -o grid_output
#$ -l paper
#$ -l h_vmem=8G

ARGS=`pull_args.py $*`

# run xrfi script on omnical calfits files
for f in ${ARGS}; do
    echo ~/src/hera_cal/scripts/omni_xrfi.py ${f}
    ~/src/hera_cal/scripts/omni_xrfi.py ${f}
done
