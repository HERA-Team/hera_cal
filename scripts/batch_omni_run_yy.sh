#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -e grid_output
#$ -o grid_output
#$ -l paper
#$ -l h_vmem=12G

FILES=`pull_args.py $*`
HERA_CAL_FILE=hsa7458_v000
EX_ANTS_Y=81
#EX_ANTS_Y=


for f in ${FILES}; do
    echo ~/src/heracal/scripts/omni_run.py -C ${HERA_CAL_FILE} -p yy --ex_ants=${EX_ANTS_Y} ${f} --firstcal="${f}.firstcal.fits"
    ~/src/heracal/scripts/omni_run.py -C ${HERA_CAL_FILE} -p yy --ex_ants=${EX_ANTS_Y} ${f} --firstcal="${f}.firstcal.fits"
done
