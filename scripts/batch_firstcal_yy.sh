#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -o grid_output
#$ -e grid_output
#$ -l paper
#$ -l h_vmem=8G

ARGS=`pull_args.py $*`
HERA_CALFILE=hsa7458_v000
EX_ANTS_Y=81
#EX_ANTS_Y=

# get extra optional parameters
observer="Zaki Ali"
cd ~/src/heracal/
git_origin_cal=`git remote -v | grep origin | grep fetch`
git_hash_cal=`git rev-parse HEAD`
cd -

for f in ${ARGS}; do 
    echo ~/src/heracal/scripts/firstcal.py ${f} -p yy --ex_ants=${EX_ANTS_Y} -C ${HERA_CALFILE} --observer=${observer} --git_origin_cal=${git_origin_cal} --git_hash_cal=${git_hash_cal}
    ~/src/heracal/scripts/firstcal.py ${f} -p yy --ex_ants=${EX_ANTS_Y} -C ${HERA_CALFILE} --observer=${observer} --git_origin_cal=${git_origin_cal} --git_hash_cal=${git_hash_cal}

done
