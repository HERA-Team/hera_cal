#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -e grid_output
#$ -o grid_output
#$ -l paper
#$ -l h_vmem=5G

FILES=`pull_args.py $*`

for f in ${FILES}; do
    echo ~/src/heracal/scripts/omni_apply.py -p xx --omnipath=${f}.fits ${f}
    ~/src/heracal/scripts/omni_apply.py -p xx --omnipath=${f}.fits ${f}
done
