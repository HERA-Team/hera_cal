#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -o grid_output
#$ -e grid_output
#$ -l paper
#$ -l h_vmem=8G

HERA_CALFILE=hsa7458_v000
POL_VAL=""

# get extra optional parameters
observer="Zaki Ali"
cd ~/src/heracal/
git_origin_cal=`git remote -v | grep origin | grep fetch`
git_hash_cal=`git rev-parse HEAD`
cd -

# process command line options
while getopts ":p:" opt; do
    case $opt in
	p)
	    # make value passed in lowercase
	    POL_VAL=`echo "${OPTARG}" | tr '[:upper:]' '[:lower:]'`
	    ;;
	\?)
	    echo "Invalid option: -$OPTARG"
	    exit 1
	    ;;
	:)
	    echo "Polarization option requires an argument"
	    exit 1
	    ;;
    esac
done
shift $((OPTIND-1))

ARGS=`pull_args.py $*`

# make sure polarization is valid
if [ "${POL_VAL}" == "" ]; then
    echo "Pass in a polarization value with the -p option"
    exit 1
fi

if [ "${POL_VAL}" != "xx" && "${POL_VAL}" != "yy" ]; then
    echo "polarization value must be 'xx' or 'yy'"
    exit 1
fi

for f in ${ARGS}; do 
    echo ~/src/heracal/scripts/firstcal.py ${f} -p ${POL_VAL} --ex_ants=`cat ${f}.badants.txt` -C ${HERA_CALFILE} --observer=${observer} --git_origin_cal=${git_origin_cal} --git_hash_cal=${git_hash_cal}
    ~/src/heracal/scripts/firstcal.py ${f} -p ${POL_VAL} --ex_ants=`cat ${f}.badants.txt` -C ${HERA_CALFILE} --observer=${observer} --git_origin_cal=${git_origin_cal} --git_hash_cal=${git_hash_cal}

done
