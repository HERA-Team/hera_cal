#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -e grid_output
#$ -o grid_output
#$ -l paper
#$ -l h_vmem=12G

HERA_CAL_FILE=hsa7458_v000
POL_VAL=""

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
    echo ~/src/heracal/scripts/omni_run.py -C ${HERA_CAL_FILE} -p $POL_VAL --ex_ants=`cat ${f}.badants.txt` ${f} --firstcal="${f}.firstcal.fits"
    ~/src/heracal/scripts/omni_run.py -C ${HERA_CAL_FILE} -p $POL_VAL --ex_ants=`cat ${f}.badants.txt` ${f} --firstcal="${f}.firstcal.fits"
done
