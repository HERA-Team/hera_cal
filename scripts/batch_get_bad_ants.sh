#$ -S /bin/bash
#$ -V
#$ -cwd
#$ -e grid_output
#$ -o grid_output
#$ -l paper
#$ -l h_vmem=8G

# init
BAD_ANTS=""
CALFILE=hsa7458_v000


# process command line options
while getopts ":a:" opt; do
    case $opt in
	a)
	    # make value passed in lowercase
	    BAD_ANTS="${OPTARG}"
	    ;;
	\?)
	    echo "Invalid option: -$OPTARG"
	    exit 1
	    ;;
    esac
done
shift $((OPTIND-1))

ARGS=`pull_args.py $*`

for f in ${ARGS}; do
    echo ~/src/heracal/scripts/get_bad_ants.py -C ${CALFILE} --ex_ants=${BAD_ANTS} ${f} --write
    ~/src/heracal/scripts/get_bad_ants.py -C ${CALFILE} --ex_ants=${BAD_ANTS} ${f} --write
done
