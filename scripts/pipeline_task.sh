#!/bin/bash

# define bad antennas
EX_ANTS_X=81
EX_ANTS_Y=81

# cal info
HERA_CAL=hsa7458_v000
observer="Zaki Ali"
cd ~/src/heracal/
git_origin_cal=`git remote -v | grep origin | grep fetch`
git_hash_cal=`git rev-parse HEAD`
cd -

# initialize
POL_VAL=""

# get polarization information
for getopts ":p:" opt; do
    case $opt in
	p)
	    POL_VAL=`echo "${OPTARG}" | tr '[:upper:]' '[:lower:]`
	    ;;
	\?)
	    echo "Invalid option: -$OPTARG"
	    exit 1
	    ;;
	:)
	    echo "polarization option requires an argument"
	    exit 1
	    ;;
    esac
done
shift $((OPTIND-1))

# process polarization arguments
if [ "${POL_VAL}" == "" ]; then
    echo "Please pass in a polarization value with the -p options"
    exit 1
fi

if [ "${POL_VAL}" != "xx" ] && [ "${POL_VAL}" != "yy" ]; then
    echo "Polarization value must be 'xx' or 'yy'"
    exit 1
fi

if [ "${POL_VAL}" == "xx" ]; then
    BAD_ANTS=$EX_ANTS_X
elif [ "${POL_VAL}" == "yy" ]; then
    BAD_ANTS=$EX_ANTS_Y
fi

# get file args
ARGS=`pull_args.py $1`

# tick
date

# run badants
# loop several times to provide some fault tolerance
for i in {0..2}; do
    echo "badants iteration $i"
    echo ~/src/heracal/scripts/get_bad_ants.py -C ${CALFILE} --ex_ants=${BAD_ANTS} ${ARGS} --write
    ~/src/heracal/scripts/get_bad_ants.py -C ${CALFILE} --ex_ants=${BAD_ANTS} ${ARGS} --write
    if [ $? -eq 0 ]; then
	break
    fi
done

if [ ! -e ${ARGS}.badants.txt ]; then
    echo "badants did not produce expected output"
    exit 1
fi

# run firstcal
for i in {0..2}; do
    echo "firstcal iteration $i"
    echo ~/src/heracal/scripts/firstcal.py ${ARGS} -p ${POL_VAL} --ex_ants=`cat ${ARGS}.badants.txt` -C ${HERA_CALFILE} --observer=${observer} --git_origin_cal=${git_origin_cal} --git_hash_cal=${git_hash_cal}
    ~/src/heracal/scripts/firstcal.py ${ARGS} -p ${POL_VAL} --ex_ants=`cat ${ARGS}.badants.txt` -C ${HERA_CALFILE} --observer=${observer} --git_origin_cal=${git_origin_cal} --git_hash_cal=${git_hash_cal}
    if [ $? -eq 0]; then
	break
    fi
done

if [ ! -e ${ARGS}.first.calfits ]; then
    echo "firstcal did not produce expected output"
    exit 1
fi

# run omnical
for i in {0..2}; do
    echo "omni_run iteration $i"
    echo ~/src/heracal/scripts/omni_run.py -C ${HERA_CAL_FILE} -p $POL_VAL --ex_ants=`cat ${ARGS}.badants.txt` ${ARGS} --firstcal="${ARGS}.first.calfits" --omnipath=`dirname ${ARGS}`
    ~/src/heracal/scripts/omni_run.py -C ${HERA_CAL_FILE} -p $POL_VAL --ex_ants=`cat ${ARGS}.badants.txt` ${ARGS} --firstcal="${ARGS}.first.calfits" --omnipath=`dirname ${ARGS}`
    if [ $? -eq 0]; then
	break
    fi
done

if [ ! -e ${ARGS}.omni.calfits ]; then
    echo "omni_run did not produce expected output"
    exit 1
fi

# run omni_apply
for i in {0..2}; do
    echo "omni_apply iteration $i"
    echo ~/src/heracal/scripts/omni_apply.py -p $POL_VAL --omnipath=${ARGS}.omni.calfits --extension="O" ${ARGS}
    ~/src/heracal/scripts/omni_apply.py -p $POL_VAL --omnipath=${ARGS}.omni.calfits --extension="O" ${ARGS}
    if [ $? -eq 0 ]; then
	break
    fi
done

if [ ! -e ${ARGS}O ]; then
    echo "omni_apply did not produce expected output"
    exit 1
fi

# run xrfi
for i in {0..2}; do
    echo "xrfi iteration $i"
    echo ~/src/heracal/scripts/omni_xrfi.py ${ARGS}
    ~/src/heracal/scripts/omni_xrfi.py ${ARGS}
    if [ $? -eq 0 ]; then
	break
    fi
done

if [ ! -e ${ARGS}.omni.xrfi.calfits ]; then
    echo "xrfi did not produce expected output"
    exit 1
fi

# run omni_xrfi_apply
for i in {0..2}; do
    echo "omni_xrfi_apply itertaion $i"
    echo ~/src/heracal/scripts/omni_apply.py -p $POL_VAL --omnipath=${f}.omni.xrfi.calfits --extension="OR" ${ARGS}
    ~/src/heracal/scripts/omni_apply.py -p $POL_VAL --omnipath=${f}.omni.xrfi.calfits --extension="OR" ${ARGS}
    if [ $? -eq 0 ]; then
	break
    fi
done

if [ ! -e ${ARGS}OR ]; then
    echo "omni_xrfi_apply did not produce expected output"
    exit 1
fi

echo "processing for ${ARGS} complete"

# tock
date
