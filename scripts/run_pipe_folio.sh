#!/bin/bash

# enter directory passed in
cd ${1}

# run pipe for xx files
for f in zen*.xx.HH.uvc; do
    qsub ~/src/hera_cal/scripts/pipeline_task.sh -p xx ${f}
done

# run pipe for yy files
for f in zen*.yy.HH.uvc; do
    qsub ~/src/hera_cal/scripts/pipeline_task.sh -p yy ${f}
done

# go back to previous directory
cd ..
