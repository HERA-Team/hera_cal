#set up
jd=${1}
echo cd ${jd}
cd ${jd}
echo mkdir grid_output
mkdir grid_output

EX_ANTS_X=81
EX_ANTS_Y=81

# Find the badants. The EX_ANTS gets passed into batch_get_bad_ants
echo badants_yid=qsub -t 1-72 ~/src/hera_cal/scripts/batch_get_bad_ants.sh -a ${EX_ANTS_Y} /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc
badants_yid=`qsub -t 1-72 ~/src/hera_cal/scripts/batch_get_bad_ants.sh -a ${EX_ANTS_Y} /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
badants_yid=`echo ${badants_yid} | cut -f 1 -d . | cut -f 3 -d ' '`
echo badants_xid=qsub -t 1-72 ~/src/hera_cal/scripts/batch_get_bad_ants.sh -a ${EX_ANTS_X} /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc
badants_xid=`qsub -t 1-72 ~/src/hera_cal/scripts/batch_get_bad_ants.sh -a ${EX_ANTS_X} /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
badants_xid=`echo ${badants_yid} | cut -f 1 -d . | cut -f 3 -d ' '`


# Firstcal
echo firstcal_yid=qsub -hold_jid ${badants_yid} -t 1-72 ~/src/hera_cal/scripts/batch_firstcal.sh -p yy /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc
firstcal_yid=`qsub -hold_jid ${badants_yid} -t 1-72 ~/src/hera_cal/scripts/batch_firstcal.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
firstcal_yid=`echo ${firstcal_yid} | cut -f 1 -d . | cut -f 3 -d ' '`
#bad_ants in this script
echo firstcal_xid=qsub -hold_jid ${badants_xid} -t 1-72 ~/src/hera_cal/scripts/batch_firstcal.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc
firstcal_xid=`qsub -hold_jid ${badants_xid} -t 1-72 ~/src/hera_cal/scripts/batch_firstcal.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
firstcal_xid=`echo ${firstcal_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


# run omnical
echo omni_run_yid=qsub -hold_jid ${firstcal_yid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_run.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc
omni_run_yid=`qsub -hold_jid ${firstcal_yid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_run.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
omni_run_yid=`echo ${omni_run_yid} | cut -f 1 -d . | cut -f 3 -d ' '`
echo omni_run_xid=qsub -hold_jid ${firstcal_xid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_run.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc
omni_run_xid=`qsub -hold_jid ${firstcal_xid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_run.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
omni_run_xid=`echo ${omni_run_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


# apply omnical solutions to files
echo omni_apply_yid=qsub -hold_jid ${omni_run_yid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_apply.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc
omni_apply_yid=`qsub -hold_jid ${omni_run_yid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_apply.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
omni_apply_yid=`echo ${omni_apply_yid} | cut -f 1 -d . | cut -f 3 -d ' '`
echo omni_apply_xid=qsub -hold_jid ${omni_run_xid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_apply.sh -p xx  /data4/paper/HERA2015/${jd}/zen.{jd}.*.xx.HH.uvc
omni_apply_xid=`qsub -hold_jid ${omni_run_xid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_apply.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
omni_apply_xid=`echo ${omni_apply_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


# run xrfi
echo xrfi_yid=qsub -hold_jid ${omni_apply_yid} -t 1-72 ~/src/hera_cal/scripts/batch_xrfi.sh /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc.omni.calfits
xrfi_yid=`qsub -hold_jid ${omni_apply_yid} -t 1-72 ~/src/hera_cal/scripts/batch_xrfi.sh /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc.omni.calfits`
xrfi_yid=`echo ${xrfi_yid} | cut -f 1 -d . | cut -f 3 -d ' '`
echo xrfi_xid=qsub -hold_jid ${omni_apply_xid} -t 1-72 ~/src/hera_cal/scripts/batch_xrfi.sh /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc.omni.calfits
xrfi_xid=`qsub -hold_jid ${omni_apply_xid} -t 1-72 ~/src/hera_cal/scripts/batch_xrfi.sh /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc.omni.calfits`
xrfi_xid=`echo ${xrfi_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


# apply xrfi omnical solutions
echo omni_apply_yid=qsub -hold_jid ${xrfi_yid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_xrfi_apply.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc
omni_apply_xrfi_yid=`qsub -hold_jid ${xrfi_yid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_xrfi_apply.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
omni_apply_xrfi_yid=`echo ${omni_apply_xrfi_yid} | cut -f 1 -d . | cut -f 3 -d ' '`
echo omni_apply_xid=qsub -hold_jid ${xrfi_xid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_xrfi_apply.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc
omni_apply_xrfi_xid=`qsub -hold_jid ${xrfi_xid} -t 1-72 ~/src/hera_cal/scripts/batch_omni_xrfi_apply.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
omni_apply_xrfi_xid=`echo ${omni_apply_xrfi_xid} | cut -f 1 -d . | cut -f 3 -d ' '`
echo cd ..
cd ..

