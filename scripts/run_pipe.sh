#set up
jd=${1}
echo cd ${jd}
cd ${jd}
echo mkdir grid_output
mkdir grid_output

#bad_ants in this script
#echo firstcal_yid=`qsub -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal.sh -p yy /data4/paper/HERA2015/${jid}/zen.${jd}.*.yy.HH.uvc`
firstcal_yid=`qsub -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
firstcal_yid=`echo ${firstcal_yid} | cut -f 1 -d . | cut -f 3 -d ' '`

#bad_ants in this script
#echo firstcal_xid=`qsub -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal.sh -p xx  /data4/paper/HERA2015/${jid}/zen.${jd}.*.xx.HH.uvc`
firstcal_xid=`qsub -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
firstcal_xid=`echo ${firstcal_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


#echo median_yid=`qsub -hold_jid ${firstcal_yid} -t 1-2 ~/src_mycapo/zsa/scripts/batch_write_median_firstcal_files.sh /data4/paper/HERA2015/${jid}/zen.${jd}.*.yy*.npz`
#median_yid=`qsub -hold_jid ${firstcal_yid} -t 1 ~/src/mycapo/zsa/scripts/batch_write_median_firstcal_files.sh /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy*.npz`
#median_yid=`echo ${median_yid} | cut -f 1 -d . | cut -f 3 -d ' '`

#echo median_xid=`qsub -hold_jid ${firstcal_xid} -t 1-2 ~/src_mycapo/zsa/scripts/batch_write_median_firstcal_files.sh /data4/paper/HERA2015/${jid}/zen.${jd}.*.xx*.npz`
#median_xid=`qsub -hold_jid ${firstcal_xid} -t 1 ~/src/mycapo/zsa/scripts/batch_write_median_firstcal_files.sh /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx*.npz`
#median_xid=`echo ${median_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


#echo firstcal_apply_yid=qsub -hold_jid ${median_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply_yy.sh  /data4/paper/HERA2015/{jd}/zen.${jd}.*.yy.HH.uvc
#firstcal_apply_yid=`qsub -hold_jid ${median_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply_yy.sh  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
#firstcal_apply_yid=`echo ${firstcal_apply_yid} | cut -f 1 -d . | cut -f 3 -d ' '`
#
#echo firstcal_apply_xid=qsub -hold_jid ${median_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply.sh  /data4/paper/HERA2015/{jd}/zen.${jd}.*.xx.HH.uvc
#firstcal_apply_xid=`qsub -hold_jid ${median_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply.sh  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
#firstcal_apply_xid=`echo ${firstcal_apply_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


echo firstcal_apply_yid=qsub -hold_jid ${firstcal_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply.sh -p yy  /data4/paper/HERA2015/{jd}/zen.${jd}.*.yy.HH.uvc
firstcal_apply_yid=`qsub -hold_jid ${firstcal_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply_yy.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
firstcal_apply_yid=`echo ${firstcal_apply_yid} | cut -f 1 -d . | cut -f 3 -d ' '`

echo firstcal_apply_xid=qsub -hold_jid ${firstcal_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply.sh -p xx  /data4/paper/HERA2015/{jd}/zen.${jd}.*.xx.HH.uvc
firstcal_apply_xid=`qsub -hold_jid ${firstcal_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_firstcal_apply.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
firstcal_apply_xid=`echo ${firstcal_apply_xid} | cut -f 1 -d . | cut -f 3 -d ' '`



#bad_ants in this script
echo omni_run_yid=qsub -hold_jid ${firstcal_apply_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_run.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc
omni_run_yid=`qsub -hold_jid ${firstcal_apply_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_run.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
omni_run_yid=`echo ${omni_run_yid} | cut -f 1 -d . | cut -f 3 -d ' '`

#bad_ants in this script
echo omni_run_xid=qsub -hold_jid ${firstcal_apply_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_run.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc
omni_run_xid=`qsub -hold_jid ${firstcal_apply_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_run.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`
omni_run_xid=`echo ${omni_run_xid} | cut -f 1 -d . | cut -f 3 -d ' '`


echo omni_apply_yid=qsub -hold_jid ${omni_run_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_apply.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc
omni_apply_yid=`qsub -hold_jid ${omni_run_yid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_apply.sh -p yy  /data4/paper/HERA2015/${jd}/zen.${jd}.*.yy.HH.uvc`
echo omni_apply_xid=qsub -hold_jid ${omni_run_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_apply.sh -p xx  /data4/paper/HERA2015/${jd}/zen.{jd}.*.xx.HH.uvc
omni_apply_xid=`qsub -hold_jid ${omni_run_xid} -t 1-72 ~/src/mycapo/zsa/scripts/batch_omni_apply.sh -p xx  /data4/paper/HERA2015/${jd}/zen.${jd}.*.xx.HH.uvc`

echo cd ..
cd ..

