# write a table of threshold vs MAE into a file

outfile="thr_vs_mae.csv"

echo "#Threshold  MAE" > $outfile
grep "^MAE" * >> $outfile

sed -i "s/^.*thr//g" $outfile
sed -i "s/\.out:MAE/  /g" $outfile














