# write a table of trainratio vs MAE into a file

outfile="trainratio_vs_mae.csv"

echo "#trainratio  MAE" > $outfile
grep "^MAE" * | grep -v "param_size" >> $outfile

sed -i "s/^.*psize_size//g" $outfile
sed -i "s/\.out:MAE/  /g" $outfile














