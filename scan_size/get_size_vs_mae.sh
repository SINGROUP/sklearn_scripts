# write a table of threshold vs MAE into a file

outfile="size_vs_mae.csv"

echo "#trainratio  MAE" > temp1
grep "^MAE" * >> temp1

sed -i "s/^.*size//g" temp1
sed -i "s/\.out:MAE/  /g" temp1

paste temp1 > $outfile

rm temp1










