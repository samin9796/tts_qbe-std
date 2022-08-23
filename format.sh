
cwd=$(pwd)

TARGET="$cwd/data/raw/datasets/gos-kdl/synthesized_queries"
if [ -d "$TARGET" ]; then rm -Rf $TARGET; fi
mkdir $TARGET

a=0;
for i in `ls *.wav`
do

let a++;
echo "Processing file $i"
sox $i -r 16000 -b 16 -c 1 "$TARGET/TTS_$i"
rm -f $i 

done