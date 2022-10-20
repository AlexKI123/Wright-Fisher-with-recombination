#!/bin/bash


try_host=(....)


datapoints=${#try_host[@]}
echo "Datapoints: $datapoints"


i=0
for a in ${try_host[@]}; do
    if [ "$(ssh $a 'echo "OK"')" == "OK" ];then
        echo "PC: $a"
        echo "i: $i"
        ssh $a "... python3 ....py "$i" >/dev/null 2>&1 </dev/null &"
        i=$(($i+1))
    fi
done

wait 
echo "All done"

