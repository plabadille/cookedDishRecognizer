#!/bin/bash

#this script handle renaming and moving data contain in two folder imbrication far. The data are moved where this script is.

total=0
for folder in *; do
    if [ $folder != *".sh" ]
        then
        i=0
        for img in ${folder}/*; do
            ((i++))
            mv ${img} ${folder}_${i}".jpg"    
        done
        ((total+=i))
    fi
done
echo "You have moved and rename $total files"