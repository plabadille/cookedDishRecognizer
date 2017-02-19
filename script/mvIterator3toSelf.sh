#!/bin/bash

#this script handle renaming and moving data contain in 3 folder imbrication far. The data are moved where this script is.

total=0
for folder in *; do
    if [ $folder != *".sh" ] && [ $folder != ".gitignore" ]
        then
        i=0
        for class in ${folder}/*; do
            for img in ${class}/*; do
                ((i++))
                mv ${img} ${folder}_${i}".jpg"
            done  
        done
        ((total+=i))
    fi
done
echo "You have moved and rename $total files"