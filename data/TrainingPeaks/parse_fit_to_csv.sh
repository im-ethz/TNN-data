#!/bin/bash
path='data/TrainingPeaks/'
name="kusztor"
i="13"

path_raw="${path}export/${name}/"
path_fit="${path}fit/${i}/"
path_csv="${path}csv/${i}/"

mkdir ${path_fit}
mkdir ${path_csv}
for filepath in "${path_raw}"*.gz
do
    filename=$(basename "${filepath}" .fit.gz)
    if ! [ -f "${path_csv}record/${filename}_record.csv" ] ; then
        echo "${filename}"
        gzip -dk "${filepath}"
        mv "${path_raw}/${filename}.fit" "${path_fit}/${filename}.fit"
        python3 data/TrainingPeaks/parse_fit_to_csv.py "${filename}.fit" -i ${path_fit} -o ${path_csv}
    fi
done