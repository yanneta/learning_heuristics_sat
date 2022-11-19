#!/usr/bin/env bash


epochs=30
run_dataset()
{
    path="../data/$1"
    echo $path
    aux=`echo $1 | cut -d "/" -f 2`
    python train_ls.py --dir_path=$path --epochs=$epochs > out$aux &
}


run_all()
{
for path in "domset/4-12-0.2" "kclique/3-20-0.05" "kcolor/4-15-0.5" "kcolor/5-20-0.5"
do	
    run_dataset $path
done 
}

run_rand3sat()
{
for path in "50-213" "75-320" "100-426" 
do
    run_dataset rand3sat/$path
done
}

run_rand3sat
run_all
