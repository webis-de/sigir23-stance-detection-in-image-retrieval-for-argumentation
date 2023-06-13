#!/bin/bash

input=$(readlink -f "$1")
output=$(readlink -f "$2")
method="$3"

pushd $(dirname $0)/..

./scripts/elasticsearch.sh start
echo "Waiting for elasticsearch"
tail -f ../elastic.log | sed '/Node.*started/ q'
echo "Elasticsearch started"
sleep 1
if [ -e ./working/feature_index_complete.pkl ];then
  echo "Elastic indexing"
  python3 ./startup.py -i $input -w ./working -f -esidx -njobs 1
else
  echo "Normal indexing"
  python3 ./startup.py -i $input -w ./working -f -idx -njobs 1
fi
python3 startup.py -i $input -w ./working -o $output -f -qrel -mtag "$method" -njobs 1
# "webis#1.0:elastic#1.0:formula#1.0:random"

