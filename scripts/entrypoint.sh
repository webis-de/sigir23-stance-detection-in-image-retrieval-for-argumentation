#!/bin/bash

./scripts/elasticsearch.sh start
echo "Waiting for elasticsearch"
tail -f ../elastic.log | sed '/Node.*started/ q'
echo "Elasticsearch started"
sleep 1
python3 ./startup.py -i /data -w ./working -f "$@"
# python3 startup.py -i /data -w ./working -f -qrel -mtag "webis#1.0:elastic#1.0:formula#1.0:random"

