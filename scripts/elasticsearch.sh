#!/bin/bash

case $1 in
  start)
    /home/user/elasticsearch/elasticsearch-*/bin/elasticsearch 1> /home/user/elastic.log 2>&1 &
    echo $! > /home/user/elastic.pid.txt
    ;;
  stop)
    pid=$(cat /home/user/elastic.pid.txt)
    kill $pid
    rm /home/user/elastic.pid.txt
    ;;
  *)
    echo Unknown command: $1
esac
