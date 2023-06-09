# On Stance Detection in Image Retrieval for Argumentation

This repository includes different attempts to retrieve argumentative images from a data set for or against an entered query. The task was proposed in [Touché Task 3: Image Retrival for Arguments](https://webis.de/events/touche-22/shared-task-3.html). The results were published in the reproducability track of the SIGIR conference 2023. 

This is the branch for TIRA usage.

## Setup
You need [Docker](https://www.docker.com/101-tutorial) installed.
```
mkdir out
docker build -t registry.webis.de/code-research/tira/tira-user-minsc/sigir23-stance-detection-in-image-retrieval-for-argumentation:2.0.5-tira .
```

## Run
This will first index the dataset (here assumed to be in `./data/touche23-image-search-main`) and then run the retrieval.
```
docker run -it --rm -v $PWD/data/touche23-image-search-main/:/input/ -v $PWD/out/:/output/ registry.webis.de/code-research/tira/tira-user-minsc/sigir23-stance-detection-in-image-retrieval-for-argumentation:2.0.5-tira '/bin/bash'
> chmod 777 /output/; sudo -u user /home/user/app/scripts/entrypoint.sh /input/ /output/ "webis#1.0:elastic#1.0:formula#0.0:random"
```

The last parameter is the method tag ``webis#{topicWeight}#{argumentWeight}:{ArgumentModel}#{stanceWeight}:{StanceModel}``:
 - Topic weight: a float in ``[0,1]`` wich represents the use of the topic score in the retrieval process
 - Argument weight: a float in ``[0,1]`` wich represents the use of the argument score in the retrieval process
 - ArgumentModel: ``formula or NN-{model_name}`` where ``model_name`` is the name of a trained neural net 
 - Stance weight: a float in ``[0,1]`` wich represents the use of the stance score in the retrieval process
 - StanceModel: ``random, bert, formula or NN-{model_name}`` where ``model_name`` is the name of a trained neural net 
 
## TIRA
Command: `mkdir -p $outputDir; chmod 777 $outputDir; sudo -u user /home/user/app/scripts/entrypoint.sh $inputDataset $outputDir "webis#1.0:elastic#1.0:formula#0.0:random"`

