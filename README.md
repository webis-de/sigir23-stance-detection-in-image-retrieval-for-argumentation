# On Stance Detection in Image Retrieval for Argumentation

This repository includes different attempts to retrive argumentative images from a data set for or against an entered query. The task was proposed in [Touché Task 3: Image Retrival for Arguments](https://webis.de/events/touche-22/shared-task-3.html). The results were published in the reproducability track of the SIGIR conference 2023. 

## Setup
You can use a virtual environment or you use [Docker](https://www.docker.com/101-tutorial) to setup this project. If you are not familiar with Docker, 
please visit the linked tutorial. 

For indexing or retrieval a [ElasticSearch](https://www.elastic.co/elasticsearch/) instance is needed at ``http://localhost:9200`` without any security settings enabled.

Clone this repository and create a docker image with the ``Dockerfile``. This image contains the entrypoint to the ``startup.py``. 
The program needs three directories to work correctly:
 - an input directory where the [data](https://files.webis.de/corpora/corpora-webis/corpus-touche-image-search-22/) is located (default: ``./data``)
 - a working directory where the index and other stuff is saved for multiple use (default: ``./working``)
 - an output directory where the results are saved (default: ``./out``)
 
It's possible to set the directories in the config.json, if so the ``config.json - PATH`` has to be the
parameter after ``-cfg``. 

The data can be structured in two different ways. If you download the data from the Touché website it is 
splited in six different directories with different parts of each image id. You can just unzip those files and move them to the input directory.
But you can also merge these six directories into one where only the image subdirectory is left. Then you have to pass the ``-f`` parameter to the program.   

### Image-detection
If you use Docker, all the requirements (including tesseract) are included in the docker-image and the following part is 
not important to get the system started. But we would like to mention the used technologies and the possibility to use
image-detection as stand-alone, without the retrieval-system.

The image-detection is only needed during indexing. 
To run the image-detection without docker you have to download the tesseract5-installer 
[directly](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-rc1.20211030.exe)
or from the [Uni-Mannheim-page](https://github.com/UB-Mannheim/tesseract/wiki) and install it.
Please use tesseract 5 or higher. In config the setting ``on_windows`` must be ``True``.

You can install the application directly into ``\properties`` or install it anywhere and copy the ``tesseract`` folder into ``\properties``.
Check, if the ``\properties\tesseract\tesseract.exe`` exists, but if you have done everything correctly this file should be there.


## Functions
The programm has different functions:
 - Indexing ``-idx``
 - Retrieval run ``-qrel`` + ``-mtag {method tag}``
 - Web application with search/evaluation interface ``-web``

In the script [scripts/tira-run.sh](scripts/tira-run.sh) is an example docker run, where only the input/output directories and the function parameter are needed.
To start for example the indexing process run ``sh ./tira-run.sh -i $inputDir -o $outDir -idx``.
For a complete view of the possible parameter use ``-help``. We provide our final models in ``working/models/``.

### Method tag
The method tag ``webis#{topicWeight}#{argumentWeight}:{ArgumentModel}#{stanceWeight}:{StanceModel}`` for retrieval run has five parameters:
 - Topic weight: a float in ``[0,1]`` wich represents the use of the topic score in the retrieval process
 - Argument weight: a float in ``[0,1]`` wich represents the use of the argument score in the retrieval process
 - ArgumentModel: ``formula or NN-{model_name}`` where ``model_name`` is the name of a trained neural net 
 - Stance weight: a float in ``[0,1]`` wich represents the use of the stance score in the retrieval process
 - StanceModel: ``random, bert, formula or NN-{model_name}`` where ``model_name`` is the name of a trained neural net 
 



## Evaluation
Our program offers an evaluation website (located under ``0.0.0.0/evaluation``). 
If you don't see images, please check if you set a username in the top right corner.

We evaluated ~9500 images from the [Touché22 Task3 data](https://files.webis.de/corpora/corpora-webis/corpus-touche-image-search-22/). 
These evaluation can be found in [working/image_eval.txt](working/image_eval.txt). 
The users where anonymized. A description of the labels can be found in out paper.

In the [analysis_labeled_data.md](analysis_labeled_data_table.md) file is the result of our analysis of the Touché dataset.
For details check the paper.
