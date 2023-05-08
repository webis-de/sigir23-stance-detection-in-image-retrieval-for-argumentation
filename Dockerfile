# syntax=docker/dockerfile:1

FROM eclipse-temurin:17

RUN apt update && apt install -y --no-install-recommends \
    curl \
    python3-pip \
    python3-opencv \
    libenchant-2-2 \
    tesseract-ocr
RUN pip3 install --upgrade pip
RUN pip3 install virtualenv

RUN useradd -ms /bin/bash user
USER user

RUN mkdir -p /home/user/elasticsearch /home/user/app/data

WORKDIR /home/user/elasticsearch
RUN curl -L -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.7.1-linux-x86_64.tar.gz \
  && tar xzf elasticsearch-*.tar.gz \
  && rm elasticsearch-*.tar.gz \
  && sed -i 's/^#path\.data.*/path.data: \/home\/user\/elasticsearch\/index/' elasticsearch-*/config/elasticsearch.yml \
  && sed -i 's/^#path\.logs.*/path.logs: \/home\/user\/elasticsearch\/logs/' elasticsearch-*/config/elasticsearch.yml \
  && echo "xpack.security.enabled: false" | tee -a elasticsearch-*/config/elasticsearch.yml

WORKDIR /home/user/app
RUN wget https://files.webis.de/corpora/corpora-webis/corpus-touche-image-search-22/touche-task3-001-050-relevance.qrels -O data/touche-task3-001-050-relevance.qrels

COPY requirements.txt requirements.txt

RUN virtualenv env \
  && /bin/bash -c "source env/bin/activate" \
  && pip3 install -r requirements.txt --default-timeout=5000

RUN python3 -m spacy download en_core_web_sm
RUN python3 -c "exec(\"import nltk\nnltk.download('vader_lexicon')\")"
RUN python3 -c "exec(\"import nltk\nnltk.download('punkt')\")"

COPY . .

# TIRA stuff
USER root
RUN apt install -y --no-install-recommends sudo

# sudo -u user '/home/user/app/scripts/entrypoint.sh /input /output "<method>"'
# ENTRYPOINT ["./scripts/entrypoint.sh"]
