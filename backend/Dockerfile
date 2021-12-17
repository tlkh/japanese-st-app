FROM python:3.8.12-slim-buster
MAINTAINER Timothy Liu "timothy_liu@mymail.sutd.edu.sg"
USER root
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN python3 -m pip install --no-cache-dir torch==1.10.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    python3 -m pip --no-cache-dir install \
        transformers tokenizers sentencepiece flask flask_cors 

RUN python3 -c 'import transformers;transformers.MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")'

COPY . /app

RUN python3 cache_model.py

CMD python3 server.py

EXPOSE 5000

