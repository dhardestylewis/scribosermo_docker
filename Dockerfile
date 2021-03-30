FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
WORKDIR /

RUN apt-get update && apt-get install -y libsndfile1

# Use some tools from DeepSpeech project
RUN git clone --depth 1 https://github.com/mozilla/DeepSpeech.git
# CTC decoder (the next line is required for building with shallow git clone)
RUN sed -i 's/git describe --long --tags/git describe --long --tags --always/g' /DeepSpeech/native_client/bazel_workspace_status_cmd.sh
RUN apt-get update && apt-get install -y libmagic-dev
RUN cd /DeepSpeech/native_client/ctcdecode && make NUM_PROCESSES=$(nproc) bindings
RUN pip3 install --upgrade /DeepSpeech/native_client/ctcdecode/dist/*.whl
# KenLM
RUN apt-get update && apt-get install -y libboost-all-dev
RUN cd /DeepSpeech/native_client/ && \
  rm -rf kenlm && \
  git clone --depth 1  https://github.com/kpu/kenlm && \
  mkdir -p kenlm/build && \
  cd kenlm/build && \
  cmake .. && \
  make -j $(nproc)
## Graph converter
#RUN python3 /DeepSpeech/util/taskcluster.py --source tensorflow --branch r1.15 \
#  --artifact convert_graphdef_memmapped_format  --target /DeepSpeech/

# Get prebuilt scorer generator script
RUN cd /DeepSpeech/data/lm/ \
  && curl -LO https://github.com/mozilla/DeepSpeech/releases/latest/download/native_client.amd64.cpu.linux.tar.xz \
  && tar xvf native_client.*.tar.xz

# Solve broken pip "ImportError: No module named pip._internal.cli.main"
RUN python3 -m pip install --upgrade pip

# Dependencies for noise normalization
RUN apt-get update && apt-get install -y ffmpeg
RUN pip install --no-cache-dir --upgrade pydub

# Pre-install some libraries for faster installation time of training package
RUN pip3 install --no-cache-dir pandas
RUN pip3 install --no-cache-dir librosa
RUN pip3 install --no-cache-dir "tensorflow<2.4,>=2.3"
RUN pip3 install --no-cache-dir "tensorflow-addons<0.12"
RUN pip3 install --no-cache-dir "tensorflow-io<0.17"

# Install audiomate, with some fixes
RUN apt-get update && apt-get install -y sox libsox-fmt-mp3
RUN pip3 install --no-cache-dir audiomate
RUN sed -i 's/from down import Downloader/from pget.down import Downloader/g' /usr/local/lib/python3.8/dist-packages/pget/__init__.py
RUN sed -i 's/print "Resume is not applicable at this stage."/print("Resume is not applicable at this stage.")/g' /usr/local/lib/python3.8/dist-packages/pget/down.py

# Training profiler
RUN pip3 install --upgrade --no-cache-dir tensorboard-plugin-profile

# TfLite runtime
RUN pip3 install --no-cache-dir --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime

# Install corcua
RUN git clone --depth 1 https://gitlab.com/Jaco-Assistant/corcua.git
RUN pip3 install --no-cache-dir -e corcua/

# Testing requirements
COPY requirements_test.txt /Scribosermo/requirements_test.txt
RUN pip3 install --no-cache-dir -r /Scribosermo/requirements_test.txt

# Training package
COPY training/ /Scribosermo/training/
RUN pip3 install --no-cache-dir -e /Scribosermo/training/

CMD ["/bin/bash"]
