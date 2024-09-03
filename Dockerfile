FROM registry.gitee-ai.local/base/iluvatar-corex:3.2.0-bi100
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y python3.10 \
    python3-pip \
    git \
    git-lfs \
    wget \
    ffmpeg
RUN pip install imagehash moviepy gradio
RUN useradd -ms /bin/bash jupyter
USER jupyter
WORKDIR home/jupyter
EXPOSE 7860
USER root
RUN chown -R jupyter:jupyter /home

COPY . .
RUN wget "https://gitee.com/qq764073542/extract_video_ppt_to_markdown/raw/master/videos/example_video1.mp4?lfs=1" -O videos/example_video1.mp4
ENTRYPOINT ["python", "./app.py"]
