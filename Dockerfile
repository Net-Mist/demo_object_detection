FROM    node:8.16.0-jessie AS builder
WORKDIR /opt/front
RUN     apt-get update && \
            apt-get install apt-transport-https && \
            curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
            echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
            apt-get update && \
            apt-get install yarn
COPY    front /opt/front/
# Build in /opt/static
RUN     yarn && yarn build 

FROM    tensorflow/tensorflow:1.14.0-gpu-py3
# libsm6 libxrandr2 libxext6 are for cv2
RUN     apt update \
        && apt install -y --no-install-recommends \
        libsm6 libxrandr2 libxext6 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
RUN     pip install flask flask-cors opencv-python coloredlogs
WORKDIR /opt
COPY    --from=builder /opt/static static
COPY    app.py app.py
COPY    ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03 model
COPY    mscoco_label_map.pbtxt mscoco_label_map.pbtxt
CMD     python3 app.py
