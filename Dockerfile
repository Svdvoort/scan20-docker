FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN pip install \
    batchgenerators==0.24 \
    certifi==2022.6.15.1 \ 
    charset-normalizer==2.1.1 \
    future==0.18.2 \
    idna==3.3 \
    imageio==2.21.3 \
    joblib==1.1.0 \
    linecache2==1.0.0 \
    networkx==2.8.6 \
    nibabel==4.0.2 \
    numpy==1.23.3 \
    opencv-python==4.6.0.66 \
    packaging==21.3 \
    pandas==1.4.4 \
    Pillow==9.2.0 \
    pyparsing==3.0.9 \
    python-dateutil==2.8.2 \
    python-dotenv==0.21.0 \
    pytz==2022.2.1 \
    PyWavelets==1.3.0 \
    requests==2.28.1 \
    scikit-image==0.19.3 \
    scipy==1.9.1 \
    six==1.16.0 \
    threadpoolctl==3.1.0 \
    tifffile==2022.8.12 \
    traceback2==1.4.0 \ 
    typing-extensions==4.3.0 \
    unittest2==1.1.0 \
    urllib3==1.26.12


COPY app/ /app/

WORKDIR /app/
