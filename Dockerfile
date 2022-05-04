FROM gcr.io/tpu-pytorch/xla:r1.10_3.8_tpuvm

RUN pip install --upgrade pip
RUN sudo python3 -m pip uninstall -y tf-nightly tb-nightly
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install timm
RUN python3 -m pip install --upgrade tensorflow==2.6.2 wandb tensorflow-datasets jsonpickle mesh-tensorflow tpunicorn google-auth google-api-python-client cython pysimdjson absl-py ftfy jsonlines lm_dataformat tokenizers transformers ortools pytest sacred attrs opencv-python Pillow google-cloud-storage utils oauth2client scipy fastapi uvicorn click sklearn
RUN ~/.local/bin/wandb login 13f4789f1fcf20a514dd3d77b099ed4746992ae3

RUN mkdir train/
WORKDIR train/

# Assuming that you are in the directory with your code
COPY . .