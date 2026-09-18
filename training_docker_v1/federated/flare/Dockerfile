# Runtime image for the NVFlare server and every site.
# Contains libraries only: NVFlare ships the job code (client.py, model, ...)
# to each participant, and each site mounts its own data at run time.
#   docker build -t cogniwarriors-flare:latest -f federated/flare/Dockerfile .
FROM python:3.12-slim
RUN pip install --no-cache-dir "nvflare==2.9.0" pandas numpy scikit-learn \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
WORKDIR /workspace
