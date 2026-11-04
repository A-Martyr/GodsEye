# GodsEye's read-only public demo as a container image.
#
# The code is cloned from the public GitHub repository at GODSEYE_REF, so the
# image always runs what the repository says. Two ways it is hosted:
#   - Azure App Service (deploy/azure.md): GitHub Actions builds this file on
#     every push to main and publishes ghcr.io/a-martyr/godseye:latest;
#     restart the web app to pick up a new build.
#   - A Hugging Face Docker Space (deploy/huggingface/): upload this file with
#     that folder's README.md; Settings > Factory rebuild after a push.
#
# Runs the default CRNN reader on the CPU (about 600 MB resident). The
# real-photo reader (paddle, ~4 GB with its weights) is left out on purpose:
# uploads are refused on the public demo anyway.
FROM python:3.12-slim

# libGL and glib for the OpenCV that ultralytics pulls in; git for the clone
RUN apt-get update \
 && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

ARG GODSEYE_REF=main
RUN git clone https://github.com/A-Martyr/GodsEye.git . && git checkout "$GODSEYE_REF"

# CPU torch first, from the CPU index: left to itself, ultralytics would pull
# the multi-GB CUDA build
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user -r requirements.txt

# Read-only on the public hostname (api/main.py); no judge console, no GPU
ENV GODSEYE_PUBLIC_DEMO=1 GODSEYE_JUDGE=0 GODSEYE_DEVICE=cpu GODSEYE_OCR_BACKEND=crnn

EXPOSE 7860
# The container's disk is reset on every restart, so each start seeds a fresh six
# hours of city traffic (about 30 s) and the live feed carries on from there.
# --no-access-log: the demo keeps no record of who visited (web/privacy.html says so)
CMD ["sh", "-c", "python seed.py && exec uvicorn api.main:app --host 0.0.0.0 --port 7860 --no-access-log"]
