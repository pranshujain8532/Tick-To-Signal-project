# Serving image for tick-to-signal.
#
# TWO STAGES, AND THE POINT IS WHAT THE SECOND ONE LACKS.
#     `build` has pip, wheels, and whatever compilers a source distribution
#     might need. `runtime` has none of that — it receives an already-installed
#     tree and the application code. A runtime layer carrying a compiler is a
#     larger image, a larger attack surface, and an invitation to "just pip
#     install something" on a running container, which is how a deployment stops
#     matching the repository that supposedly produced it.
#
# NO TORCH IN EITHER STAGE.
#     The model ships as a 126 KiB quantised ONNX graph and is executed by
#     onnxruntime. Installing the framework that trained it would add roughly
#     2 GB to run a file smaller than this Dockerfile's build context. See
#     serving/requirements-serving.txt.
#
# The image is used by BOTH compose services. The `api` service serves the
# websocket and the REST endpoints; the optional `capture` service runs
# data_engine.capture into a shared volume. One image, two commands — a second
# image would double the build for a few hundred kilobytes of difference.

# ---------------------------------------------------------------- build stage
FROM python:3.10-slim AS build

# --prefix installs into a relocatable tree that the runtime stage copies whole.
# Rejected alternative: a virtualenv, which works but adds an activation step
# and a second copy of the interpreter for no benefit here.
COPY serving/requirements-serving.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

# -------------------------------------------------------------- runtime stage
FROM python:3.10-slim AS runtime

COPY --from=build /install /usr/local

WORKDIR /app

# Only what the serving path imports, and the evidence it serves. Copying the
# whole repository would drag in data/, checkpoints/ and the C++ build trees.
COPY serving/ /app/serving/
COPY ml/__init__.py ml/features.py /app/ml/
COPY data_engine/__init__.py data_engine/binfmt.py data_engine/book.py \
     data_engine/capture.py data_engine/replay.py /app/data_engine/
COPY benchmarks/*.json /app/benchmarks/
# The committed replay tapes and the quantised graph: this is what makes the
# demo work with no network, no exchange credentials and no prior capture run.
COPY notebooks/sample_data/btcusdt_replay_s0.tape \
     notebooks/sample_data/btcusdt_replay_s1.tape \
     notebooks/sample_data/btcusdt_replay_s2.tape \
     notebooks/sample_data/student_int8.onnx \
     /app/notebooks/sample_data/

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1

# Runs as a non-root user. Nothing here needs to write to the image, and the
# shared capture volume is mounted with matching ownership by compose.
RUN useradd --create-home --uid 10001 tts && chown -R tts:tts /app
USER tts

EXPOSE 8000

# Demo mode is the default command, so `docker run` with no arguments is a
# working demo rather than a usage message.
CMD ["python", "-m", "uvicorn", "serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
