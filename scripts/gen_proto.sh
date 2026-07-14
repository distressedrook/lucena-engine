#!/usr/bin/env bash
# Regenerate the Python gRPC stubs from the proto. Run from engine/.
# The generated files live in python/lucena_engine/_pb/ and are committed.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m grpc_tools.protoc \
  -I proto/lucena/engine/v1 \
  --python_out=python/lucena_engine/_pb \
  --grpc_python_out=python/lucena_engine/_pb \
  engine.proto

# The flat codegen emits `import engine_pb2` — make it package-relative.
sed -i '' 's/^import engine_pb2 as/from . import engine_pb2 as/' \
  python/lucena_engine/_pb/engine_pb2_grpc.py

echo "regenerated python/lucena_engine/_pb/{engine_pb2,engine_pb2_grpc}.py"
