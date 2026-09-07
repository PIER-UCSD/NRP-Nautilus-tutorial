#!/usr/bin/env bash
# Build and push the lab image, then print the DIGEST to pin in your YAML.
#
#   REGISTRY=ghcr.io/pier-ucsd bash env/build_image.sh
#
# Registry choices (confirm which one the lab uses with the admin):
#   * gitlab.nrp-nautilus.io/<group>/<project>  — NRP's own registry, closest to the cluster
#   * ghcr.io/<org>                             — GitHub Container Registry
# A public image needs no pull secret; a private one needs an imagePullSecret
# in the namespace.
set -euo pipefail

REGISTRY="${REGISTRY:?set REGISTRY, e.g. REGISTRY=ghcr.io/pier-ucsd}"
NAME="${NAME:-shi-lab-gpu}"
TAG="${TAG:-$(date +%Y%m%d)}"
IMAGE="$REGISTRY/$NAME:$TAG"

# Nautilus nodes are linux/amd64 — build for that platform even from an Apple Silicon Mac.
docker buildx build --platform linux/amd64 -t "$IMAGE" --push env/

echo
echo "pushed: $IMAGE"
echo "digest (pin THIS in your YAML, tags can be overwritten, digests cannot):"
docker buildx imagetools inspect "$IMAGE" --format '{{.Manifest.Digest}}' \
  | sed "s|^|  $REGISTRY/$NAME@|"
