#!/usr/bin/env bash
set -euo pipefail

# Download and extract SIFT1M evaluation dataset
DATASET_DIR="datasets/sift1m"
mkdir -p "${DATASET_DIR}"

TAR_FILE="${DATASET_DIR}/sift.tar.gz"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

echo "==> Downloading SIFT1M Dataset from INRIA Texmex corpus..."
if [ ! -f "${DATASET_DIR}/sift_base.fvecs" ]; then
    if command -v wget >/dev/null 2>&1; then
        wget -c "${URL}" -O "${TAR_FILE}" || curl -L "${URL}" -o "${TAR_FILE}"
    else
        curl -L "${URL}" -o "${TAR_FILE}"
    fi

    echo "==> Extracting dataset archive..."
    tar -xzf "${TAR_FILE}" -C "${DATASET_DIR}" --strip-components=1
    rm -f "${TAR_FILE}"
    echo "==> SIFT1M extracted successfully to ${DATASET_DIR}"
else
    echo "==> SIFT1M dataset already downloaded."
fi
