#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
MODEL_DIR="$PROJECT_DIR/models"

mkdir -p "$MODEL_DIR"

download_file() {
    url=$1
    target=$2
    checksum=$3
    if [ ! -f "$target" ]; then
        temporary="${target}.download"
        wget --https-only --show-progress -O "$temporary" "$url"
        mv "$temporary" "$target"
    fi
    printf '%s  %s\n' "$checksum" "$target" | sha256sum -c -
}

download_bzip2() {
    url=$1
    target=$2
    checksum=$3
    if [ ! -f "$target" ]; then
        archive="${target}.bz2"
        wget --https-only --show-progress -O "$archive" "$url"
        bzip2 -df "$archive"
    fi
    printf '%s  %s\n' "$checksum" "$target" | sha256sum -c -
}

download_bzip2 \
    "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2" \
    "$MODEL_DIR/shape_predictor_68_face_landmarks.dat" \
    "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f"

download_bzip2 \
    "https://dlib.net/files/mmod_human_face_detector.dat.bz2" \
    "$MODEL_DIR/mmod_human_face_detector.dat" \
    "4cb19393e2fbaf2b1609a9319ad5386618c886a6234ec1b971f3e87c85d87fe6"

download_file \
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/face_detector/deploy.prototxt" \
    "$MODEL_DIR/deploy.prototxt" \
    "dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9"

download_file \
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel" \
    "$MODEL_DIR/res10_300x300_ssd_iter_140000_fp16.caffemodel" \
    "510ffd2471bd81e3fcc88a5beb4eae4fb445ccf8333ebc54e7302b83f4158a76"

printf '\nModelos V3 instalados y verificados en %s\n' "$MODEL_DIR"
printf 'Backend preferido: dlib CNN CUDA; respaldos: OpenCV CUDA FP16 y dlib HOG.\n'
