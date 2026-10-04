#!/usr/bin/env bash
# Download the public datasets into data/raw. Safe to re-run: finished parts are skipped
# and interrupted downloads resume. CodeNet needs ~8.5 GB of free disk space while running.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="$ROOT/data/raw"

PROGPEDIA_URL="https://zenodo.org/records/7449056/files/progpedia.zip?download=1"
CODENET_HOST="codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud"
CODENET_BASE="https://$CODENET_HOST/dax-project-codenet/1.0.0"
CODENET_LANGS=("C#" "PHP" "JavaScript")
DROID_URL="https://huggingface.co/datasets/project-droid/DroidCollection/resolve/main/data/test-00000-of-00001.parquet"

CURL_RESOLVE=()

# Some home routers fail to resolve the IBM storage host; fall back to public DNS.
resolve_codenet_host() {
    if curl -sI --max-time 20 "$CODENET_BASE/Project_CodeNet_metadata.tar.gz" >/dev/null 2>&1; then
        return
    fi
    local ip=""
    if command -v dig >/dev/null; then
        ip="$(dig +short @8.8.8.8 "$CODENET_HOST" | grep -E '^[0-9.]+$' | tail -1)"
    elif command -v nslookup >/dev/null; then
        ip="$(nslookup "$CODENET_HOST" 8.8.8.8 | awk '/^Address: /{print $2}' | tail -1)"
    fi
    if [[ -z "$ip" ]]; then
        echo "Cannot resolve $CODENET_HOST, check the network" >&2
        exit 1
    fi
    echo "Resolved $CODENET_HOST via public DNS: $ip"
    CURL_RESOLVE=(--resolve "$CODENET_HOST:443:$ip")
}

download() {
    local url="$1" out="$2"
    for attempt in $(seq 1 10); do
        if curl -fL -C - --retry 5 --retry-delay 5 ${CURL_RESOLVE[@]+"${CURL_RESOLVE[@]}"} \
            -o "$out" "$url"; then
            return
        fi
        echo "Download interrupted (attempt $attempt), resuming..." >&2
        sleep 5
    done
    echo "Failed to download $url" >&2
    exit 1
}

tar_patterns() {
    if tar --version 2>/dev/null | grep -q "GNU tar"; then
        tar --wildcards "$@"
    else
        tar "$@"
    fi
}

progpedia() {
    local dest="$RAW/progpedia"
    if [[ -f "$dest/data.md" ]]; then
        echo "PROGpedia: already present"
        return
    fi
    mkdir -p "$dest"
    echo "PROGpedia: downloading (~70 MB)"
    download "$PROGPEDIA_URL" "$dest/progpedia.zip"
    unzip -q -o "$dest/progpedia.zip" -d "$dest"
    rm "$dest/progpedia.zip"
}

codenet() {
    local dest="$RAW/codenet"
    mkdir -p "$dest"
    resolve_codenet_host

    if [[ ! -f "$dest/Project_CodeNet/metadata/problem_list.csv" ]]; then
        echo "CodeNet: downloading metadata (~300 MB)"
        download "$CODENET_BASE/Project_CodeNet_metadata.tar.gz" "$dest/metadata.tar.gz"
        tar -xzf "$dest/metadata.tar.gz" -C "$dest"
        rm "$dest/metadata.tar.gz"
    fi

    if [[ ! -f "$dest/.code_complete" ]]; then
        echo "CodeNet: downloading the full archive (~8.3 GB), resumable"
        download "$CODENET_BASE/Project_CodeNet.tar.gz" "$dest/Project_CodeNet.tar.gz"
        local patterns=("Project_CodeNet/problem_descriptions/*")
        for lang in "${CODENET_LANGS[@]}"; do
            patterns+=("Project_CodeNet/data/*/$lang/*")
        done
        echo "CodeNet: extracting ${CODENET_LANGS[*]} and problem descriptions"
        tar_patterns -xzf "$dest/Project_CodeNet.tar.gz" -C "$dest" "${patterns[@]}"
        rm "$dest/Project_CodeNet.tar.gz"
        touch "$dest/.code_complete"
    else
        echo "CodeNet: code already present"
    fi
}

droid() {
    local dest="$RAW/droid"
    if [[ -f "$dest/test.parquet" ]]; then
        echo "Droid: already present"
        return
    fi
    mkdir -p "$dest"
    echo "Droid: downloading the test split (~65 MB)"
    download "$DROID_URL" "$dest/test.parquet"
}

case "${1:-all}" in
    progpedia) progpedia ;;
    codenet) codenet ;;
    droid) droid ;;
    all) progpedia; codenet; droid ;;
    *) echo "usage: $0 [all|progpedia|codenet|droid]" >&2; exit 2 ;;
esac
echo "Done. Data is in $RAW"
