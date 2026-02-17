#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo bash scripts/install_deps_ubuntu.sh"
  exit 1
fi

apt-get update
apt-get install -y \
  ca-certificates \
  curl \
  git \
  ffmpeg \
  postgresql-client \
  python3 \
  python3-venv \
  python3-pip \
  build-essential \
  pkg-config \
  libsndfile1

echo "System dependencies installed."
echo "Optional: install Docker Engine if you want postgres in docker-compose."
