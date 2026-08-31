#!/usr/bin/env bash
set -euo pipefail

REPO="thx-rs/thx"
BIN_NAME="thx"
RUSTC_LABEL="${THX_RUSTC_LABEL:-rust-stable}"

die() {
  printf "error: %s\n" "$*" >&2
  exit 1
}

detect_platform() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  case "${arch}" in
    x86_64 | amd64) arch="x86_64" ;;
    aarch64 | arm64) arch="aarch64" ;;
    armv7l | armv7 | armhf) arch="armv7" ;;
    *) die "unsupported architecture: ${arch}" ;;
  esac

  case "${os}" in
    Darwin)
      target="${arch}-apple-darwin"
      archive_ext="tar.gz"
      ;;
    MINGW* | MSYS* | CYGWIN* | Windows_NT)
      target="${arch}-pc-windows-msvc"
      archive_ext="zip"
      ;;
    Linux)
      if [[ -n "${ANDROID_ROOT:-}" || -d "/data/data/com.termux" ]]; then
        case "${arch}" in
          aarch64) target="aarch64-linux-android" ;;
          armv7) target="armv7-linux-androideabi" ;;
          x86_64) target="x86_64-linux-android" ;;
        esac
      else
        case "${arch}" in
          x86_64) target="x86_64-unknown-linux-gnu" ;;
          aarch64) target="aarch64-unknown-linux-gnu" ;;
          armv7) target="armv7-unknown-linux-musleabihf" ;;
        esac
      fi
      archive_ext="tar.gz"
      ;;
    *)
      die "unsupported operating system: ${os}"
      ;;
  esac
}

download() {
  local url="$1" dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${dest}"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "${url}" -O "${dest}"
  else
    die "curl or wget is required"
  fi
}

detect_platform

if [[ -n "${THX_VERSION:-}" ]]; then
  version_tag="v${THX_VERSION#v}"
  base_url="https://github.com/${REPO}/releases/download/${version_tag}"
else
  base_url="https://github.com/${REPO}/releases/latest/download"
fi

archive_name="${BIN_NAME}-${target}-${RUSTC_LABEL}.${archive_ext}"
checksum_name="${archive_name}.sha256"
archive_url="${base_url}/${archive_name}"
checksum_url="${base_url}/${checksum_name}"

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

archive_path="${tmpdir}/${archive_name}"
printf "Downloading %s\n" "${archive_url}"
download "${archive_url}" "${archive_path}"

checksum_path="${tmpdir}/${checksum_name}"
if download "${checksum_url}" "${checksum_path}" 2>/dev/null; then
  expected="$(awk '{print $1}' "${checksum_path}")"
  if command -v shasum >/dev/null 2>&1; then
    actual="$(shasum -a 256 "${archive_path}" | awk '{print $1}')"
  elif command -v sha256sum >/dev/null 2>&1; then
    actual="$(sha256sum "${archive_path}" | awk '{print $1}')"
  else
    die "shasum or sha256sum is required"
  fi
  if [[ "${expected}" != "${actual}" ]]; then
    die "checksum mismatch for ${archive_name}"
  fi
  printf "Verified SHA-256 checksum\n"
fi

extract_dir="${tmpdir}/extract"
mkdir -p "${extract_dir}"
case "${archive_ext}" in
  zip)
    if command -v unzip >/dev/null 2>&1; then
      unzip -q "${archive_path}" -d "${extract_dir}"
    else
      tar -xf "${archive_path}" -C "${extract_dir}"
    fi
    ;;
  *)
    tar -xzf "${archive_path}" -C "${extract_dir}"
    ;;
esac

bin_path="${extract_dir}/${BIN_NAME}"
if [[ "${archive_ext}" == "zip" && ! -f "${bin_path}" ]]; then
  bin_path="${bin_path}.exe"
fi
if [[ ! -f "${bin_path}" ]]; then
  die "binary not found in ${archive_name}"
fi

if [[ -n "${THX_INSTALL_DIR:-}" ]]; then
  install_dir="${THX_INSTALL_DIR}"
elif [[ -d "/data/data/com.termux" ]]; then
  install_dir="${PREFIX:-/data/data/com.termux/files/usr}/bin"
else
  install_dir="${HOME}/.local/bin"
fi

install_name="$(basename "${bin_path}")"
mkdir -p "${install_dir}"
cp -f "${bin_path}" "${install_dir}/${install_name}"
chmod +x "${install_dir}/${install_name}"

if ! command -v "${BIN_NAME}" >/dev/null 2>&1 ||
  [[ "$(command -v "${BIN_NAME}")" != "${install_dir}/${install_name}" ]]; then
  printf "Installed %s to %s\n" "${BIN_NAME}" "${install_dir}/${install_name}"
  printf "Add it to your PATH:\n  export PATH=\"%s:$PATH\"\n" "${install_dir}"
else
  printf "Installed %s to %s\n" "${BIN_NAME}" "${install_dir}/${install_name}"
fi