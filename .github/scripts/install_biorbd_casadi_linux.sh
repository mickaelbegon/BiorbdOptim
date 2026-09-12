#!/usr/bin/env bash
set -euo pipefail

readonly BIORBD_REPOSITORY="https://github.com/pyomeca/biorbd.git"
readonly BIORBD_TAG="Release_1.12.2"
readonly RBDL_REPOSITORY="https://github.com/pariterre/rbdl.git"
readonly RBDL_COMMIT="93475e2ea9bc87f37709a2312533ce3187f054b9"
readonly SCRIPT_DIRECTORY="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer is intended for Linux." >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX must point to the active benchmark environment." >&2
  exit 1
fi

casadi_package_dir="$(
  python -c 'import casadi, pathlib; print(pathlib.Path(casadi.__file__).resolve().parent)'
)"
python_site_packages="$(
  python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)"
# Official Linux wheels changed from the old libstdc++ ABI in 3.7 to the
# C++11 ABI in 3.8. Inspect a CasADi symbol rather than guessing by version.
casadi_cxx11_abi="$(python - "$casadi_package_dir/libcasadi.so" <<'PY'
import subprocess
import sys

symbols = subprocess.check_output(["nm", "-D", "-C", sys.argv[1]], text=True)
matches = [line for line in symbols.splitlines() if " T casadi::MX::_sym(" in line]
if len(matches) != 1:
    raise SystemExit("Cannot determine the CasADi libstdc++ ABI from MX::_sym")
print(int("std::__cxx11::" in matches[0]))
PY
)"
echo "Detected CasADi _GLIBCXX_USE_CXX11_ABI=$casadi_cxx11_abi"
build_root="$(mktemp -d)"
trap 'rm -rf "$build_root"' EXIT

echo "Building RBDL-CasADi ${RBDL_COMMIT} against ${casadi_package_dir}"
git clone --quiet "$RBDL_REPOSITORY" "$build_root/rbdl"
git -C "$build_root/rbdl" checkout --quiet "$RBDL_COMMIT"
cmake \
  -S "$build_root/rbdl" \
  -B "$build_root/rbdl-build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$casadi_cxx11_abi -I$CONDA_PREFIX/include/eigen3" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DCasadi_DIR="$casadi_package_dir" \
  -DCasadi_INCLUDE_DIR="$casadi_package_dir/include/casadi" \
  -DCasadi_LIBRARY="$casadi_package_dir/libcasadi.so" \
  -DRBDL_BUILD_CASADI=ON \
  -DRBDL_BUILD_EXECUTABLES=OFF \
  -DRBDL_BUILD_TESTS=OFF
cmake --build "$build_root/rbdl-build" --target install --parallel 2

# RBDL installs both math backends. If the Eigen headers remain at
# $CONDA_PREFIX/include/rbdl, biorbd finds them before include/rbdl-casadi and
# silently compiles the CasADi backend against the wrong API.
mv "$CONDA_PREFIX/include/rbdl" "$build_root/rbdl-eigen-unused"

echo "Building biorbd ${BIORBD_TAG} against the same CasADi ABI"
git clone --quiet --branch "$BIORBD_TAG" --depth 1 "$BIORBD_REPOSITORY" "$build_root/biorbd"
git -C "$build_root/biorbd" apply "$SCRIPT_DIRECTORY/../patches/biorbd-casadi38.patch"
cmake \
  -S "$build_root/biorbd" \
  -B "$build_root/biorbd-build" \
  -G Ninja \
  -DBINDER_PYTHON3=ON \
  -DBIORBD_ALLOW_SWIG4_4=ON \
  -DBUILD_EXAMPLE=OFF \
  -DBUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
  -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=$casadi_cxx11_abi -I$CONDA_PREFIX/include/eigen3" \
  -DCasadi_DIR="$casadi_package_dir/cmake" \
  -DINSTALL_DEPENDENCIES_PREFIX="$CONDA_PREFIX" \
  -DMATH_LIBRARY_BACKEND=Casadi \
  -DMODULE_KALMAN=OFF \
  -DMODULE_STATIC_OPTIM=OFF \
  -DMODULE_VTP_FILES_READER=ON \
  -DPYTHON_EXECUTABLE="$(command -v python)" \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DSWIG_EXECUTABLE="$(command -v swig)" \
  -DPython3_SITELIB_INSTALL="$python_site_packages"
cmake --build "$build_root/biorbd-build" --target install --parallel 2

python - <<'PY'
import casadi
import biorbd_casadi as biorbd

# Import alone misses incompatible SWIG runtimes: test symbols in both directions.
q = casadi.MX.sym("q", 2)
converted = biorbd.GeneralizedCoordinates(q).to_mx()
assert isinstance(converted, casadi.MX)
assert converted.shape == (2, 1)
print(f"biorbd {biorbd.__version__}, CasADi {casadi.__version__}; MX round-trip passed")
PY
