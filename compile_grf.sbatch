#!/bin/bash
#SBATCH --job-name=compile_grf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --exclude=erc-hpc-comp040
#SBATCH --output=/users/k25000843/compile_grf_%j.log
#
# Google Research Football build, no root, Conda env "epymarl_gfootball" (Python 3.9).
#
# Design rules (each one comes from a real failure in your logs, or from a risk I flag):
#   1. NEVER run `conda install` inside the epymarl env. Job 38257289 died inside that solve.
#      The GCC 12 toolchain lives in its OWN tiny env ($GRF_CC_ENV), created once.
#   2. Fail in seconds, not after a 20-minute compile: every assumption is preflight-tested.
#   3. Do not touch gym / torch / setuptools. numpy is only moved if pip needs it (<2.0).
#
# Optional one-time step on the LOGIN node (recommended, so you can watch the solve):
#   conda create -y -p $HOME/.conda/envs/grf_cc --override-channels -c conda-forge \
#       gcc_linux-64=12 gxx_linux-64=12 sysroot_linux-64=2.17
# If you skip it, the job creates it itself (with a 15 min timeout).
#
# Bypass the preflight compile/link test only if you know it is wrong:
#   sbatch --export=ALL,GRF_SKIP_PREFLIGHT=1 compile_grf.sbatch

eval "$(conda shell.bash hook)"
conda activate epymarl_gfootball
set -Eeo pipefail            # enabled AFTER activation (conda activate scripts dislike strict mode)
trap 'rc=$?; echo "!!! FAILED rc=$rc line=$LINENO node=$(hostname) cmd: $BASH_COMMAND" >&2' ERR

die()  { echo "ERROR: $*" >&2; exit 1; }
step() { echo; echo "==== [$(date +%H:%M:%S)] $* ===="; }

NCPU="${SLURM_CPUS_PER_TASK:-8}"
GRF_CC_ENV="${GRF_CC_ENV:-$HOME/.conda/envs/grf_cc}"
export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"

step "0. Sanity + snapshot"
echo "Node: $(hostname)   Env: $CONDA_PREFIX   CPUs: $NCPU"
[ "$(command -v python)" = "$CONDA_PREFIX/bin/python" ] || die "python is not the env's python: $(command -v python)"
[ "$(python -c 'import sys;print("%d.%d"%sys.version_info[:2])')" = "3.9" ] || die "env python is not 3.9"
[ -x "$CONDA_PREFIX/bin/cmake" ] || die "no cmake in $CONDA_PREFIX/bin"
echo "cmake: $("$CONDA_PREFIX/bin/cmake" --version | head -1)"
pip freeze > "$HOME/grf_pip_freeze_before_${SLURM_JOB_ID:-manual}.txt" 2>/dev/null || true
echo "pip freeze saved to ~/grf_pip_freeze_before_${SLURM_JOB_ID:-manual}.txt (rollback reference)"

# Libraries the engine links against must already be in the env (your earlier steps put them there)
for f in libboost_python39.so libboost_filesystem.so libSDL2.so libSDL2_image.so libSDL2_ttf.so libSDL2_gfx.so; do
  ls "$CONDA_PREFIX/lib/$f"* >/dev/null 2>&1 || die "missing $f in $CONDA_PREFIX/lib"
done
# Some Python builds ship only libpython3.9.so.1.0; the CMake hint below needs the plain .so
if [ ! -e "$CONDA_PREFIX/lib/libpython3.9.so" ]; then
  ls "$CONDA_PREFIX/lib/libpython3.9.so.1.0" >/dev/null 2>&1 || die "no libpython3.9 in env"
  ln -s libpython3.9.so.1.0 "$CONDA_PREFIX/lib/libpython3.9.so"
fi

# ---------------------------------------------------------------- 1. toolchain (separate env)
step "1. Toolchain: GCC 12 + glibc-2.17 sysroot in a separate env ($GRF_CC_ENV)"
CC_BIN="$GRF_CC_ENV/bin/x86_64-conda-linux-gnu-gcc"
CXX_BIN="$GRF_CC_ENV/bin/x86_64-conda-linux-gnu-g++"
if [ ! -x "$CXX_BIN" ]; then
  echo "Compiler env missing; creating it (fresh env, conda-forge only, 15 min timeout)..."
  if ! timeout 900 conda create -y -p "$GRF_CC_ENV" --override-channels -c conda-forge \
        "gcc_linux-64=12" "gxx_linux-64=12" "sysroot_linux-64=2.17"; then
    rm -rf "$GRF_CC_ENV"
    die "could not create $GRF_CC_ENV. Run the 'conda create' line from the top of this script on the login node, then resubmit."
  fi
fi
case "$("$CXX_BIN" -dumpversion)" in 12*) ;; *) die "$CXX_BIN is not GCC 12 ($("$CXX_BIN" -dumpversion))";; esac
echo "CXX: $("$CXX_BIN" --version | head -1)"
ls "$GRF_CC_ENV"/x86_64-conda-linux-gnu/sysroot/lib64/libc-*.so 2>/dev/null | sed 's/^/sysroot libc: /' || true

# Why sysroot 2.17: your first log installed sysroot_linux-64 2.39. Code compiled against glibc >= 2.38
# headers can reference GLIBC_2.38 symbols (e.g. __isoc23_strtol) that an older host glibc lacks.
# Step 9 checks the built .so against this node's glibc, so that is verified rather than assumed.

# ---------------------------------------------------------------- 2. flags (do not inherit the env's GCC-15 flags)
step "2. Build flags"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS DEBUG_CFLAGS DEBUG_CXXFLAGS DEBUG_CPPFLAGS CMAKE_ARGS CONDA_BUILD_SYSROOT
export CC="$CC_BIN" CXX="$CXX_BIN"
export CFLAGS="-O2 -fPIC -w -isystem $CONDA_PREFIX/include"
# -include cstdint: old engine code relies on <cstdint> being pulled in transitively
export CXXFLAGS="-O2 -fPIC -w -fpermissive -include cstdint -isystem $CONDA_PREFIX/include"
# RPATH (not RUNPATH) baked in so Boost/SDL2/libstdc++ resolve from the env at runtime with NO LD_LIBRARY_PATH
export LDFLAGS="-L$CONDA_PREFIX/lib -Wl,--disable-new-dtags -Wl,-rpath,$CONDA_PREFIX/lib"
# NOTE: LD_LIBRARY_PATH is deliberately NOT exported (your first log only showed harmless
# "no version information" warnings from curl and bash, but it is not needed with RPATH).

# ---------------------------------------------------------------- 3. EGL headers
step "3. EGL / KHR headers"
mkdir -p "$CONDA_PREFIX/include/EGL" "$CONDA_PREFIX/include/KHR"
BASE_URL=https://raw.githubusercontent.com/KhronosGroup/EGL-Registry/main/api
fetch() {  # fetch <path-under-api> <dest>
  if [ -s "$2" ] && [ "$(wc -c < "$2")" -gt 1000 ]; then echo "  have $2"; return; fi
  curl -fsSL --retry 3 "$BASE_URL/$1" -o "$2"
  [ "$(wc -c < "$2")" -gt 1000 ] || die "download of $1 looks truncated"
  echo "  fetched $1"
}
fetch EGL/egl.h          "$CONDA_PREFIX/include/EGL/egl.h"
fetch EGL/eglext.h       "$CONDA_PREFIX/include/EGL/eglext.h"
fetch EGL/eglplatform.h  "$CONDA_PREFIX/include/EGL/eglplatform.h"
fetch KHR/khrplatform.h  "$CONDA_PREFIX/include/KHR/khrplatform.h"

# ---------------------------------------------------------------- 4. host GL libs
step "4. Host GL libraries on $(hostname)"
SEARCH_DIRS="/usr/lib/x86_64-linux-gnu /usr/lib64 /lib64 /usr/lib /lib"
link_lib() {  # link_lib <soname> <dev-name>
  local real
  real=$(/sbin/ldconfig -p 2>/dev/null | awk -v n="$1" '$1==n && /x86-64/ {print $NF; exit}')
  [ -n "$real" ] && [ -e "$real" ] || real=$(find $SEARCH_DIRS -maxdepth 3 -name "$1" 2>/dev/null | head -n 1)
  [ -n "$real" ] || real=$(find $SEARCH_DIRS -maxdepth 3 -name "$1*" 2>/dev/null | head -n 1)
  if [ -n "$real" ]; then ln -sf "$real" "$CONDA_PREFIX/lib/$2"; echo "  $2 -> $real"
  else rm -f "$CONDA_PREFIX/lib/$2"; echo "  $2: NOT FOUND"; fi
}
link_lib libEGL.so.1    libEGL.so
link_lib libGL.so.1     libGL.so
link_lib libGLX.so.0    libGLX.so
link_lib libOpenGL.so.0 libOpenGL.so
for l in libEGL libGL; do
  [ -e "$CONDA_PREFIX/lib/$l.so" ] || die "$l.so missing on $(hostname). Use a node with the host GL stack (the gpu partition has it)."
done
export GRF_OPENGL_LIB="$CONDA_PREFIX/lib/libOpenGL.so"
[ -e "$GRF_OPENGL_LIB" ] || { echo "  (libOpenGL absent -> libGL)"; export GRF_OPENGL_LIB="$CONDA_PREFIX/lib/libGL.so"; }
export GRF_GLX_LIB="$CONDA_PREFIX/lib/libGLX.so"
[ -e "$GRF_GLX_LIB" ]    || { echo "  (libGLX absent -> libGL)";    export GRF_GLX_LIB="$CONDA_PREFIX/lib/libGL.so"; }

# ---------------------------------------------------------------- 5. cmake + make shims
step "5. cmake / make shims"
mkdir -p "$HOME/cmake_hack"

# make shim: build_game_engine.sh sizes -j from the WHOLE node (mp.cpu_count / free RAM), which would
# overrun --mem=32G. Strip any -j / --jobs the caller passes and force the SLURM allocation.
# (I could not download gfootball here to confirm exactly how it passes -j, so this strips every form.)
cat > "$HOME/cmake_hack/make" << 'EOF'
#!/bin/bash
REAL_MAKE=/usr/bin/make
[ -x "$CONDA_PREFIX/bin/make" ] && REAL_MAKE="$CONDA_PREFIX/bin/make"
[ -n "$GRF_REAL_MAKE" ] && REAL_MAKE="$GRF_REAL_MAKE"
N="${SLURM_CPUS_PER_TASK:-8}"
args=(); skip=0
for a in "$@"; do
  if [ "$skip" = 1 ]; then
    skip=0
    [[ "$a" =~ ^[0-9]+$ ]] && continue      # the number after a bare -j / --jobs
  fi
  case "$a" in
    -j)           skip=1; continue ;;
    --jobs)       skip=1; continue ;;
    -j[0-9]*)     continue ;;
    --jobs=*)     continue ;;
  esac
  args+=("$a")
done
exec "$REAL_MAKE" "-j$N" "${args[@]}"
EOF
chmod +x "$HOME/cmake_hack/make"

# cmake shim: pins compiler, Python 3.9, GL/EGL, flags. Only configure invocations get -D options.
cat > "$HOME/cmake_hack/cmake" << 'EOF'
#!/bin/bash
CM="$CONDA_PREFIX/bin/cmake"
case "$1" in
  --build|--install|--open|--version|--help|--find-package|-E|-P) exec "$CM" "$@" ;;
esac
exec "$CM" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \
  -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \
  -DOPENGL_opengl_LIBRARY="${GRF_OPENGL_LIB:-$CONDA_PREFIX/lib/libOpenGL.so}" \
  -DOPENGL_glx_LIBRARY="${GRF_GLX_LIB:-$CONDA_PREFIX/lib/libGLX.so}" \
  -DOPENGL_gl_LIBRARY="$CONDA_PREFIX/lib/libGL.so" \
  -DOPENGL_INCLUDE_DIR="$CONDA_PREFIX/include" \
  -DEGL_LIBRARY="$CONDA_PREFIX/lib/libEGL.so" \
  -DEGL_opengl_LIBRARY="$CONDA_PREFIX/lib/libGL.so" \
  -DEGL_INCLUDE_DIR="$CONDA_PREFIX/include" \
  -DEGL_INCLUDE_DIRS="$CONDA_PREFIX/include" \
  -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
  -DCMAKE_C_FLAGS="$CFLAGS" \
  -DCMAKE_SHARED_LINKER_FLAGS="$LDFLAGS" \
  -DCMAKE_EXE_LINKER_FLAGS="$LDFLAGS" \
  -DPython_ROOT_DIR="$CONDA_PREFIX" \
  -DPython_FIND_STRATEGY=LOCATION \
  -DPython_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPython_INCLUDE_DIR="$CONDA_PREFIX/include/python3.9" \
  -DPython_LIBRARY="$CONDA_PREFIX/lib/libpython3.9.so" \
  -DPython3_ROOT_DIR="$CONDA_PREFIX" \
  -DPython3_FIND_STRATEGY=LOCATION \
  -DPython3_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python" \
  -DPYTHON_INCLUDE_DIR="$CONDA_PREFIX/include/python3.9" \
  -DPYTHON_LIBRARY="$CONDA_PREFIX/lib/libpython3.9.so" \
  "$@"
EOF
chmod +x "$HOME/cmake_hack/cmake"
export PATH="$HOME/cmake_hack:$PATH"
echo "make shim: $(command -v make)   cmake shim: $(command -v cmake)"

# ---------------------------------------------------------------- 6. preflight (seconds)
step "6. Preflight"
PF=$(mktemp -d)
trap 'rm -rf "$PF"' EXIT

echo "6a. CMake resolves Python 3.9 + Boost.Python 3.9 (your first failure was python3.10)"
cat > "$PF/CMakeLists.txt" << 'CMK'
cmake_minimum_required(VERSION 3.12)
project(preflight CXX)
find_package(Python COMPONENTS Development)
message(STATUS "PF_PYTHON_VERSION=${Python_VERSION}")
find_package(Boost CONFIG REQUIRED COMPONENTS python${Python_VERSION_MAJOR}${Python_VERSION_MINOR})
message(STATUS "PF_BOOST_PYTHON_OK")
CMK
mkdir -p "$PF/cm" && ( cd "$PF/cm" && cmake .. ) > "$PF/cm.txt" 2>&1 || { cat "$PF/cm.txt"; die "preflight cmake configure failed"; }
grep -q "PF_PYTHON_VERSION=3.9" "$PF/cm.txt" && grep -q "PF_BOOST_PYTHON_OK" "$PF/cm.txt" \
  || { cat "$PF/cm.txt"; die "CMake did not resolve Python 3.9 + Boost.Python 3.9"; }
grep -E "PF_PYTHON_VERSION|Found Python|CXX compiler identification" "$PF/cm.txt"

if [ "${GRF_SKIP_PREFLIGHT:-0}" != 1 ]; then
  echo "6b. Compile + link a probe .so with GCC 12 against the env's EGL / SDL2 / Boost.Python"
  cat > "$PF/probe.cpp" << 'CPP'
#include <cstdint>
#include <EGL/egl.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>
#include <boost/python.hpp>
#include <boost/filesystem.hpp>
int probe(int n) {
  SDL_Init(0);
  EGLDisplay d = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  const GLubyte* s = glGetString(GL_VERSION);
  boost::filesystem::path p("/tmp");
  if (n == 42) boost::python::throw_error_already_set();
  return (int)(intptr_t)d + (s ? 1 : 0) + (int)p.size();
}
CPP
  if ! $CXX $CXXFLAGS -I"$CONDA_PREFIX/include/python3.9" -shared -fPIC "$PF/probe.cpp" -o "$PF/probe.so" \
        $LDFLAGS -lboost_python39 -lboost_filesystem -lSDL2 -lEGL -lGL > "$PF/probe.txt" 2>&1; then
    cat "$PF/probe.txt"; die "probe compile/link failed (see above). Nothing was wasted on a full build."
  fi
  "$GRF_CC_ENV/bin/x86_64-conda-linux-gnu-readelf" -d "$PF/probe.so" | grep -E "NEEDED|RPATH|RUNPATH" | sed 's/^/  /'
fi
echo "Preflight OK."

# ---------------------------------------------------------------- 7. build gfootball
step "7. Build gfootball (no deps, so pip cannot move numpy/gym/torch)"
python -c "import setuptools, wheel, psutil" 2>/dev/null || pip install --no-deps wheel psutil
pip install --no-cache-dir --no-build-isolation --no-deps "gfootball==2.10.2"

# ---------------------------------------------------------------- 8. runtime python deps
step "8. Python deps (numpy kept < 2.0; opencv headless < 4.12 because 4.12+ requires numpy 2)"
CONS="$(mktemp)"
printf '%s\n' 'numpy>=1.22.4,<2' 'opencv-python-headless<4.12' > "$CONS"
# numpy>=1.22.4 is what scipy 1.13 needs; the env currently has conda's numpy 1.19.5 from your first run.
pip install --no-cache-dir -c "$CONS" "pygame>=1.9.6" "opencv-python-headless==4.10.0.84" scipy absl-py psutil \
  || pip install --no-cache-dir --ignore-installed numpy -c "$CONS" "pygame>=1.9.6" "opencv-python-headless==4.10.0.84" scipy absl-py psutil
# gym: left alone if present. If absent, 0.26.2 is what pip itself chose in your first log.
# Install whatever gym version your MARL repo pins AFTERWARDS.
python -c "import gym" 2>/dev/null || pip install --no-cache-dir -c "$CONS" "gym==0.26.2"
python -c "import numpy; print('numpy', numpy.__version__)"

# ---------------------------------------------------------------- 9. verify the binary
step "9. Verify the built engine"
SP="$(python -c 'import site;print(site.getsitepackages()[0])')"
SO="$(find "$SP" -name '_gameplayfootball*.so' | head -n 1)"
[ -n "$SO" ] || die "no _gameplayfootball*.so found under $SP"
echo "engine: $SO"

echo "-- unresolved shared libraries (must be empty):"
( ldd "$SO" | grep "not found" ) || echo "  none"
echo "-- libEGL / libGL / libstdc++ resolve to:"
ldd "$SO" | grep -E "libEGL|libGL|libstdc\+\+" | sed 's/^/  /'

NEED="$("$GRF_CC_ENV/bin/x86_64-conda-linux-gnu-objdump" -T "$SO" | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1 | sed 's/GLIBC_//')"
HAVE="$(ldd --version | head -1 | grep -o '[0-9]*\.[0-9]*$')"
echo "-- highest GLIBC symbol needed: ${NEED:-none}   host glibc: $HAVE"
if [ -n "$NEED" ] && [ "$(printf '%s\n%s\n' "$NEED" "$HAVE" | sort -V | tail -1)" != "$HAVE" ]; then
  die "engine needs GLIBC_$NEED but this node has $HAVE: the sysroot is wrong, rebuild the compiler env with sysroot_linux-64=2.17"
fi

echo "-- smoke test (from a neutral directory, no LD_LIBRARY_PATH):"
cd "$(mktemp -d)"
python - << 'PY' || echo "WARNING: smoke test failed. The BUILD succeeded; read the traceback above (gym-version / EGL-on-this-node issues show up here)."
import gfootball
import gfootball.env as football_env
print("gfootball:", gfootball.__file__)
env = football_env.create_environment(
    env_name="academy_run_pass_and_shoot_with_keeper",
    representation="simple115",
    rewards="scoring",
    render=False,
    number_of_left_players_agent_controls=2,
)
obs = env.reset()
obs, rew, done, info = env.step([0, 0])
print("OK: env created, reset and stepped. obs shape:", getattr(obs, "shape", type(obs)))
env.close()
PY

echo; echo "Done at $(date)."
