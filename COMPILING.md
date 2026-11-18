# Compiling UAMMD-structured as a portable bundle

This guide builds `UAMMDlauncher` on any Linux x86-64 machine (no GPU needed) and packs it into a
self-contained folder that can be copied (`scp -r`) to a GPU cluster and run there without installing
anything. All build tools come from conda-forge, so no root access and no system CUDA are required.

Tested with CUDA 12.3, gcc 11, CMake 4.x and Ninja, building for `sm_86` (RTX 3090 / A40).

## 1. What you need to know about the target machine

Run these on a GPU node of the cluster where the program will run:

```bash
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv   # GPU model, compute capability, driver
nvidia-smi | grep "CUDA Version"                                       # newest CUDA the driver supports
ldd --version | head -1                                                # glibc version
```

From this you choose three things:

| Setting | Rule | Example |
|---|---|---|
| CUDA toolkit version | must be **≤** the "CUDA Version" printed by `nvidia-smi` | driver shows 12.3 → use CUDA ≤ 12.3 |
| GPU architecture(s) | compute capability without the dot; list all GPU types you will use | 3090/A40 → `86`; V100 → `70`; A100 → `80`; L40/4090 → `89`; H100 → `90` |
| glibc (sysroot) | must be **≤** the cluster's glibc | cluster has 2.28 → `sysroot_linux-64=2.28` |

Several architectures can be built at once (`"70;86"`); each one adds compile time and size.
Compiling only for the architectures you need keeps the build fast.

## 2. Create the build environment (once)

Any conda-compatible tool works; the examples use `micromamba`.

```bash
micromamba create -n uammd-cuda -c conda-forge \
    cuda-version=12.3 cuda-nvcc cuda-libraries-dev \
    gcc=11 gxx=11 sysroot_linux-64=2.28 \
    cmake ninja make git \
    libblas liblapacke blas-devel \
    patchelf
```

Notes:
- `cuda-version` pins the whole CUDA toolkit (compiler and libraries) to that release.
- Pinning `sysroot_linux-64` is essential: without it the solver may pick a recent glibc and the
  resulting binary fails on older clusters with `GLIBC_2.xx not found`.
- `patchelf` is only needed for the bundle step (section 4).

## 3. Configure and build

```bash
git clone https://github.com/saassenza/UAMMD-structured.git
micromamba activate uammd-cuda            # or prefix the commands with: micromamba run -n uammd-cuda

cmake -S UAMMD-structured -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
    -DCMAKE_INSTALL_PREFIX="$PWD/install" \
    -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib'
ninja -C build -j4
ninja -C build install
```

- `-j4`: each CUDA compilation can use a few GB of RAM; raise it only if you have memory to spare.
  With `-j4` the full build takes roughly 15 minutes on a desktop CPU (most of it in the final link,
  which uses link-time optimisation).
- Do not add `-march=native` or similar host flags: the binary must run on the cluster's CPUs,
  which may differ from the build machine.
- The components that get compiled are listed in `structured/Components.json`.
- The configure step rewrites a few files in `structured/include/Definitions/` (whitespace only).
  If `git status` shows them as modified, restore them with
  `git checkout -- structured/include/Definitions/`.

The result is `install/bin/UAMMDlauncher` plus `install/lib/libuammd_structured.so`.

### CUDA runtime

`CMakeLists.txt` sets `CMAKE_CUDA_RUNTIME_LIBRARY` to `Shared` by default. Do not switch it back to
`Static` while `libuammd_structured.so` is a separate shared library: the launcher and the library would
each carry their own copy of the CUDA runtime, some kernels get launched through the wrong copy, and the
simulation stops at the first neighbour-list build with `an illegal memory access was encountered`
(error 700). With `Shared`, `libcudart.so` must be available at run time; the bundle below includes it.

## 4. Make a relocatable bundle

The installed binary still depends on libraries from the conda environment (CUDA runtime, cuFFT, cuBLAS,
OpenBLAS, libstdc++, ...). The script below copies them next to the program and sets relative library
search paths, so the folder works wherever it is copied. The NVIDIA driver library (`libcuda.so`) is
deliberately **not** copied: it must come from the GPU node.

Save as `make_bundle.sh`, then run it inside the activated environment:
`./make_bundle.sh install UAMMD-structured_bundle 2.28` (the last argument is the cluster's glibc).

```bash
#!/bin/bash
# Usage: ./make_bundle.sh <install_prefix> <bundle_dir> [max_glibc]
set -euo pipefail
INST=$1; D=$2; MAXG=${3:-2.28}
ENV=${CONDA_PREFIX:?activate the build environment first}

rm -rf "$D"; mkdir -p "$D/bin" "$D/lib"
cp "$INST/bin/UAMMDlauncher" "$D/bin/"
cp "$INST/lib/libuammd_structured.so" "$D/lib/"
# copy every dependency that resolves into the conda environment (glibc itself stays on the target)
for l in $(env -u LD_LIBRARY_PATH ldd "$INST/bin/UAMMDlauncher" | awk -v e="$ENV" 'index($3,e)==1{print $3}'); do
  cp -L "$l" "$D/lib/$(basename "$l")"
done
# relative search paths: bin/ looks in ../lib, libraries look next to themselves
patchelf --set-rpath '$ORIGIN/../lib' "$D/bin/UAMMDlauncher"
for f in "$D"/lib/*.so*; do patchelf --set-rpath '$ORIGIN' "$f"; done

# checks
echo "--- libraries resolved outside the bundle (should only be glibc parts and libcuda):"
env -i PATH=/usr/bin:/bin ldd "$D/bin/UAMMDlauncher" | grep -v "$D" || true
need=$(for f in "$D"/bin/* "$D"/lib/*; do objdump -T "$f" | grep -o 'GLIBC_[0-9.]*'; done | sed 's/GLIBC_//' | sort -uV | tail -1)
echo "--- highest glibc required: $need (target: $MAXG)"
if [ "$(printf '%s\n%s\n' "$need" "$MAXG" | sort -V | tail -1)" != "$MAXG" ]; then
  echo "ERROR: bundle needs glibc $need > $MAXG"; exit 1
fi
du -sh "$D"
```

The bundle is about 1.6 GB, mostly CUDA math libraries.

## 5. Run on the cluster

```bash
scp -r UAMMD-structured_bundle cluster:~/sw/
# on a GPU node, in the folder with simulation.json:
~/sw/UAMMD-structured_bundle/bin/UAMMDlauncher simulation.json
```

No `module load` or `LD_LIBRARY_PATH` is needed. A quick check that the bundle starts (works on any
node, even without a GPU): running `UAMMDlauncher` without arguments prints
`[CRITICAL] No input file provided!` (followed by an "unrecoverable error" exit, which is expected here).
A `GLIBC` or `cannot open shared object file` message instead means the bundle does not fit the machine.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `GLIBC_2.xx not found` | built against a newer glibc than the cluster has: pin `sysroot_linux-64` to the cluster's version and rebuild |
| `no kernel image is available for execution on the device` | the GPU's architecture was not in `CMAKE_CUDA_ARCHITECTURES` |
| `CUDA driver version is insufficient for CUDA runtime version` | toolkit newer than the driver supports: use an older `cuda-version` |
| `illegal memory access` (700) at the first step | see "CUDA runtime" above; for any other kernel error rerun with `CUDA_LAUNCH_BLOCKING=1` so the message names the kernel that failed |
| `Illegal instruction` | host code built with CPU-specific flags (`-march=native`): rebuild without them |

Note on debugging: UAMMD-structured runs the simulation in a forked child process, so tools such as
`compute-sanitizer` may not attach to it; `CUDA_LAUNCH_BLOCKING=1` is the simplest way to locate a
failing kernel. Paths of the build machine that appear in error messages are compile-time source
file names and do not need to exist on the cluster.
