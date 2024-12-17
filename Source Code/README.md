# A Hybrid CPU-GPU Framework with Multithreading, SIMD, and Evaluation of Efficiency Metrics
## Dependencies:
- [clang](https://clang.llvm.org/)
- [cuda](https://developer.nvidia.com/cuda-zone)
- [dialog](https://invisible-island.net/dialog/)
- [glib](https://docs.gtk.org/glib/)
- [intel-oneapi-dpcpp-cpp](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [make](https://www.gnu.org/software/make/)
- [openmp](https://openmp.llvm.org/)
- [pkg-config](https://freedesktop.org/wiki/Software/pkg-config/)
- [python](https://www.python.org/)
- - -
## Prerequisites:
### Arch Linux:
```bash
pacman -S clang cuda dialog glib intel-oneapi-dpcpp-cpp make openmp pkg-config python
```
### macOS:
```bash
brew install dialog glib libomp pkg-config
```
### Generally:
After installing the relevant dependencies, create a virtual environment and install the required python packages:
```bash
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade matplotlib numpy pandas pip tensorflow tensorflow_datasets
deactivate
```
- - -
## Building:
```bash
make config
```
- - -
## Debugging:
```bash
source ./venv/bin/activate
gdb -ex "set debuginfod enabled on" -ex run ./build/tf
valgrind --leak-check=full --show-leak-kinds=all ./build/tf
gprof ./build/tf gmon.out > gmon.txt
deactivate
```
- - -
## Running:
```bash
source ./venv/bin/activate
./build/tf
deactivate
```
- - -
## Benchmarking:
```bash
source ./venv/bin/activate
make benchmark
deactivate
```
Including ICPX:
```bash
source ./venv/bin/activate
make benchmark_intel
deactivate
```
Including ICPX and CUDA:
```bash
source ./venv/bin/activate
make benchmark_nvidia
deactivate
```
