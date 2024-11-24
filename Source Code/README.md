# A Hybrid CPU-GPU Framework with Multithreading, SIMD, and Evaluation of Efficiency Metrics
## Dependencies:
- [gcc](https://gcc.gnu.org/)
- [glib](https://docs.gtk.org/glib/)
- [intel-oneapi-dpcpp-cpp](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [make](https://www.gnu.org/software/make/)
- [pkg-config](https://freedesktop.org/wiki/Software/pkg-config/)
- [python](https://www.python.org/)
- - -
## Prerequisites:
```bash
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade matplotlib numpy pandas pip tensorflow tensorflow_datasets
deactivate
```
- - -
## Building:
```bash
make clean
make
```
Intel:
```bash
make clean
make intel
```
XL:
```bash
make clean
make xl
```
XL and Intel:
```bash
make clean
make xl_intel
```
- - -
## Debugging:
```bash
make clean
make debug
source ./venv/bin/activate
gdb -ex run ./build/tf
valgrind --leak-check=full --show-leak-kinds=all ./build/tf
deactivate
```
Intel:
```bash
make clean
make debug_intel
source ./venv/bin/activate
gdb -ex run ./build/tf
valgrind --leak-check=full --show-leak-kinds=all ./build/tf
deactivate
```
- - -
## Running:
> **! Make sure you are in the Source Code directory !**
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
Including Intel:
```bash
source ./venv/bin/activate
make benchmark_intel
deactivate
```
