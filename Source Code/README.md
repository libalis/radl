# A Hybrid CPU-GPU Framework with Multithreading, SIMD, and Evaluation of Efficiency Metrics
- - -
## Dependencies:
- [gcc](https://gcc.gnu.org/)
- [glib](https://docs.gtk.org/glib/)
- [intel-oneapi-dpcpp-cpp](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
- [make](https://www.gnu.org/software/make/)
- [pkg-config](https://freedesktop.org/wiki/Software/pkg-config/)
- [python](https://www.python.org/)
- - -
## Building:
```sh
make clean
make
```
Intel:
```sh
make clean
make intel
```
XL:
```sh
make clean
make xl
```
XL and Intel:
```sh
make clean
make xl_intel
```
- - -
## Debugging:
```sh
make clean
make debug
gdb -ex run ./build/tf
valgrind --leak-check=full --show-leak-kinds=all ./build/tf
```
Intel:
```sh
make clean
make debug_intel
gdb -ex run ./build/tf
valgrind --leak-check=full --show-leak-kinds=all ./build/tf
```
- - -
## Running:
> **! Make sure you are in the Source Code directory !**
```sh
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip tensorflow tensorflow_datasets
./build/tf
deactivate
```
- - -
## Benchmarking:
```sh
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip tensorflow tensorflow_datasets
make benchmark
deactivate
```
Including Intel:
```sh
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip tensorflow tensorflow_datasets
make benchmark_intel
deactivate
```
