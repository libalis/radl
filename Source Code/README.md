A Hybrid CPU-GPU Framework with Multithreading, SIMD, and Evaluation of Efficiency Metrics
- - -
Dependencies:
- [make](https://www.gnu.org/software/make/)
- [gcc](https://gcc.gnu.org/)
- [pkg-config](https://freedesktop.org/wiki/Software/pkg-config/)
- [glib](https://docs.gtk.org/glib/)
- [python](https://www.python.org/)
- - -
Building:
```sh
make clean
make
```
- - -
Debugging:
```sh
make clean
make debug
gdb -ex run ./build/tf
valgrind --leak-check=full --show-leak-kinds=all ./build/tf
```
- - -
Running:
> **! Make sure you are in the Source Code directory !**
```sh
python -m venv ./venv
source ./venv/bin/activate
pip install --upgrade pip tensorflow tensorflow_datasets
./build/tf
deactivate
```
