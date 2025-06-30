# Cell Framework Example with C++23 Modules

This example demonstrates the Cell framework using a modules-first design.
The build system uses CMake 3.26+ with the experimental C++ module API and
automatically generates traditional headers from module interface files.

## Building

```bash
mkdir build && cd build
cmake ..
cmake --build .
./cell_app
```

During configuration, module interfaces located in the `Cell/` directory are
converted into headers under `build/include/Cell`. These generated headers
allow interoperability with code that still relies on the traditional `#include`
mechanism.
