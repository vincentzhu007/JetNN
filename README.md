# JetNN
JetNN is a neural network training/inference framework. It aims to provide a simple and fluent deep learning toolkit for users.

### Build From Source

1. Clone git repository

```shell
git clone https://github.com/VincentZhu007/JetNN.git
```

2. Build it

```shell
cdJetNN
mkdir build && cd build
cmake .. && make -j
```

3. Run build-in tests (recommended)

```shell
./CTest
```

4. Install distributed files (optional)

```shell
mkdir output
cmake clean && cmake -DCMAKE_INSTALL_PREFIX=output/ ..
make install
```

### TODO
 - have no idea how to compose a cool nn framework, so let's see what can do.

