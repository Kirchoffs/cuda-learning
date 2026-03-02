# Notes
## Commands
### Run
```
>> nvcc -o outputs/main main.cu
>> ./outputs/main
```

```
>> mkdir -p outputs && nvcc -o outputs/main main.cu && ./outputs/main; rm -rf outputs
```

### Report
#### Generate the report
```
>> ncu -o ./outputs/report.rpt ./outputs/main
```

or
```
>> sudo $(which ncu) -o ./outputs/report.rpt ./outputs/main
>> sudo `which ncu` -o ./outputs/report.rpt ./outputs/main
```

#### Check the report
```
>> ncu --import ./outputs/report.rpt.ncu-rep
```

## Basics
### CUDA Programming General Process
1. `cudaMalloc`: Allocate GPU memory
2. `cudaMemcpy (H -> D)`: Copy data from host to device
3. `kernel<<<...>>>`: Execute parallel kernel
4. `cudaMemcpy (D -> H)`: Copy data from device to host

### CUDA Kernel
A CUDA kernel is a GPU-resident function engineered for massive parallelism. Unlike standard sequential functions that run a single time, kernels are instantiated simultaneously across thousands of concurrent threads.

### Programming Model Perspective
```
Kernel Launch > Grid > Block > Thread
```

### Execution Model Perspective
```
GPU > SM > Block > Thread
```

### Memory Perspective
- Global Memory：visible to all threads in all blocks (whole device)
- Shared Memory：visible to all threads within the same block (lifetime = block lifetime)
- Registers：private to each individual thread

### Cache
Generally speaking, the essential differences between CPU caches (L1, L2, L3) are mainly reflected in "the distance from the core" and the resulting "speed, size, and degree of sharing".

#### Location and Physical Distance (The Most Fundamental Root)
- L1 Cache (Level 1): Closest to the CPU core, usually built inside the core and running at the same frequency as the core. It is the first stop for data the CPU needs.

- L2 Cache (Level 2): Located between L1 and L3. In older architectures, it was outside the core, but in modern architectures, it is also inside the core, though physically slightly farther away than L1.

- L3 Cache (Level 3): Farthest from the core. It is typically a large, shared cache pool located between multiple cores.

#### Speed and Latency
- L1: Extremely Fast. Access latency is typically 2-4 clock cycles. It needs to keep up with the CPU core's computation speed, aiming for almost zero latency.

- L2: Very Fast. Access latency is typically 10-20 clock cycles. Although slower than L1, it is still orders of magnitude faster than accessing main memory (RAM).

- L3: Fast. Access latency is typically 30-70 clock cycles. It is the last line of defense within the cache hierarchy. If data is found here, it avoids the need to access the significantly slower main memory.

#### Capacity
- L1: Extremely Small. Usually, each core has only around 64KB (split into 32KB for data + 32KB for instructions). Due to the extreme demands of speed and manufacturing technology, it cannot be made large.

- L2: Medium. In modern CPUs, each core typically has L2 cache ranging from 256KB to 1MB.

- L3: Relatively Large. Shared by all cores, its capacity usually ranges from a few MB to tens of MB (e.g., 16MB, 32MB).

#### Architecture and Sharing Relationship
- L1 and L2: Exclusive. Each CPU core has its own independent L1 and L2 cache. Core A cannot directly read the contents of Core B's L1 cache.

- Shared. It acts as a bridge between all cores. When Core A needs data that isn't in its own L1/L2 but might be in Core B's cache, the exchange needs to happen via L3 or through the cache coherence protocol.
