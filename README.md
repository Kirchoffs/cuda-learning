# Notes
## Run
```
>> nvcc -o outputs/main main.cu
>> ./outputs/main
```

```
>> mkdir -p outputs && nvcc -o outputs/main main.cu && ./outputs/main; rm -rf outputs
```

## Basics
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
