# PyAPL

https://aplwiki.com/wiki/Main_Page

## Concurrency
An attempts to implement concurrent operations in the Core indexing function and Array operators, but most of the operations
were bottlenecked by the BinOperations, by the fact that Python does not support Tail Recursion Optimization, and by the delay
on pool allocation. There were no improvements in the performance; rather, it mostly harmed the performance. 


There was a brief consideration of implementing core in C with ffi interfacing, but the ctype and cffi documentation
proved itself to be a great pain. 

So, at this point, there is nothing can be done over the matter of performance. It is just a comprehensive implementation 
of APL (partial) as a Python EDSL. Personally, if it was implemented in more strongly typed and static language with matured 
supports over optimization of recursive calls, the performance would have been tolerable. 

## Core Design
The `APLArray` object is the core part of this implementation. Unlike typical N-dementional list in Python, which 
literally puts lists inside of lists N times to accomplish, APLArray seperates the demention from it's data. This means
that is keeps two arrays, one for data and other for it shape. This has to be done because the "typical" N-dementional list
in python have no capability to extract its shape (in a sane way), no easy way to transform shapes, and provided not enough 
abstractions. 

Later, turns how this way of showing N-dementional Array in used in [Arther Witney's original one-page 
interpreter](https://code.jsoftware.com/wiki/Essays/Incunabulum), K Language, a modern APL dialect, and 
also by Numpy. 


In Arther Witney's original one-page interpreter, 
```c
typedef struct a{I t,r,d[3],p[2];}*A;
```
is almost equivlant to this project's `APLArray` object. `d` is data, `r` is rank, `p` is shape; 
`.arr` is data, `len(.shape)` is rank, `.shape` is shape in `APLArray`.
