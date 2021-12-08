# PyAPL
(Partial) Implmentation of [APL](https://aplwiki.com/wiki/Main_Page) on top of Python--Python APL EDSL

School Project, Dec 2021.

## Core Design
### APL Array
The `APLArray` object is the core part of this implementation. Unlike typical N-dementional list in Python, which 
literally puts lists inside of lists N times to accomplish, APLArray seperates the demention from it's data. This means
that it keeps two arrays, one for data and other for it shape. This has to be done because the "typical" N-dementional list
in python have no capability to extract its shape (in a sane way), no easy way to transform shapes, and provided not enough 
abstractions. 

Later, turns how this way of showing N-dementional Array in used in [Arther Witney's original one-page 
interpreter](https://code.jsoftware.com/wiki/Essays/Incunabulum), K Language, a modern APL dialect, and 
also by Numpy. 


In Arther Witney's original one-page interpreter, 
```c
typedef struct a{I t,r,d[3],p[2];}*A;
```
is almost equivlant to this project's `APLArray` object: `d` is data, `r` is rank, `p` is shape; 
`.arr` is data, `len(.shape)` is rank, `.shape` is shape in `APLArray`.

Arther Witney's terse C code is very cryptic, but, thankfully, I was able to understand his implmentation of 
Array. 

It's a big suprise as I did not researched the implementations of vector programming language in
forehand, yet still yielded similar fundumental implmentations.

### Dy/Monadic function
Monad here is not Monad from the type theory that provides binding. Monad in APL means an operator
with only single inputs Array. So `* 5` is a monadic use of `*` operator; it will return `e^5`. 

Dyadic, as its name implys, have two inputs Arrays, from left and right. So `2 * 5` is a dyadic use
of `*` operator; it will return `2^5`. 

As shown, most of APL Operators can be used both monadically and dyadically. For example, `÷` is
reciprocal monadically and divide dyadically; `⌽` is "Reverse" monadically and "Rotate" dyadically; and 
`↓` is "Split" monadically and "Drop" dyadically. 

Thanks to the flexibility of Python functions and optional arguments, APL Operators was implmented
cleanly in one symbol like those of APL. 

## Concurrency
An attempts to implement concurrent operations in the Core indexing function and Array operators was made, but most of the operations
were bottlenecked by the BinOperations, by the fact that Python does not support Tail Recursion Optimization, and by the delay
on pool allocation. There were no improvements in the performance; rather, it mostly harmed the performance. 


There was a brief consideration of implementing core in C with ffi interfacing, but the ctype and cffi documentation
proved itself to be a great pain; struggling in untyped maddness of Python was enough for me.

So, at this point, there is nothing can be done over the matter of performance. It is just a comprehensive implementation 
of APL (partial) as a Python EDSL. Personally, if it was implemented in more strongly typed and static language with matured 
supports over optimization of recursive calls, the performance would have been tolerable. 


## Applications
Like traditional APL, APL provides convenience on manupulating complex data by providing
framework for array operations like inner products, outer products, rotate, and drop. 

Long, uncomprehensive and overly verbose code that does simple data manupulation is only hindering one
from understanding the tasks code does. APL, on the other hand, compresses these operations
and it expressive to the readers. This EDSL provides same expressive operations on top of pure
Python, allowing one to write same terse code in Python. 

Famous [Conways' Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) is implemented in
PyAPL as `Example_GameOfLife.py` as an example. Again, it is pretty slow: each generations takes `~ 0.1 Seconds`. But, nothing can be done. 

Other simple examples, like Average of an array, can be found on APL wiki, then be translated into PyAPL.

## Tests
Some test cases are provided in `test_APL.py` with some array operations and type checkings. Sadly, did not
have enough time to generate longer test cases, but it will do the job for now. 