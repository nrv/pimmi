Python 3.8.0 (default, Mar  3 2022, 13:36:28) 

Type 'copyright', 'credits' or 'license' for more information

IPython 8.4.0 -- An enhanced Interactive Python. Type '?' for help.

PyDev console: using IPython 8.4.0

Python 3.8.0 (default, Mar  3 2022, 13:36:28) 

[GCC 9.3.0] on linux

```python
from main import *

%timeit taynaud_community()

> 4.64 ms ± 59.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit nx_community()

> 3.5 ms ± 20 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit igraph_community()

> 168 µs ± 325 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)

%timeit leidenalg_community()

> 383 µs ± 4.36 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
