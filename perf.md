1. Baseline
```
===== Timings =====
Levels   1 - 10: 1.768s
Levels  11 - 20: 0.790s
Levels  21 - 30: 9.553s
Levels  31 - 40: 29.506s
Levels  41 - 50: 4.495s
Levels  51 - 60: 147.716s
Levels  61 - 70: 3.336s
Levels  71 - 71: 0.469s
Total          : 197.632s
```
2. After trivial optos (no lambdas, no excess BinOp allocs):
```
===== Timings =====
Levels   1 - 10: 0.637s
Levels  11 - 20: 1.604s
Levels  21 - 30: 9.130s
Levels  31 - 40: 3.556s
Levels  41 - 50: 1.173s
Levels  51 - 60: 21.613s
Levels  61 - 70: 3.045s
Levels  71 - 71: 0.415s
Total          : 41.174s
```

3. 