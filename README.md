## Uneval

This is the solution for [Uneval](https://2026.andgein.ru/editorial?lang=en#uneval) problem
aka "Can you generate an arithmetic* Python expression that `eval`s into a target number with given symbols and length constraints?"

`*` -- well, if you consider things like `True<<~(~-~True<<-~True)^-~(-~-~True<<-~True)` and `~(()==())*-~-~-~-((-~-~-~-~-~-~(()==()))**-~-~(()==()))` to be arithmetic. Oh, and `~(not[])<<~-~(not[])`

### What's inside?

Tests are in `testdata.txt` -- `targetNumber maxLen prohibitedSymbols`.
There is no golden solution and the algorithm does not try to find the minimal one,
so for, let's say `4 5 023456789` both `1+1+1+1` and `1-~1+1` are accepted

You can run all tests with and look at all the solutions with `python uneval.py`.
Under the hood it's ~~five nested for loops~~, well, an expression generator. 
It's optimized a bit more than it should've been, but it's kind of the point.

Optimizations:
* Meet-in-the-middle (basically, for any already existing expression, try to find a complement one amongst already existing)
* Some length/value-based branch-and-bound pruning
* `isinstance` avoidance because it is unreasonably slow even for union types
