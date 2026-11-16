# Strong chromatic index certificates

This archive contains the certificates and verification code for the
$1.642\Delta^2$ upper bound on the strong chromatic index.

Run from this directory.

~~~sh
python3 -B verify_all.py
~~~

The default verifier requires Python 3.10 or later and uses only the
standard library. It checks the file checksums and certificates, then
verifies the palette calculation. A successful run ends with
`All certificate checks passed.`

The file `palette_bounds.txt` contains the bounds for the 949 phases
and the completion step. The Python verifier checks every proposed bound by
subdividing the relevant intervals and recomputes the weight updates
and palette total with exact integers.

Use `--appendix C` to verify just the palette calculation and
`--jobs 1` to run it in one process. The default uses up to eight
processes. The optional `--cpp` flag replays the original C++ search.
That flag requires GNU g++ with C++20, OpenMP, and signed 128-bit
integer support.

The local flag basis and SDPA coefficients use the implementation of
Davey, Hurley, de Joannis de Verclos, Kang, and Volec at
[commit 3a429d27d1c6](https://github.com/rossjkang/localflagalgebras/tree/3a429d27d1c6a365f985f2342e43353c934b6563).
See [UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt).
