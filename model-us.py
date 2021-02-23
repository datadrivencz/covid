#!/usr/bin/env python3

import math
import scipy.optimize as optimize

from libsimple import *

STEP_DAYS = 5
POPULATION = 330000000

numTests = [
  51740.5714285714,
  60149.2857142857,
  65468,
  66507,
  64830.4285714286,
  60186.5714285714,
  53607.7142857143,
  53042.5714285714,
  50835.1428571429,
  42956.8571428572,
  41088.7142857143,
  41531.2857142857,
  39670.2857142857,
  34451,
  38066.7142857143,
  40699.8571428571,
  44086.8571428571,
  42838.7142857143,
  43493,
  48946.2857142857,
  54557.5714285714,
  59291.7142857143,
  69436.8571428571,
  79146.1428571429,
  92685.8571428572,
  119171.142857143,
  146379.571428571,
  165825.428571429,
  172539.285714286,
  158213.857142857,
  186095.857142857,
  205647,
  228374.857142857,
  214939.142857143,
  210362.53968254,
  210880,
  212546.857142857,
  247156.428571429,
  234593.428571429,
  197930.285714286,
  167494.857142857,
  152507.857142857,
  157309,
  133992.571428571,
  93741]

numDeaths = [
  540.142857142857,
  715.857142857143,
  738.142857142857,
  854.142857142857,
  1023.85714285714,
  1133.28571428571,
  1056,
  1080,
  1067.28571428571,
  989.714285714286,
  948.714285714286,
  904.857142857143,
  837.142857142857,
  746,
  860.428571428571,
  764.142857142857,
  751.571428571429,
  704,
  687.285714285714,
  698,
  678.142857142857,
  759,
  796,
  809.571428571429,
  933.142857142857,
  992,
  1127.57142857143,
  1415.42857142857,
  1643.85714285714,
  1454,
  2159.57142857143,
  2332.42857142857,
  2565,
  2633.14285714286,
  2425.71428571429,
  2287.28571428571,
  2640.85714285714,
  3121.71428571429,
  3317.42857142857,
  2997,
  3049,
  3207.57142857143,
  3041.14285714286,
  3100.57142857143,
  3109]

positivePct = [
  0.075058262595191,
  0.078371133255691,
  0.07763711640719,
  0.078033266213817,
  0.072667180243364,
  0.069558534007688,
  0.066261047396713,
  0.070239533478359,
  0.063832841720817,
  0.053226180795053,
  0.052342408684587,
  0.050364561237921,
  0.045827038261369,
  0.043890770936366,
  0.048753830929504,
  0.043473360799738,
  0.04523384689766,
  0.042260660348973,
  0.042605477745478,
  0.045244246243456,
  0.048818589323062,
  0.054477988267117,
  0.059510090504034,
  0.064230397197707,
  0.071416689379013,
  0.086252221692125,
  0.098352616724787,
  0.102023258429865,
  0.095094291821347,
  0.09838224201969,
  0.113891846998587,
  0.112398848435454,
  0.119716020533573,
  0.122004420083525,
  0.109005407952565,
  0.115942984911142,
  0.13744228943736,
  0.140626733008362,
  0.120027075084334,
  0.101049869597954,
  0.089938024597088,
  0.084672994767627,
  0.093430687597047,
  0.08602171562221,
  0.058138270120509]

hospitalized = [
  39544.7142857143,
  48420.1428571429,
  55895.4285714286,
  58619.5714285714,
  58986.5714285714,
  56262,
  53268.8571428572,
  49753.2857142857,
  45972.7142857143,
  42493.1428571429,
  39025.5714285714,
  36554.4285714286,
  34406.7142857143,
  32362.7142857143,
  30686.8571428571,
  29486.4285714286,
  29441.8571428571,
  30077,
  30913.4285714286,
  33660,
  35989,
  37966.7142857143,
  41227.2857142857,
  44872.5714285714,
  49394.8571428571,
  56296.1428571429,
  65952.2857142857,
  75937.8571428571,
  84822.8571428571,
  91321,
  98758.2857142857,
  103521,
  109033.857142857,
  113290.714285714,
  117040.285714286,
  120900.571428571,
  125387.142857143,
  130352.142857143,
  130329,
  126394.142857143,
  118688.285714286,
  107982.857142857,
  96533.7142857143,
  86441.1428571429,
  76094.8571428571]

if len(sys.argv) > 1:
  idx = int(sys.argv[1])
  numTests = numTests[-idx:]
  numDeaths = numDeaths[-idx:]
  positivePct = positivePct[-idx:]
  hospitalized = hospitalized[-idx:]

params, paramRanges, scales = computeInitialParams(numTests, positivePct, hospitalized, numDeaths, POPULATION, STEP_DAYS)

runOptimization(params, paramRanges, scales, (numTests, positivePct, numDeaths, hospitalized), w = (2.0, 2.0, 1.0, 1.0, 1.0, 0.0))
