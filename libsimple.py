#!/usr/bin/env python3

import sys
import math
import scipy.optimize as optimize


def computeInitialParams(numTests, positivePct, hospitalized, numDeaths, population, stepDays):

  global POPULATION
  global STEP_DAYS

  POPULATION = population
  STEP_DAYS = stepDays

  inp = (numTests, positivePct, hospitalized, numDeaths)
  inpl = [map(lambda x: len(x), inp)]
  for l in inpl:
    if inpl[0] != l:
      raise Exception("All input parameters must have equal length, got %d" % (l, ))

  #scales = [max(numTests), max(positivePct), max(hospitalized), max(numDeaths)]
  scales = [math.sqrt(i) for i in (max(numTests), max(positivePct), max(hospitalized), max(numDeaths))]

  params = [
    numTests[0] * 10,   # initial contacts
    20,    # num of contacts
    0.1,   # probability of being infected
    3,     # contacts in quaranteen
    0.4,   # tracing prob
    POPULATION * 0.001, # tracing capacity
    0.33,   # tested of untraced infected
    0.001, # prob of being randomly tested
    0.04,  # prob of being hospitalized
    0.005]  # death rate 
  
  paramRanges = [
    [numTests[0] * 5, 100 * numTests[0]],  # initial infected
    [10, 50],
    [0.01, 0.9],
    [0.05, 10],
    [0.01, 0.8],
    [0,POPULATION / 100],
    [0.05, 0.4],
    [0.0, 0.1],
    [0.0, 1.0],
    [0.005, 0.02]]
  
  for i in range (0, len(numTests)):
    # contactFactor
    params.append(1.0)
    paramRanges.append([0.0, 1.5])


  return params, paramRanges, scales
#enddef


def diff(a, b, scale):
  #return ((max(a, b) + 0.00001) / (min(a, b) + 0.00001) - 1.0) ** 2
  return min(1000000, ((a - b) / scale) ** 2)


def loss(params, paramRanges, scales, data, debug=False):
  numTests, positivePct, numDeaths, hospitalized = data
  coreParams = len(params) - len(numTests)
  loss = 0.0
  stepContacts = params[0]
  steps = []
  for i in range (0, len(numTests)):
    step = evalStep(stepContacts, steps, params[1:coreParams], params[coreParams + i], i, debug=debug)
    steps.append(step)
    stepContacts = step[-1]
    loss += math.sqrt(i+1) * OPT_WEIGHTS[0] * diff(step[6], numTests[i], scales[0])
    positiveTestsPct = (step[-2] + 0.0001) / (step[-3] + 0.0001)
    loss += math.sqrt(i+1) * OPT_WEIGHTS[1] * diff(positiveTestsPct, positivePct[i], scales[1])
    if i > 3:
      hosp = sum(map(lambda x: x[1], steps[-5:-2])) * params[8]
      loss += math.sqrt(i+1) * OPT_WEIGHTS[2] * diff(hosp, hospitalized[i], scales[2])
    # deaths last few steps are unreliable
    if i > 3:
      deaths = steps[-5][1] * params[9]
      deathLoss = diff(deaths, numDeaths[i], scales[3])
      loss += math.sqrt(i+1) * OPT_WEIGHTS[3] * deathLoss if deaths >= numDeaths[i] else OPT_WEIGHTS[4] * deathLoss
  #endfor
  return loss * score(params, paramRanges, coreParams)
#enddef

def score(params, paramRanges, coreParams):
  def rangePenalty(params):
    totalPenalty = 0.0
    for (idx, value) in enumerate(params):
      r = paramRanges[idx]
      totalPenalty += r[0] - value if value < r[0] else value - r[1] if value > r[1] else 0.0
    return totalPenalty
  return 1.0 + 10 * rangePenalty(params)

def evalStep(contacts, lastSteps, params, contactFactor, r, debug=False):
  numOfContacts, probInfected, quaranteenContacts, tracingProb, tracingCapacity, probOfBeingTested, probTestedRandomly, hospRate, deathRate = params
  withImmunity = STEP_DAYS * sum(map(lambda x: x[1], lastSteps[-39:]))
  infectiousness = (1 - min(0.7, withImmunity / POPULATION)) * probInfected
  totalInfected = contacts * infectiousness
  tracedTotal = min(tracingCapacity, contacts * tracingProb)
  tracedPositive = tracedTotal * infectiousness
  untracedPositive = totalInfected - tracedPositive
  testAll = tracedTotal + untracedPositive * probOfBeingTested + probTestedRandomly * (POPULATION - withImmunity)
  testPositive = tracedPositive + untracedPositive * probOfBeingTested
  newContacts = (tracedPositive * quaranteenContacts + untracedPositive * numOfContacts) * contactFactor
  return (infectiousness, totalInfected, tracedTotal, tracedPositive, untracedPositive, testAll, testPositive, newContacts)
#enddef

def printResult(res, data, paramRanges):
  for item in res:
    print(item)
 
  numTests, positivePct, numDeaths, hospitalized = data 
  coreParams = len(res) - len(numTests)
  contacts = res[0]
  steps = []
  deaths = []
  hosp = []
  for i in range (0, len(numTests)):
    step = evalStep(contacts, steps, res[1:coreParams], res[coreParams + i], i, debug=True)
    steps.append(step)
    contacts = step[-1]
    deaths.append(step[1] * res[9])
    hosp.append(0 if i < 4 else sum(map(lambda x: x[1], steps[-5:-2])) * res[8])
    print (step, None if i < 4 else deaths[-5], numDeaths[i], None if i < 2 else hosp[-2], hospitalized[i], step[-2], numTests[i], step[-2] / step[-3], positivePct[i], score(res, paramRanges, coreParams))

def runOptimization(params, paramRanges, scales, data, w = (1, 1, 1, 1, 1)):

  global OPT_WEIGHTS
  OPT_WEIGHTS = w

  print(paramRanges)

  startingLoss = loss(params, paramRanges, scales, data)
  print("loss: %f" % (startingLoss, ))
 
  def opt(optParams):
    return loss(optParams, paramRanges, scales, data)
 
  r = 0
  while True:
    print("%d-CG" % (r, ))
    res = (optimize.minimize(opt, params, method='CG', options={"maxfev": 1000000}))
    params = res.x
    print("loss: %f" % (loss(params, paramRanges, scales, data)))
    print("%d-Nelder-Mead" % (r, ))
    res = (optimize.minimize(opt, params, method='Nelder-Mead', options={"maxfev": 1000000}))
    params = res.x
    endingLoss = loss(params, paramRanges, scales, data)
    print("loss: %f" % (endingLoss, ))
    if (startingLoss - endingLoss) / startingLoss < 0.005:
      break
    startingLoss = endingLoss
    r += 1

  printResult(res.x, data, paramRanges)

