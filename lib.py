#!/usr/bin/env python3

import sys
import math
import scipy.optimize as optimize


def computeInitialParams(numTests, positivePct, hospitalized, numDeaths, population, stepDays):

  global POPULATION
  global STEP_DAYS

  POPULATION = population
  STEP_DAYS = stepDays

  if len(numTests) != len(numDeaths) or len(numDeaths) != len(positivePct):
    raise Exception("Tests and Deaths have to match in length")

  scales = [max(numTests), max(positivePct), max(hospitalized), max(numDeaths)]

  params = [
    50,   # initial infected
    20,    # num of contacts
    0.1,   # probability of being infected
    3,     # contacts in quaranteen
    0.4,   # tracing prob
    20000, # tracing capacity
    0.33,   # patients with symptomps
    0.3,   # prob of tested when having symptoms
    0.04,  # prob of being hospitalized
    0.01]  # death rate 
  
  paramRanges = [
    [0, 100 * numTests[0]],  # initial infected
    [10, 50],
    [0.01, 0.9],
    [0.05, 5],
    [0.01, 0.8],
    [0,200000],
    [0.05, 0.95],
    [0.1, 0.6],
    [0.0, 1.0],
    [0.001, 0.1]]
  
  for i in range (0, len(numTests)):
    # contactFactor
    params.append(1.0)
    paramRanges.append([0.0, 2.0])


  print(scales)
  return params, paramRanges, scales
#enddef


def diff(a, b, scale):
  #return ((max(a, b) + 0.00001) / (min(a, b) + 0.00001) - 1.0) ** 2
  return ((a - b) / scale) ** 2


def loss(params, paramRanges, scales, data, debug=False):
  numTests, positivePct, numDeaths, hospitalized = data
  coreParams = len(params) - len(numTests)
  loss = 0.0
  stepInfected = [params[0]]
  stepTotalInfected = []
  for i in range (0, len(numTests)):
    step = evalStep(stepInfected, params[1:coreParams], params[coreParams + i], i, debug=debug)
    stepTotalInfected.append(step[1])
    #loss += (step[6] - numTests[i]) ** 2
    loss += OPT_WEIGHTS[0] * diff(step[6], numTests[i], scales[0])
    positiveTestsPct = (step[-2] + 0.0001) / (step[-3] + 0.0001)
    #loss += 10000 ** 2 * (positiveTestsPct / positivePct[i] - 1.0) ** 2
    loss += OPT_WEIGHTS[1] * diff(positiveTestsPct, positivePct[i], scales[1])
    if i > 3:
      hosp = sum(stepTotalInfected[-5:-2]) * params[8]
      #loss += 10 ** 2 * (hosp - hospitalized[i]) ** 2
      loss += OPT_WEIGHTS[2] * diff(hosp, hospitalized[i], scales[2])
    # deaths last few steps are unreliable
    if i > 3:
      deaths = stepTotalInfected[-5] * params[9]
      #deathLoss = 100 ** 2 * (deaths - numDeaths[i]) ** 2
      deathLoss = diff(deaths, numDeaths[i], scales[3])
      #loss += deathLoss if deaths >= numDeaths[i] else 5 * deathLoss
      loss += OPT_WEIGHTS[3] * deathLoss
    stepInfected.append(step[-1])
  #endfor
  return loss * score(params, paramRanges, coreParams)
#enddef

def score(params, paramRanges, coreParams):
  def maxRatio(params):
    return 1.0
    previous = 1.0
    maxRatio = 0.0
    for i in range (coreParams, coreParams + len(numTests)):
      maxRatio = max(maxRatio, abs(params[i] / previous))
      previous = params[i]
    return maxRatio
  def rangePenalty(params):
    totalPenalty = 0.0
    for (idx, value) in enumerate(params):
      r = paramRanges[idx]
      totalPenalty += r[0] - value if value < r[0] else value - r[1] if value > r[1] else 0.0
    return totalPenalty
  return max(1.0, maxRatio(params)) + 10 * rangePenalty(params)

def evalStep(infectedAll, params, contactFactor, r, debug=False):
  numOfContacts, probInfected, quaranteenContacts, tracingProb, tracingCapacity, patientsWithSymptoms, probOfBeingTested, hospRate, deathRate = params
  infected = infectedAll[-1]
  contacts = infected * contactFactor * numOfContacts
  if debug:
    print("contacts: %f, infected: %f, factor: %f, numOfContacts: %f" % (contacts, infected, contactFactor, numOfContacts))
  effectiveProbInfected = probInfected * (1.0 - min(POPULATION, sum(infectedAll[-19:]) * STEP_DAYS) / POPULATION)
  totalInfected = contacts * effectiveProbInfected
  totalTraced = min(tracingCapacity, contacts)
  tracedPositive = totalTraced * tracingProb * effectiveProbInfected
  positiveTests = tracedPositive + (totalInfected - tracedPositive) * patientsWithSymptoms * probOfBeingTested
  positiveOutQuaranteen = totalInfected - positiveTests
  totalTests = totalTraced + positiveOutQuaranteen * patientsWithSymptoms * probOfBeingTested
  infected = (positiveTests * quaranteenContacts + positiveOutQuaranteen * numOfContacts) * contactFactor * effectiveProbInfected
  return (contacts, totalInfected, totalTraced, tracedPositive, positiveOutQuaranteen, totalTests, positiveTests, infected)
#enddef

def runOptimization(params, paramRanges, scales, data, w = (1, 1, 1, 1)):

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
  
  for item in res.x:
    print(item)
 
  numTests, positivePct, numDeaths, hospitalized = data 
  coreParams = len(params) - len(numTests)
  infected = [res.x[0]]
  totalInfected = []
  deaths = []
  hosp = []
  for i in range (0, len(numTests)):
    step = evalStep(infected, res.x[1:coreParams], res.x[coreParams + i], i, debug=True)
    totalInfected.append(step[1])
    deaths.append(step[1] * res.x[9])
    hosp.append(0 if i < 4 else sum(totalInfected[-5:-2]) * res.x[8])
    print (step, None if i < 4 else deaths[-5], numDeaths[i], None if i < 2 else hosp[-2], hospitalized[i], step[-2], numTests[i], step[-2] / step[-3], positivePct[i], score(res.x, paramRanges, coreParams))
    infected.append(step[-1])
