#!/usr/bin/env python3

import math
import scipy.optimize as optimize

STEP_DAYS = 5
POPULATION = 330000000

params = [
  50000,   # initial infected
  20,    # num of contacts
  0.1,   # probability of being infected
  3,     # contacts in quaranteen
  0.4,   # tracing prob
  500000, # tracing capacity
  0.33,   # patients with symptomps
  0.3,   # prob of tested when having symptoms
  0.04,  # prob of being hospitalized
  0.01]  # death rate 

coreParams = len(params)

numTests = [
  55594.4285714286,
  63805,
  65905.2857142857,
  65777.8571428571,
  62732.5714285714,
  56689,
  51968,
  52560.1428571429,
  45980.2857142857,
  41215.2857142857,
  41122.8571428571,
  40493.1428571429,
  36476.8571428571,
  34989.4285714286,
  39734.4285714286,
  42556.8571428572,
  41701.8571428571,
  42931.7142857143,
  46852.4285714286,
  51850.4285714286,
  56922.4285714286,
  66342.4285714286,
  75975.5714285714,
  85173,
  107547.142857143,
  136440.571428571]

numDeaths = [
  665.714285714286,
  707.857142857143,
  793.857142857143,
  912.428571428571,
  1118.14285714286,
  1054.14285714286,
  1045.14285714286,
  1069.57142857143,
  1030.42857142857,
  966.857142857143,
  918.714285714286,
  854.714285714286,
  728.857142857143,
  747.714285714286,
  789.857142857143,
  741.285714285714,
  728,
  692.428571428571,
  694.142857142857,
  685.428571428571,
  708.285714285714,
  807.857142857143,
  796.857142857143,
  909.428571428571,
  1003.14285714286,
  1069.71428571429]

positivePct = [
0.075504126887962,
0.077237891157903,
0.077948753091087,
0.075082645333765,
0.07031991623985,
0.068122610735097,
0.065063873218275,
0.066148983116431,
0.055890679392553,
0.052442697939808,
0.050887706632186,
0.047371994028899,
0.044440647403424,
0.047466881439692,
0.047011238537684,
0.043711077687601,
0.040163329390916,
0.041190828478089,
0.044468111176949,
0.046380926443752,
0.050495990599673,
0.058088449107242,
0.062147577127549,
0.068445093501436,
0.079735260243606,
0.095806737397725]

hospitalized = [
  44394.1428571429,
  54188.7142857143,
  57801,
  59236,
  57609.8571428572,
  54366.1428571429,
  51209,
  47401.7142857143,
  43882.5714285714,
  40376,
  37530,
  35303.8571428571,
  33068.2857142857,
  31289.4285714286,
  29821,
  29329.4285714286,
  29826.5714285714,
  30368.7142857143,
  32366.7142857143,
  35205.7142857143,
  37052.1428571429,
  39737.2857142857,
  43347.4285714286,
  47361.8571428572,
  53092,
  62123.2857142857]

paramRanges = [
  [0, 1000000],  # initial infected
  [10, 50],
  [0.05, 0.3],
  [0.05, 10],
  [0.01, 0.8],
  [10000,5000000],
  [0.05, 0.95],
  [0.1, 0.6],
  [0.0, 1.0],
  [0.001, 0.1]]

for i in range (0, len(numTests)):
  # contactFactor
  params.append(1.0)
  paramRanges.append([0.0, 1.1])


if len(numTests) != len(numDeaths) or len(numDeaths) != len(positivePct):
  raise Exception("Tests and Deaths have to match in length")

def loss(params):
  loss = 0.0
  stepInfected = [params[0]]
  for i in range (0, len(numTests)):
    step = evalStep(stepInfected, params[1:coreParams], params[coreParams + i])
    loss += min(10000000, step[6] - numTests[i]) ** 2
    positiveTestsPct = (step[-2] + 0.0001) / (step[-3] + 0.0001)
    loss += 100000 * (positiveTestsPct / positivePct[i] - 1.0) ** 2
    if i > 1:
      hosp = stepInfected[-2] * params[8]
      loss += 40 * min(10000000, hosp - hospitalized[i]) ** 2
    if i > 3:
      deaths = stepInfected[-4] * params[9]
      deathLoss = 1000 * min(10000000, deaths - numDeaths[i]) ** 2
      #loss += deathLoss if deaths >= numDeaths[i] else 5 * deathLoss
      loss += deathLoss
    stepInfected.append(step[-1])
  #endfor
  return loss * score(params)
#enddef

def score(params):
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

def evalStep(infectedAll, params, contactFactor, debug=False):
  numOfContacts, probInfected, quaranteenContacts, tracingProb, tracingCapacity, patientsWithSymptoms, probOfBeingTested, hospRate, deathRate = params
  infected = infectedAll[-1]
  contacts = infected * contactFactor * numOfContacts
  effectiveProbInfected = probInfected * (1.0 - min(POPULATION, sum(infectedAll[-19:]) * STEP_DAYS) / POPULATION)
  totalInfected = contacts * effectiveProbInfected
  totalTraced = min(tracingCapacity, contacts)
  tracedPositive = totalTraced * tracingProb * effectiveProbInfected
  positiveOutQuaranteen = totalInfected - tracedPositive
  totalTests = totalTraced + positiveOutQuaranteen * patientsWithSymptoms * probOfBeingTested
  positiveTests = tracedPositive + positiveOutQuaranteen * patientsWithSymptoms * probOfBeingTested
  infected = (tracedPositive * quaranteenContacts + positiveOutQuaranteen * numOfContacts) * contactFactor * effectiveProbInfected
  return (contacts, totalInfected, totalTraced, tracedPositive, positiveOutQuaranteen, totalTests, positiveTests, infected)
#enddef


startingLoss = loss(params)
print("loss: %f" % (startingLoss, ))
r = 0
while True:
  print("%d-CG" % (r, ))
  res = (optimize.minimize(loss, params, method='CG', options={"maxfev": 1000000}))
  params = res.x
  print("loss: %f" % (loss(params)))
  print("%d-Nelder-Mead" % (r, ))
  res = (optimize.minimize(loss, params, method='Nelder-Mead', options={"maxfev": 1000000}))
  params = res.x
  endingLoss = loss(params)
  print("loss: %f" % (endingLoss, ))
  if (startingLoss - endingLoss) / startingLoss < 0.005:
    break
  startingLoss = endingLoss
  r += 1

for item in res.x:
  print(item)

infected = [res.x[0]]
deaths = []
hosp = []
for i in range (0, len(numTests)):
  step = evalStep(infected, res.x[1:coreParams], res.x[coreParams + i], debug=True)
  deaths.append(step[1] * res.x[9])
  hosp.append(step[1] * res.x[8])
  print (step, None if i < 4 else deaths[-4], numDeaths[i], None if i < 2 else hosp[-2], hospitalized[i], step[-2], numTests[i], step[-2] / step[-3], positivePct[i], score(res.x))
  infected.append(step[-1])