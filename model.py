#!/usr/bin/env python3

import math
import scipy.optimize as optimize

STEP_DAYS = 5
POPULATION = 10700000

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

coreParams = len(params)

numTests = [
  224,
  233.142857142857,
  246.142857142857,
  281.857142857143,
  329.857142857143,
  450.571428571429,
  679.142857142857,
  1147.57142857143,
  1770.85714285714,
  2061.85714285714,
  2159.71428571429,
  2474.71428571429,
  3785.71428571429,
  5674.28571428571,
  8110.85714285714,
  11173.2857142857,
  12585,
  11788.2857142857,
  10872.7142857143,
  7820.14285714286,
  7088.75]

numDeaths = [
  1,
  0.857142857142857,
  2,
  1.85714285714286,
  1.71428571428571,
  1.42857142857143,
  3.14285714285714,
  4.14285714285714,
  7.57142857142857,
  11.4285714285714,
  13.5714285714286,
  18,
  29.2857142857143,
  50.2857142857143,
  69,
  104.857142857143,
  147.142857142857,
  193.857142857143,
  210,
  182.285714285714,
  125.6]

positivePct = [
  0.033385714285714,
  0.035642857142857,
  0.038585714285714,
  0.042571428571429,
  0.044228571428572,
  0.0498,
  0.064342857142857,
  0.083528571428572,
  0.100857142857143,
  0.107085714285714,
  0.117342857142857,
  0.135828571428571,
  0.196171428571429,
  0.2477,
  0.270071428571429,
  0.302742857142857,
  0.320414285714286,
  0.309685714285714,
  0.302928571428571,
  0.264214285714286,
  0.2579]

hospitalized = [
  109.142857142857,
  107.428571428571,
  108,
  117,
  139.142857142857,
  168.285714285714,
  208.571428571429,
  271.142857142857,
  391.428571428571,
  559,
  756.285714285714,
  1023.28571428571,
  1458,
  2164.57142857143,
  3125.14285714286,
  4408,
  5819.28571428572,
  7204.85714285714,
  7962,
  7799.85714285714,
  7300.25]

paramRanges = [
  [0, 500],  # initial infected
  [10, 30],
  [0.05, 0.3],
  [0.05, 10],
  [0.01, 0.8],
  [5000,80000],
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
    loss += min(100000, step[6] - numTests[i]) ** 2
    positiveTestsPct = (step[-2] + 0.0001) / (step[-3] + 0.0001)
    loss += 500000 * (positiveTestsPct / positivePct[i] - 1.0) ** 2
    if i > 1:
      hosp = stepInfected[-2] * params[8]
      loss += 400 * min(100000, hosp - hospitalized[i]) ** 2
    if i > 3:
      deaths = stepInfected[-4] * params[9]
      deathLoss = 100 * min(100000, deaths - numDeaths[i]) ** 2
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
