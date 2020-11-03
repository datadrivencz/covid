#!/usr/bin/env python3

import scipy.optimize as optimize

params = [
  600,   # initial infected
  19,    # num of contacts
  0.115, # probability of being infected
  2.5,   # contacts in quaranteen
  0.7,   # tracing prob
  30000, # tracing capacity
  0.5,   # patients with symptomps
  0.3,   # prob of tested when having symptoms
  0.04,  # prob of being hospitalized
  0.01]  # death rate 

coreParams = len(params)
probParams = [2, 4, 6, 7, 8, 9]

numTests = [680, 1148, 1771, 2061, 2160, 2473, 3786, 5675, 8111, 11173, 12586, 11789]
numDeaths = [2, 3, 7, 10, 13, 18, 28, 44, 62, 98, 134, 184]

for i in range (0, len(numTests)):
  #params.append(0.5)
  params.append(max(1 / ((i + 1.) ** 2), 0.5))

#params = [ 5.71265109e+02,  1.94990837e+01,  1.25296389e-01,  3.25013977e+00,
#        6.27645901e-01,  2.24629204e+04,  7.03269545e-01,  5.14049128e-01,
#        1.90536847e-02,  7.84025668e+00,  6.39755737e-01,  5.91899772e-01,
#        5.42348912e-01,  4.08684606e-01,  3.70925672e-01,  4.51165928e-01,
#        5.46888629e-01,  3.58402956e-01,  2.91379327e-01,  2.60871255e-01,
#        2.01787364e-01, -1.30938102e+02]

if len(numTests) != len(numDeaths):
  raise Exception("Tests and Deaths have to match in length")

def loss(params):
  infected = params[0]
  params = norm(params)
  loss = 0.0
  stepInfected = []
  for i in range (0, len(numTests)):
    step = evalStep(infected, params[1:coreParams], params[coreParams + i])
    infected = step[-1]
    loss += (step[6] - numTests[i]) ** 2
    stepInfected.append(step[1])
    if i > 3:
      deaths = stepInfected[-4] * params[9]
      loss += (400 * (deaths - numDeaths[i])) ** 2
  #endfor
  return loss
#enddef

def norm(params):
  return params
  def normParam(idx, val):
    if val < 0:
      return 0.0
    if val > 1.0 and idx in probParams:
      return 1.0
    return val
  #enddef
  newParams = []
  for idx, val in enumerate(params):
    newParams.append(normParam(idx, val))
  return newParams
#enddef

def evalStep(infected, params, contactFactor):
  numOfContacts, probInfected, quaranteenContacts, tracingProb, tracingCapacity, patientsWithSymptoms, probOfBeingTested, hospRate, deathRate = params[0:coreParams]
  contacts = infected * contactFactor * numOfContacts
  totalInfected = contacts * probInfected
  totalTraced = min(tracingCapacity, contacts)
  tracedPositive = totalTraced * tracingProb * probInfected
  positiveOutQuaranteen = totalInfected - tracedPositive
  totalTests = totalTraced + positiveOutQuaranteen * patientsWithSymptoms * probOfBeingTested
  positiveTests = tracedPositive + positiveOutQuaranteen * patientsWithSymptoms * probOfBeingTested
  infected = tracedPositive * quaranteenContacts * probInfected + positiveOutQuaranteen * numOfContacts * probInfected 
  return (contacts, totalInfected, totalTraced, tracedPositive, positiveOutQuaranteen, totalTests, positiveTests, infected)
#enddef


res = (optimize.minimize(loss, params, method='Nelder-Mead', options={"maxfev": 1000000}))
for item in res.x:
  print("%02f" % item)

init = res.x[0]
deaths = []
for i in range (0, len(numTests)):
  step = evalStep(init, res.x[1:coreParams], res.x[coreParams + i])
  deaths.append(step[1] * res.x[9])
  print (step, None if i < 4 else deaths[-4], numDeaths[i], step[-2], numTests[i])
  init = step[-1]
