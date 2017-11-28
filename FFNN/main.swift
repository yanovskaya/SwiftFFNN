//
//  main.swift
//  FFNN
//
//  Created by Елена Яновская on 28.11.2017.
//  Copyright © 2017 Елена Яновская. All rights reserved.
//

import Accelerate

//Neuron activation function
func sigmoidFunction(x: Double) -> Double {
    return 1.0 / (1.0 + exp(-x))
}
//For calc back propagation delta weights
func derivativeSigmoidFunction(yourMom: Double) -> Double {
    return sigmoidFunction(x: yourMom) * (1 - sigmoidFunction(x: yourMom))
}
//Initial neuron weights
func randomWeights(number: Int) -> [Double] {
    return (0..<number).map{ _ in Double(arc4random()) / Double(UInt32.max)}
}

var network: Network = Network(layerStructure: [3,2,1], learningRate: 0.4)

let trainEpochs = 1000

let trainingPatterns = [[0.0,0.0,0.0],
                        [0.0,0.0,1.0],
                        [0.0,1.0,0.0],
                        [0.0,1.0,1.0],
                        [1.0,0.0,0.0],
                        [1.0,0.0,1.0],
                        [1.0,1.0,0.0]]

let expectedResults = [[0.0],
                       [1.0],
                       [0.0],
                       [0.0],
                       [1.0],
                       [1.0],
                       [0.0]]

//Condition for program termination
if trainingPatterns.count == expectedResults.count && network.lastL == 1 && network.numberOfLayers >= 2 {
    for (trainingPattern, expectedResult) in zip(trainingPatterns, expectedResults) {
        if trainingPattern.count == network.firstL && expectedResult.count == 1 {
        } else {
            print("Error")
            exit(0)
        }
    }
}
else {
    print("Error")
    exit(0)
}

for _ in 0..<trainEpochs {
    network.train(inputs: trainingPatterns, expecteds: expectedResults)
}



print ("\nWeights of hidden layer")
for _ in 1..<network.layers.count-1{
    for neuron in network.layers[1].neurons {
        print("\(neuron.weights)")
    }
}

print ("\nWeights of output layer")
for neuron in network.layers[network.layers.count-1].neurons {
    print("\(neuron.weights)\n")
}

for i in 0..<trainingPatterns.count {
    let results = network.validate(input: trainingPatterns[i], expected: expectedResults[i][0])
    print("For input:\(trainingPatterns[i]) the prediction is:\(results.result), expected:\(results.expected)")
}

let results = network.validate(input: [1.0, 1.0, 1.0], expected: 1)

print("For input:\([1.0, 1.0, 1.0]) the prediction is:\(results.result)")
