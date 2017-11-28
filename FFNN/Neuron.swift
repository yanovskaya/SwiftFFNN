//
//  Neuron.swift
//  FFNN
//
//  Created by Елена Яновская on 28.11.2017.
//  Copyright © 2017 Елена Яновская. All rights reserved.
//

class Neuron {
    var weights: [Double]
    var activationFunction: (Double) -> Double
    var derivativeActivationFunction: (Double) -> Double
    var delta: Double = 0.0
    var inputCache: Double = 0.0
    var learningRate: Double
    
    init(weights: [Double],
         activationFunction: @escaping (Double) -> Double,
        derivativeActivationFunction: @escaping (Double) -> Double,
        learningRate: Double) {
        self.weights = weights
        self.activationFunction = activationFunction
        self.derivativeActivationFunction = derivativeActivationFunction
        self.learningRate = learningRate
    }
    
    func neuronOutput(inputs: [Double]) -> Double {
        inputCache = zip(inputs, self.weights).map(*).reduce(0, +)
        return activationFunction(inputCache) 
    }
}

