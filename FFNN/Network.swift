//
//  Network.swift
//  FFNN
//
//  Created by Елена Яновская on 28.11.2017.
//  Copyright © 2017 Елена Яновская. All rights reserved.
//

import Accelerate

class Network {
    var layers: [Layer]
    var firstL: Int
    var lastL: Int
    var numberOfLayers: Int
    init(layerStructure:[Int],
        activationFunction: @escaping (Double) -> Double = sigmoidFunction,
        derivativeActivationFunction: @escaping (Double) -> Double = derivativeSigmoidFunction,
        learningRate: Double) {
        layers = [Layer]()
        self.firstL = layerStructure[0]
        self.numberOfLayers = layerStructure.count
        self.lastL = layerStructure.last!
        
        //Create input layer
        layers.append (Layer(numberOfNeurons: layerStructure[0],
            activationFunction: activationFunction,
            derivativeActivationFunction: derivativeActivationFunction,
            learningRate: learningRate))
        
        //Create hidden layers and output layer
        for layer in layerStructure.enumerated() where layer.offset != 0 {
            layers.append (Layer(previousLayer: layers[layer.offset - 1],
                numberOfNeurons: layer.element,
                activationFunction: activationFunction,
                derivativeActivationFunction: derivativeActivationFunction,
                learningRate: learningRate))
        }
    }
    
    //Forward propagation prediction
    func outputs(input: [Double]) -> [Double] {
        return layers.reduce(input) { $1.outputSinapses(inputs: $0) }
    }
    
    //Backward propagation training, calc deltas
    func backwardPropagationMethod(expected: [Double]) {
        layers.last!.calculateDeltasForOutputLayer(expected: expected)
        for l in (1..<layers.count - 1).reversed() {
            layers[l].calculateDeltasForHiddenLayer(nextLayer: layers[l + 1])
        }
    }
    
    //Apply new weights for neurons after each learning epoch
    func updateWeightsAfterLearn() {
        for layer in layers {
            for neuron in layer.neurons {
                for w in 0..<neuron.weights.count {
                    neuron.weights[w] = neuron.weights[w] + (neuron.learningRate * (layer.previousLayer?.layerOutputCache[w])!  * neuron.delta)
                }
            }
        }
    }
    
    //Training network epoch
    func train(inputs:[[Double]], expecteds:[[Double]]) {
        for (position, input) in inputs.enumerated() {
            let expectedOutputs = expecteds[position]
            let currentOutputs = outputs(input: input)
            let diffrencesBetweenPredictionAndExpected = zip(currentOutputs, expectedOutputs).map{$0-$1}
            let meanSquaredError = sqrt(diffrencesBetweenPredictionAndExpected.map{$0*$0}.reduce(0,+))
            print("Training loss: \(meanSquaredError)")
            
            backwardPropagationMethod(expected: expectedOutputs)
            updateWeightsAfterLearn()
        }
    }
    
    //Validation results
    func validate(input:[Double], expected:Double) -> (result: Double, expected:Double) {
        let result = outputs(input: input)[0]
        return (result,expected) 
    }
}
