//
//  Layer.swift
//  FFNN
//
//  Created by Елена Яновская on 28.11.2017.
//  Copyright © 2017 Елена Яновская. All rights reserved.
//


class Layer {
    let previousLayer: Layer?
    var neurons: [Neuron]
    var layerOutputCache: [Double]
    
    init(previousLayer: Layer? = nil,
         numberOfNeurons: Int,
         activationFunction: @escaping (Double) -> Double,
         derivativeActivationFunction: @escaping (Double)-> Double,
         learningRate: Double) {
        
        self.previousLayer = previousLayer
        self.neurons = Array<Neuron>()
        for _ in 0..<numberOfNeurons {
            self.neurons.append (Neuron(weights: randomWeights(number: previousLayer?.neurons.count ?? 0),
                                        activationFunction: activationFunction,
                                        derivativeActivationFunction: derivativeActivationFunction,
                                        learningRate: learningRate))
        }
        self.layerOutputCache = Array<Double>(repeating: 0.0,
                                              count: neurons.count)
    }
    
    //Forward propagation prediction outputs calc
    func outputSinapses(inputs: [Double]) -> [Double] {
        if previousLayer == nil { //Input layer
            layerOutputCache = inputs
        } else { //Hidden and output layers
            layerOutputCache = neurons.map { $0.neuronOutput(inputs: inputs) }
        }
        return layerOutputCache
    }
    
    //Backward propagation deltas calc
    func calculateDeltasForOutputLayer(expected: [Double]) {
        for n in 0..<neurons.count {
            neurons[n].delta = neurons[n].derivativeActivationFunction( neurons[n].inputCache) * (expected[n] - layerOutputCache[n])
        }
    }
    
    //Backward propagation deltas calc
    func calculateDeltasForHiddenLayer(nextLayer: Layer) {
        for (index, neuron) in neurons.enumerated() {
            let nextWeights = nextLayer.neurons.map { $0.weights[index] }
            let nextDeltas = nextLayer.neurons.map { $0.delta }
            let error = zip(nextWeights, nextDeltas).map(*).reduce(0, +)
            neuron.delta = neuron.derivativeActivationFunction(neuron.inputCache) * error
        }
    }
}
