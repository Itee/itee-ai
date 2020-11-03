/**
 * @author [Tristan Valcke]{@link https://github.com/Itee}
 * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
 *
 * @file Todo
 *
 * @example Todo
 *
 */

import { ActivationFunctionFactory } from '../factories/ActivationFunctionFactory'
import { Neuron } from '../brains/Neuron'
import { NeuralLayer } from '../brains/NeuralLayer'
import { NeuralNetwork } from '../brains/NeuralNetwork'
import { ArtificialIntelligence } from '../brains/ArtificialIntelligence'

//////////////////////////////////////////////////////

export function getRandom ( min, max ) {

    return Math.random() * (max - min) + min

}

export function makeNeuron ( numberOfWeights, activationFunctionType ) {

    const weights = []
    for ( let weightIndex = 0 ; weightIndex < numberOfWeights ; weightIndex++ ) {
        weights.push( getRandom( -1.0, 1.0 ) )
    }

    const bias = getRandom( -1.0, 1.0 )

    const activationFunction = ActivationFunctionFactory.get( activationFunctionType )

    return new Neuron( weights, bias, activationFunction.base, activationFunction.derivated )

}

export function makeLayer ( layerTopology, numberOfNeuronsOnPreviousLayer ) {

    let neurons = []
    for ( let neuronIndex = 0, numberOfNeurons = layerTopology.numberOfNeurons ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
        neurons.push( makeNeuron( numberOfNeuronsOnPreviousLayer, 'sigmoid' ) )
    }

    return new NeuralLayer( layerTopology.name, neurons )

}

export function makeNeuralNetwork ( neuralNetworkTopology ) {

    const layersTopology = neuralNetworkTopology.layers

    let layers = []
    for ( let layerIndex = 0, numberOfLayers = layersTopology.length ; layerIndex < numberOfLayers ; layerIndex++ ) {
        let layerTopology = layersTopology[ layerIndex ]
        layers.push( makeLayer( layerTopology, (layerIndex === 0) ? layerTopology.numberOfNeurons : layersTopology[ layerIndex - 1 ].numberOfNeurons ) )
    }

    return new NeuralNetwork( neuralNetworkTopology.name, layers )

}

export function makeArtificialIntelligence ( topology ) {

    const neuralNetworkTopology = topology.networks

    let networks = []
    for ( let networkIndex = 0, numberOfNetworks = neuralNetworkTopology.length ; networkIndex < numberOfNetworks ; networkIndex++ ) {
        networks.push( makeNeuralNetwork( neuralNetworkTopology[ networkIndex ] ) )
    }

    return new ArtificialIntelligence( topology.name, networks )

}
