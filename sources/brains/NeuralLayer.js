/**
 * @author [Tristan Valcke]{@link https://github.com/Itee}
 * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
 *
 * @file Todo
 *
 * @example Todo
 *
 */

import { Neuron } from './Neuron'

let neuralLayerInstanceCounter = 0

class NeuralLayer {

    constructor ( name, neurons ) {

        this.name        = name || `NeuralLayer_${neuralLayerInstanceCounter++}`
        this.neurons     = neurons || []

        this.activations = undefined
        this.gradients   = undefined

    }

    evaluate ( previousLayerResult ) {

        const neurons         = this.neurons
        const numberOfNeurons = neurons.length

        let currentLayerResult = new Array( numberOfNeurons )
        for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
            currentLayerResult[ neuronIndex ] = neurons[ neuronIndex ].evaluate( previousLayerResult )
        }

        this.activations = currentLayerResult

    }

    learn ( previousLayer, nextLayer ) {

        const neurons         = this.neurons
        const numberOfNeurons = neurons.length

        let currentLocalsGradients = new Array( numberOfNeurons )
        for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
            currentLocalsGradients[ neuronIndex ] = neurons[ neuronIndex ].learn( previousLayer, nextLayer, neuronIndex )
        }

        this.gradients = currentLocalsGradients

    }

    fromJSON ( jsonData ) {

        const numberOfNeurons = jsonData.length

        this.neurons = []
        for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {

            const neuron = new Neuron()
            neuron.fromJSON( jsonData[ neuronIndex ] )
            this.neurons.push( neuron )

        }

    }

    toJSON () {

        const neurons         = this.neurons
        const numberOfNeurons = neurons.length

        let neuronsArray = []
        for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
            neuronsArray.push( neurons[ neuronIndex ].toJSON() )
        }

        return neuronsArray

    }

}

export { NeuralLayer }
