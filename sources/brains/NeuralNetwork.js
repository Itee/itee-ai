/**
 * @author [Tristan Valcke]{@link https://github.com/Itee}
 * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
 *
 * @file Todo
 *
 * @example Todo
 *
 */

import { NeuralLayer } from './NeuralLayer'

let neuralNetworkInstanceCounter = 0

class NeuralNetwork {

    constructor ( name, neuralLayers ) {

        this.name         = name || `NeuralNetwork_${neuralNetworkInstanceCounter++}`
        this.neuralLayers = neuralLayers || []

    }

    evaluate ( values ) {

        const neuralLayers   = this.neuralLayers
        const numberOfLayers = neuralLayers.length

        let previousLayer = undefined

        for ( let layerIndex = 0 ; layerIndex < numberOfLayers ; layerIndex++ ) {

            previousLayer = ( layerIndex === 0 ) ? values : neuralLayers[ layerIndex - 1 ].activations
            neuralLayers[ layerIndex ].evaluate( previousLayer )

        }

    }

    learn ( values, expectedResults ) {

        const neuralLayers   = this.neuralLayers
        const numberOfLayers = neuralLayers.length

        let previousLayer = undefined
        let currentLayer  = undefined
        let nextLayer     = undefined

        for ( let layerIndex = numberOfLayers - 1 ; layerIndex >= 0 ; layerIndex-- ) {

            previousLayer = ( layerIndex === 0 ) ? values : neuralLayers[ layerIndex - 1 ]
            currentLayer  = neuralLayers[ layerIndex ]
            nextLayer     = ( layerIndex === numberOfLayers - 1 ) ? expectedResults : neuralLayers[ layerIndex + 1 ]

            currentLayer.learn( previousLayer, nextLayer )

        }

    }

    fromJSON ( jsonData ) {

        const numberOfNeuralLayers = jsonData.length

        this.neuralLayers = []
        for ( let neuralLayerIndex = 0 ; neuralLayerIndex < numberOfNeuralLayers ; neuralLayerIndex++ ) {

            const neuralLayer = new NeuralLayer()
            neuralLayer.fromJSON( jsonData[ neuralLayerIndex ] )
            this.neuralLayers.push( neuralLayer )

        }

    }

    toJSON () {

        const neuralLayers    = this.neuralLayers
        const numberOfNeurons = neuralLayers.length

        let neuralLayersArray = []
        for ( let neuralLayerIndex = 0 ; neuralLayerIndex < numberOfNeurons ; neuralLayerIndex++ ) {
            neuralLayersArray.push( neuralLayers[ neuralLayerIndex ].toJSON() )
        }

        return neuralLayersArray

    }

}

export { NeuralNetwork }
