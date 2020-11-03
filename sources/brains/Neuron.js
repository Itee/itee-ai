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

function isObject ( value ) {

    return ( value !== null && typeof value === 'object' && !Array.isArray( value ) )

}

class Neuron {

    constructor ( weights, bias, activationFunction, derivatedActivationFunction ) {

        this.weights                     = weights || [ 1.0 ]
        this.bias                        = bias || 0.0
        this.activationFunction          = activationFunction || undefined
        this.derivatedActivationFunction = derivatedActivationFunction || undefined

        this.weightedSum   = undefined
        this.activation    = undefined
        this.localGradient = undefined

        this.LEARNING_RATE = 1.0
        //        this.LEARNING_RATE = 0.01

    }

    evaluate ( inputs ) {

        const numberOfInputs = inputs.length
        const weights        = this.weights

        // Weights
        this.weightedSum = 0.0
        for ( let index = 0 ; index < numberOfInputs ; index++ ) {
            this.weightedSum += inputs[ index ] * weights[ index ]
        }
        // Bias
        this.weightedSum += -1.0 * this.bias

        this.activation = this.activationFunction( this.weightedSum )

        return this.activation

    }

    learn ( previousLayer, nextLayer, thisIndex ) {

        const previousLayerValues = ( isObject( previousLayer ) ) ? previousLayer.activations : previousLayer
        const localError          = ( isObject( nextLayer ) ) ? computeSumOfLocalsGradient( nextLayer, thisIndex ) : (nextLayer[ thisIndex ] - this.activation)
        const derivatedResult     = this.derivatedActivationFunction( this.weightedSum )
        const localGradient       = localError * derivatedResult

        // Weights
        for ( let index = 0, numberOfPreviousNeurons = previousLayerValues.length ; index < numberOfPreviousNeurons ; index++ ) {

            this.weights[ index ] += this.LEARNING_RATE * localGradient * previousLayerValues[ index ]

        }
        // Bias
        this.bias += this.LEARNING_RATE * -1.0 * localGradient

        this.localGradient = localGradient

        return this.localGradient

        function computeSumOfLocalsGradient ( nextLayer, thisIndex ) {

            let sum = 0.0

            for ( let neuronIndex = 0, numberOfNeuron = nextLayer.length ; neuronIndex < numberOfNeuron ; neuronIndex++ ) {
                let nextNeuron = nextLayer[ neuronIndex ]
                sum += nextNeuron.localGradient * nextNeuron.weights[ thisIndex ]
            }

            return sum

        }

    }

    fromJSON ( jsonData ) {

        this.weights            = jsonData.weights
        this.activationFunction = ActivationFunctionFactory.get( jsonData.activationFunction )
        this.bias               = jsonData.bias

    }

    toJSON () {

        return {
            weights:            this.weights,
            activationFunction: this.activationFunction.name,
            bias:               this.bias
        }

    }

}

export { Neuron }
