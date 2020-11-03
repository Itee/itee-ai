/**
 * @author [Tristan Valcke]{@link https://github.com/Itee}
 * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
 *
 * @file Todo
 *
 * @example Todo
 *
 */

import { NeuralNetwork } from './NeuralNetwork'

let instanceCounter = 0

class ArtificialIntelligence {

    constructor ( name, neuralNetwork ) {

        this.name          = name || `IA_${instanceCounter++}`
        this.neuralNetwork = neuralNetwork || []

    }

    train ( dataset ) {

        for ( let i = 0 ; i <= 50000 ; i++ ) {

            let errorSum = 0.0

            for ( let dataIndex = 0, numberOfDatas = dataset.length ; dataIndex < numberOfDatas ; dataIndex++ ) {

                const data            = dataset[ dataIndex ]
                const values          = data.values
                const expectedResults = data.expects

                this.neuralNetwork.evaluate( values )
                this.neuralNetwork.learn( values, expectedResults )

                errorSum += Math.pow( expectedResults[0] - this.neuralNetwork.neuralLayers[this.neuralNetwork.neuralLayers.length - 1].activations[0], 2 )

            }

            console.log( `A: ${this.neuralNetwork.neuralLayers[this.neuralNetwork.neuralLayers.length - 1].activations} -> Squared error: ${errorSum}` )

        }

        console.log(this.neuralNetwork)

    }

    learn ( deltaErrors ) {

        let totalError = 0.0

        for ( let errorIndex = 0, numberOfDeltas = deltaErrors.length ; errorIndex < numberOfDeltas ; errorIndex++ ) {
            totalError += deltaErrors[ errorIndex ]
        }

    }

    fromJSON ( jsonData ) {

        this.name = jsonData.name

        this.neuralNetwork = new NeuralNetwork()
        this.neuralNetwork.fromJSON( jsonData.neuralNetwork )

    }

    toJSON () {

        return {
            name:          this.name,
            neuralNetwork: this.neuralNetwork.toJSON()
        }

    }

}

export { ArtificialIntelligence }

