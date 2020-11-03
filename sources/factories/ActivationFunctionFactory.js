/**
 * @author [Tristan Valcke]{@link https://github.com/Itee}
 * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
 *
 * @file Todo
 *
 * @example Todo
 *
 */

class ActivationFunctionFactory {

    constructor ( catalog ) {

        this.catalog = catalog || {}

    }

    register ( key, value ) {

        this.catalog[ key ] = value

    }

    unregister ( key ) {

        delete this.catalog[ key ]

    }

    get ( key ) {

        return this.catalog[ key ]

    }

}

const factory = new ActivationFunctionFactory( {

    'sigmoid': {

        base: function sigmoid ( value ) {

            return 1.0 / (1.0 + Math.exp( -value ))

        },

        derivated: function derivatedSigmoid ( value ) {

            return ( (1.0 / (1.0 + Math.exp( -value ))) * (1 - (1.0 / (1.0 + Math.exp( -value )))) )

        }

    }

} )

export { factory as ActivationFunctionFactory }
