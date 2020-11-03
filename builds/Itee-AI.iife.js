var AI = (function (exports) {
    'use strict';

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

            this.catalog = catalog || {};

        }

        register ( key, value ) {

            this.catalog[ key ] = value;

        }

        unregister ( key ) {

            delete this.catalog[ key ];

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

    } );

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    function isObject ( value ) {

        return ( value !== null && typeof value === 'object' && !Array.isArray( value ) )

    }

    class Neuron {

        constructor ( weights, bias, activationFunction, derivatedActivationFunction ) {

            this.weights                     = weights || [ 1.0 ];
            this.bias                        = bias || 0.0;
            this.activationFunction          = activationFunction || undefined;
            this.derivatedActivationFunction = derivatedActivationFunction || undefined;

            this.weightedSum   = undefined;
            this.activation    = undefined;
            this.localGradient = undefined;

            this.LEARNING_RATE = 1.0;
            //        this.LEARNING_RATE = 0.01

        }

        evaluate ( inputs ) {

            const numberOfInputs = inputs.length;
            const weights        = this.weights;

            // Weights
            this.weightedSum = 0.0;
            for ( let index = 0 ; index < numberOfInputs ; index++ ) {
                this.weightedSum += inputs[ index ] * weights[ index ];
            }
            // Bias
            this.weightedSum += -1.0 * this.bias;

            this.activation = this.activationFunction( this.weightedSum );

            return this.activation

        }

        learn ( previousLayer, nextLayer, thisIndex ) {

            const previousLayerValues = ( isObject( previousLayer ) ) ? previousLayer.activations : previousLayer;
            const localError          = ( isObject( nextLayer ) ) ? computeSumOfLocalsGradient( nextLayer, thisIndex ) : (nextLayer[ thisIndex ] - this.activation);
            const derivatedResult     = this.derivatedActivationFunction( this.weightedSum );
            const localGradient       = localError * derivatedResult;

            // Weights
            for ( let index = 0, numberOfPreviousNeurons = previousLayerValues.length ; index < numberOfPreviousNeurons ; index++ ) {

                this.weights[ index ] += this.LEARNING_RATE * localGradient * previousLayerValues[ index ];

            }
            // Bias
            this.bias += this.LEARNING_RATE * -1.0 * localGradient;

            this.localGradient = localGradient;

            return this.localGradient

            function computeSumOfLocalsGradient ( nextLayer, thisIndex ) {

                let sum = 0.0;

                for ( let neuronIndex = 0, numberOfNeuron = nextLayer.length ; neuronIndex < numberOfNeuron ; neuronIndex++ ) {
                    let nextNeuron = nextLayer[ neuronIndex ];
                    sum += nextNeuron.localGradient * nextNeuron.weights[ thisIndex ];
                }

                return sum

            }

        }

        fromJSON ( jsonData ) {

            this.weights            = jsonData.weights;
            this.activationFunction = factory.get( jsonData.activationFunction );
            this.bias               = jsonData.bias;

        }

        toJSON () {

            return {
                weights:            this.weights,
                activationFunction: this.activationFunction.name,
                bias:               this.bias
            }

        }

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    let neuralLayerInstanceCounter = 0;

    class NeuralLayer {

        constructor ( name, neurons ) {

            this.name        = name || `NeuralLayer_${neuralLayerInstanceCounter++}`;
            this.neurons     = neurons || [];

            this.activations = undefined;
            this.gradients   = undefined;

        }

        evaluate ( previousLayerResult ) {

            const neurons         = this.neurons;
            const numberOfNeurons = neurons.length;

            let currentLayerResult = new Array( numberOfNeurons );
            for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
                currentLayerResult[ neuronIndex ] = neurons[ neuronIndex ].evaluate( previousLayerResult );
            }

            this.activations = currentLayerResult;

        }

        learn ( previousLayer, nextLayer ) {

            const neurons         = this.neurons;
            const numberOfNeurons = neurons.length;

            let currentLocalsGradients = new Array( numberOfNeurons );
            for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
                currentLocalsGradients[ neuronIndex ] = neurons[ neuronIndex ].learn( previousLayer, nextLayer, neuronIndex );
            }

            this.gradients = currentLocalsGradients;

        }

        fromJSON ( jsonData ) {

            const numberOfNeurons = jsonData.length;

            this.neurons = [];
            for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {

                const neuron = new Neuron();
                neuron.fromJSON( jsonData[ neuronIndex ] );
                this.neurons.push( neuron );

            }

        }

        toJSON () {

            const neurons         = this.neurons;
            const numberOfNeurons = neurons.length;

            let neuronsArray = [];
            for ( let neuronIndex = 0 ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
                neuronsArray.push( neurons[ neuronIndex ].toJSON() );
            }

            return neuronsArray

        }

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    let neuralNetworkInstanceCounter = 0;

    class NeuralNetwork {

        constructor ( name, neuralLayers ) {

            this.name         = name || `NeuralNetwork_${neuralNetworkInstanceCounter++}`;
            this.neuralLayers = neuralLayers || [];

        }

        evaluate ( values ) {

            const neuralLayers   = this.neuralLayers;
            const numberOfLayers = neuralLayers.length;

            let previousLayer = undefined;

            for ( let layerIndex = 0 ; layerIndex < numberOfLayers ; layerIndex++ ) {

                previousLayer = ( layerIndex === 0 ) ? values : neuralLayers[ layerIndex - 1 ].activations;
                neuralLayers[ layerIndex ].evaluate( previousLayer );

            }

        }

        learn ( values, expectedResults ) {

            const neuralLayers   = this.neuralLayers;
            const numberOfLayers = neuralLayers.length;

            let previousLayer = undefined;
            let currentLayer  = undefined;
            let nextLayer     = undefined;

            for ( let layerIndex = numberOfLayers - 1 ; layerIndex >= 0 ; layerIndex-- ) {

                previousLayer = ( layerIndex === 0 ) ? values : neuralLayers[ layerIndex - 1 ];
                currentLayer  = neuralLayers[ layerIndex ];
                nextLayer     = ( layerIndex === numberOfLayers - 1 ) ? expectedResults : neuralLayers[ layerIndex + 1 ];

                currentLayer.learn( previousLayer, nextLayer );

            }

        }

        fromJSON ( jsonData ) {

            const numberOfNeuralLayers = jsonData.length;

            this.neuralLayers = [];
            for ( let neuralLayerIndex = 0 ; neuralLayerIndex < numberOfNeuralLayers ; neuralLayerIndex++ ) {

                const neuralLayer = new NeuralLayer();
                neuralLayer.fromJSON( jsonData[ neuralLayerIndex ] );
                this.neuralLayers.push( neuralLayer );

            }

        }

        toJSON () {

            const neuralLayers    = this.neuralLayers;
            const numberOfNeurons = neuralLayers.length;

            let neuralLayersArray = [];
            for ( let neuralLayerIndex = 0 ; neuralLayerIndex < numberOfNeurons ; neuralLayerIndex++ ) {
                neuralLayersArray.push( neuralLayers[ neuralLayerIndex ].toJSON() );
            }

            return neuralLayersArray

        }

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    let instanceCounter = 0;

    class ArtificialIntelligence {

        constructor ( name, neuralNetwork ) {

            this.name          = name || `IA_${instanceCounter++}`;
            this.neuralNetwork = neuralNetwork || [];

        }

        train ( dataset ) {

            for ( let i = 0 ; i <= 50000 ; i++ ) {

                let errorSum = 0.0;

                for ( let dataIndex = 0, numberOfDatas = dataset.length ; dataIndex < numberOfDatas ; dataIndex++ ) {

                    const data            = dataset[ dataIndex ];
                    const values          = data.values;
                    const expectedResults = data.expects;

                    this.neuralNetwork.evaluate( values );
                    this.neuralNetwork.learn( values, expectedResults );

                    errorSum += Math.pow( expectedResults[0] - this.neuralNetwork.neuralLayers[this.neuralNetwork.neuralLayers.length - 1].activations[0], 2 );

                }

                console.log( `A: ${this.neuralNetwork.neuralLayers[this.neuralNetwork.neuralLayers.length - 1].activations} -> Squared error: ${errorSum}` );

            }

            console.log(this.neuralNetwork);

        }

        learn ( deltaErrors ) {

            let totalError = 0.0;

            for ( let errorIndex = 0, numberOfDeltas = deltaErrors.length ; errorIndex < numberOfDeltas ; errorIndex++ ) {
                totalError += deltaErrors[ errorIndex ];
            }

        }

        fromJSON ( jsonData ) {

            this.name = jsonData.name;

            this.neuralNetwork = new NeuralNetwork();
            this.neuralNetwork.fromJSON( jsonData.neuralNetwork );

        }

        toJSON () {

            return {
                name:          this.name,
                neuralNetwork: this.neuralNetwork.toJSON()
            }

        }

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    //////////////////////////////////////////////////////

    function getRandom ( min, max ) {

        return Math.random() * (max - min) + min

    }

    function makeNeuron ( numberOfWeights, activationFunctionType ) {

        const weights = [];
        for ( let weightIndex = 0 ; weightIndex < numberOfWeights ; weightIndex++ ) {
            weights.push( getRandom( -1.0, 1.0 ) );
        }

        const bias = getRandom( -1.0, 1.0 );

        const activationFunction = factory.get( activationFunctionType );

        return new Neuron( weights, bias, activationFunction.base, activationFunction.derivated )

    }

    function makeLayer ( layerTopology, numberOfNeuronsOnPreviousLayer ) {

        let neurons = [];
        for ( let neuronIndex = 0, numberOfNeurons = layerTopology.numberOfNeurons ; neuronIndex < numberOfNeurons ; neuronIndex++ ) {
            neurons.push( makeNeuron( numberOfNeuronsOnPreviousLayer, 'sigmoid' ) );
        }

        return new NeuralLayer( layerTopology.name, neurons )

    }

    function makeNeuralNetwork ( neuralNetworkTopology ) {

        const layersTopology = neuralNetworkTopology.layers;

        let layers = [];
        for ( let layerIndex = 0, numberOfLayers = layersTopology.length ; layerIndex < numberOfLayers ; layerIndex++ ) {
            let layerTopology = layersTopology[ layerIndex ];
            layers.push( makeLayer( layerTopology, (layerIndex === 0) ? layerTopology.numberOfNeurons : layersTopology[ layerIndex - 1 ].numberOfNeurons ) );
        }

        return new NeuralNetwork( neuralNetworkTopology.name, layers )

    }

    function makeArtificialIntelligence ( topology ) {

        const neuralNetworkTopology = topology.networks;

        let networks = [];
        for ( let networkIndex = 0, numberOfNetworks = neuralNetworkTopology.length ; networkIndex < numberOfNetworks ; networkIndex++ ) {
            networks.push( makeNeuralNetwork( neuralNetworkTopology[ networkIndex ] ) );
        }

        return new ArtificialIntelligence( topology.name, networks )

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    var Cache = {

        enabled: false,

        files: {},

        add: function ( key, file ) {

            if ( this.enabled === false ) { return; }

            // console.log( 'Cache', 'Adding key:', key );

            this.files[ key ] = file;

        },

        get: function ( key ) {

            if ( this.enabled === false ) { return; }

            // console.log( 'Cache', 'Checking key:', key );

            return this.files[ key ];

        },

        remove: function ( key ) {

            delete this.files[ key ];

        },

        clear: function () {

            this.files = {};

        }

    };

    function LoadingManager ( onLoad, onProgress, onError ) {

        var scope = this;

        var isLoading   = false;
        var itemsLoaded = 0;
        var itemsTotal  = 0;
        var urlModifier = undefined;

        this.onStart    = undefined;
        this.onLoad     = onLoad;
        this.onProgress = onProgress;
        this.onError    = onError;

        this.itemStart = function ( url ) {

            itemsTotal++;

            if ( isLoading === false ) {

                if ( scope.onStart !== undefined ) {

                    scope.onStart( url, itemsLoaded, itemsTotal );

                }

            }

            isLoading = true;

        };

        this.itemEnd = function ( url ) {

            itemsLoaded++;

            if ( scope.onProgress !== undefined ) {

                scope.onProgress( url, itemsLoaded, itemsTotal );

            }

            if ( itemsLoaded === itemsTotal ) {

                isLoading = false;

                if ( scope.onLoad !== undefined ) {

                    scope.onLoad();

                }

            }

        };

        this.itemError = function ( url ) {

            if ( scope.onError !== undefined ) {

                scope.onError( url );

            }

        };

        this.resolveURL = function ( url ) {

            if ( urlModifier ) {

                return urlModifier( url );

            }

            return url;

        };

        this.setURLModifier = function ( transform ) {

            urlModifier = transform;
            return this;

        };

    }

    var loading = {};

    class FileLoader {

        constructor () {

            this.manager = new LoadingManager();

        }

        load ( url, onLoad, onProgress, onError ) {
            var this$1 = this;

            if ( url === undefined ) { url = ''; }

            if ( this.path !== undefined ) { url = this.path + url; }

            url = this.manager.resolveURL( url );

            var scope = this;

            var cached = Cache.get( url );

            if ( cached !== undefined ) {

                scope.manager.itemStart( url );

                setTimeout( function () {

                    if ( onLoad ) { onLoad( cached ); }

                    scope.manager.itemEnd( url );

                }, 0 );

                return cached;

            }

            // Check if request is duplicate

            if ( loading[ url ] !== undefined ) {

                loading[ url ].push( {

                    onLoad:     onLoad,
                    onProgress: onProgress,
                    onError:    onError

                } );

                return;

            }

            // Check for data: URI
            var dataUriRegex       = /^data:(.*?)(;base64)?,(.*)$/;
            var dataUriRegexResult = url.match( dataUriRegex );

            // Safari can not handle Data URIs through XMLHttpRequest so process manually
            if ( dataUriRegexResult ) {

                var mimeType = dataUriRegexResult[ 1 ];
                var isBase64 = !!dataUriRegexResult[ 2 ];
                var data     = dataUriRegexResult[ 3 ];

                data = window.decodeURIComponent( data );

                if ( isBase64 ) { data = window.atob( data ); }

                try {

                    var response;
                    var responseType = ( this.responseType || '' ).toLowerCase();

                    switch ( responseType ) {

                        case 'arraybuffer':
                        case 'blob':

                            var view = new Uint8Array( data.length );

                            for ( var i = 0 ; i < data.length ; i++ ) {

                                view[ i ] = data.charCodeAt( i );

                            }

                            if ( responseType === 'blob' ) {

                                response = new Blob( [ view.buffer ], { type: mimeType } );

                            } else {

                                response = view.buffer;

                            }

                            break;

                        case 'document':

                            var parser = new DOMParser();
                            response   = parser.parseFromString( data, mimeType );

                            break;

                        case 'json':

                            response = JSON.parse( data );

                            break;

                        default: // 'text' or other

                            response = data;

                            break;

                    }

                    // Wait for next browser tick like standard XMLHttpRequest event dispatching does
                    window.setTimeout( function () {

                        if ( onLoad ) { onLoad( response ); }

                        scope.manager.itemEnd( url );

                    }, 0 );

                } catch ( error ) {

                    // Wait for next browser tick like standard XMLHttpRequest event dispatching does
                    window.setTimeout( function () {

                        if ( onError ) { onError( error ); }

                        scope.manager.itemEnd( url );
                        scope.manager.itemError( url );

                    }, 0 );

                }

            } else {

                // Initialise array for duplicate requests

                loading[ url ] = [];

                loading[ url ].push( {

                    onLoad:     onLoad,
                    onProgress: onProgress,
                    onError:    onError

                } );

                var request = new XMLHttpRequest();

                request.open( 'GET', url, true );

                request.addEventListener( 'load', function ( event ) {

                    var response = this.response;

                    Cache.add( url, response );

                    var callbacks = loading[ url ];

                    delete loading[ url ];

                    if ( this.status === 200 ) {

                        for ( var i = 0, il = callbacks.length ; i < il ; i++ ) {

                            var callback = callbacks[ i ];
                            if ( callback.onLoad ) { callback.onLoad( response ); }

                        }

                        scope.manager.itemEnd( url );

                    } else if ( this.status === 0 ) {

                        // Some browsers return HTTP Status 0 when using non-http protocol
                        // e.g. 'file://' or 'data://'. Handle as success.

                        console.warn( 'FileLoader: HTTP Status 0 received.' );

                        for ( var i = 0, il = callbacks.length ; i < il ; i++ ) {

                            var callback = callbacks[ i ];
                            if ( callback.onLoad ) { callback.onLoad( response ); }

                        }

                        scope.manager.itemEnd( url );

                    } else {

                        for ( var i = 0, il = callbacks.length ; i < il ; i++ ) {

                            var callback = callbacks[ i ];
                            if ( callback.onError ) { callback.onError( event ); }

                        }

                        scope.manager.itemEnd( url );
                        scope.manager.itemError( url );

                    }

                }, false );

                request.addEventListener( 'progress', function ( event ) {

                    var callbacks = loading[ url ];

                    for ( var i = 0, il = callbacks.length ; i < il ; i++ ) {

                        var callback = callbacks[ i ];
                        if ( callback.onProgress ) { callback.onProgress( event ); }

                    }

                }, false );

                request.addEventListener( 'error', function ( event ) {

                    var callbacks = loading[ url ];

                    delete loading[ url ];

                    for ( var i = 0, il = callbacks.length ; i < il ; i++ ) {

                        var callback = callbacks[ i ];
                        if ( callback.onError ) { callback.onError( event ); }

                    }

                    scope.manager.itemEnd( url );
                    scope.manager.itemError( url );

                }, false );

                if ( this.responseType !== undefined ) { request.responseType = this.responseType; }
                if ( this.withCredentials !== undefined ) { request.withCredentials = this.withCredentials; }

                if ( request.overrideMimeType ) { request.overrideMimeType( this.mimeType !== undefined ? this.mimeType : 'text/plain' ); }

                for ( var header in this$1.requestHeader ) {

                    request.setRequestHeader( header, this$1.requestHeader[ header ] );

                }

                request.send( null );

            }

            scope.manager.itemStart( url );

            return request;

        }

        setPath ( value ) {

            this.path = value;
            return this;

        }

        setResponseType ( value ) {

            this.responseType = value;
            return this;

        }

        setWithCredentials ( value ) {

            this.withCredentials = value;
            return this;

        }

        setMimeType ( value ) {

            this.mimeType = value;
            return this;

        }

        setRequestHeader ( value ) {

            this.requestHeader = value;
            return this;

        }

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @class Todo...
     * @classdesc Todo...
     * @example Todo...
     *
     */

    /* eslint-env browser */

    /**
     *
     * @type {Object}
     */
    const Endianness = Object.freeze( {
        Little: true,
        Big:    false
    } );

    /**
     *
     * @type {Object}
     */
    const Byte = Object.freeze( {
        One:    1,
        Two:    2,
        Four:   4,
        Height: 8
    } );

    /**
     *
     * @param buffer
     * @param offset
     * @param length
     * @param endianness
     * @constructor
     */
    function BinaryReader ( buffer, offset, length, endianness ) {

        this._buffer     = buffer || new ArrayBuffer( 0 );
        this._offset     = offset || 0;
        this._length     = length || (buffer) ? buffer.byteLength : 0;
        this._endianness = (typeof endianness === 'boolean') ? endianness : Endianness.Little;

        this._updateDataView();

    }

    Object.assign( BinaryReader.prototype, {

        constructor: BinaryReader,

        /**
         *
         * @param increment
         * @return {*}
         * @private
         */
        _getAndUpdateOffsetBy ( increment ) {

            const currentOffset = this._offset;
            this._offset += increment;
            return currentOffset

        },

        /**
         *
         * @private
         */
        _updateDataView () {

            this._dataView = new DataView( this._buffer, this._offset, this._length );

        },

        /**
         *
         * @return {boolean}
         */
        isEndOfFile () {

            return ( this._offset === this._length )

        },

        /**
         *
         */
        getOffset () {
            return this._offset
        },

        /**
         *
         * @param offset
         * @return {skipOffsetTo}
         */
        skipOffsetTo ( offset ) {

            this._offset = offset;

            return this

        },

        /**
         *
         * @param nBytes
         * @return {skipOffsetOf}
         */
        skipOffsetOf ( nBytes ) {

            this._offset += nBytes;

            return this

        },

        /**
         *
         * @param buffer
         * @param offset
         * @param length
         * @return {setBuffer}
         */
        setBuffer ( buffer, offset, length ) {

            this._buffer = buffer;
            this._offset = offset || 0;
            this._length = length || buffer.byteLength;

            this._updateDataView();

            return this

        },

        /**
         *
         * @param endianess
         * @return {setEndianess}
         */
        setEndianess ( endianess ) {

            this._endianness = endianess;

            return this

        },

        getBoolean () {

            return (( this.getUint8() & 1 ) === 1)

        },

        getBooleanArray ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getBoolean() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getInt8 () {

            return this._dataView.getInt8( this._getAndUpdateOffsetBy( Byte.One ) )

        },

        getInt8Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getInt8() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getUint8 () {

            return this._dataView.getUint8( this._getAndUpdateOffsetBy( Byte.One ) )

        },

        getUint8Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getUint8() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getInt16 () {

            return this._dataView.getInt16( this._getAndUpdateOffsetBy( Byte.Two ), this._endianness )

        },

        getInt16Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getInt16() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getUint16 () {

            return this._dataView.getUint16( this._getAndUpdateOffsetBy( Byte.Two ), this._endianness )

        },

        getUint16Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getUint16() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getInt32 () {

            return this._dataView.getInt32( this._getAndUpdateOffsetBy( Byte.Four ), this._endianness )

        },

        getInt32Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getInt32() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getUint32 () {

            return this._dataView.getUint32( this._getAndUpdateOffsetBy( Byte.Four ), this._endianness )

        },

        getUint32Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getUint32() );

            }

            return array

        },

        // From THREE.FBXLoader
        // JavaScript doesn't support 64-bit integer so attempting to calculate by ourselves.
        // 1 << 32 will return 1 so using multiply operation instead here.
        // There'd be a possibility that this method returns wrong value if the value
        // is out of the range between Number.MAX_SAFE_INTEGER and Number.MIN_SAFE_INTEGER.
        // TODO: safely handle 64-bit integer
        getInt64 () {

            let low  = undefined;
            let high = undefined;

            if ( this._endianness === Endianness.Little ) {

                low  = this.getUint32();
                high = this.getUint32();

            } else {

                high = this.getUint32();
                low  = this.getUint32();

            }

            // calculate negative value
            if ( high & 0x80000000 ) {

                high = ~high & 0xFFFFFFFF;
                low  = ~low & 0xFFFFFFFF;

                if ( low === 0xFFFFFFFF ) {
                    high = ( high + 1 ) & 0xFFFFFFFF;
                }

                low = ( low + 1 ) & 0xFFFFFFFF;

                return -( high * 0x100000000 + low )

            }

            return high * 0x100000000 + low

        },

        getInt64Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getInt64() );

            }

            return array

        },

        // Note: see getInt64() comment
        getUint64 () {

            let low  = undefined;
            let high = undefined;

            if ( this._endianness === Endianness.Little ) {

                low  = this.getUint32();
                high = this.getUint32();

            } else {

                high = this.getUint32();
                low  = this.getUint32();

            }

            return high * 0x100000000 + low

        },

        getUint64Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getUint64() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getFloat32 () {

            return this._dataView.getFloat32( this._getAndUpdateOffsetBy( Byte.Four ), this._endianness )

        },

        getFloat32Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getFloat32() );

            }

            return array

        },

        /**
         *
         * @return {number}
         */
        getFloat64 () {

            return this._dataView.getFloat64( this._getAndUpdateOffsetBy( Byte.Height ), this._endianness )

        },

        getFloat64Array ( length ) {

            const array = [];

            for ( let i = 0 ; i < length ; i++ ) {

                array.push( this.getFloat64() );

            }

            return array

        },

        /**
         *
         * @return {string}
         */
        getChar () {

            return String.fromCharCode( this.getUInt8() )

        },

        /**
         *
         * @param length
         * @param trim
         * @return {string}
         */
        getString ( length, trim = true ) {

            let string   = '';
            let charCode = undefined;

            for ( let i = 0 ; i < length ; i++ ) {
                charCode = this.getUInt8();

                if ( charCode === 0 ) {
                    continue
                }

                string += String.fromCharCode( charCode );
            }

            if ( trim ) {
                string = string.trim();
            }

            return string

        },

        getArrayBuffer ( size ) {

            const offset = this._getAndUpdateOffsetBy( size );
            return this._dataView.buffer.slice( offset, offset + size )

        },

    } );

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */


    class IDXImagesLoader {

        constructor () {

        }

        load ( url, onSuccess, onProgress, onError ) {

            const scope = this;

            const loader = new FileLoader( scope.manager );
            loader.setResponseType( 'arraybuffer' );
            loader.load( url, data => {

                scope.parse( data, onSuccess, onProgress, onError );

            }, onProgress, onError );

        }

        parse ( data, onSuccess, onProgress, onError ) {

            const binaryReader = new BinaryReader( data, 0, data.byteLength, Endianness.Big );

            const magicNumber = binaryReader.getInt32();
            if ( magicNumber !== 2051 ) {
                if ( onError ) {
                    onError( 'Invalid magic number flag ! Abort parsing...' );
                }
                return
            }

            const numberOfItems = binaryReader.getInt32();
            if ( magicNumber === 0 ) {
                if ( onError ) {
                    onError( 'Number of labels is 0 ! Abort parsing...' );
                }
                return
            }

            const imageHeight = binaryReader.getInt32();
            if ( imageHeight !== 28 ) {
                if ( onError ) {
                    onError( 'Invalid image height ! Abort parsing...' );
                }
                return
            }

            const imageWidth = binaryReader.getInt32();
            if ( imageWidth !== 28 ) {
                if ( onError ) {
                    onError( 'Invalid image width ! Abort parsing...' );
                }
                return
            }

            const length = imageHeight * imageWidth;

            let images = new Array( numberOfItems );
            for ( let index = 0 ; index < numberOfItems ; index++ ) {

                let image = new Uint8Array( length );
                for ( let pixelIndex = 0 ; pixelIndex < length ; pixelIndex++ ) {
                    image[ pixelIndex ] = binaryReader.getUint8();
                }
                images[ index ] = image;

                if ( onProgress ) {
                    onProgress( index + 1, numberOfItems );
                }

            }

            onSuccess( images );

        }

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    class IDXLabelsLoader {

        constructor () {}

        load ( url, onSuccess, onProgress, onError ) {

            const scope = this;

            const loader = new FileLoader( scope.manager );
            loader.setResponseType( 'arraybuffer' );
            loader.load( url, data => {

                scope.parse( data, onSuccess, onProgress, onError );

            }, onProgress, onError );

        }

        parse ( data, onSuccess, onProgress, onError ) {

            const binaryReader = new BinaryReader( data, 0, data.byteLength, Endianness.Big );

            const magicNumber = binaryReader.getInt32();
            if ( magicNumber !== 2049 ) {
                if ( onError ) {
                    onError( 'Invalid magic number flag ! Abort parsing...' );
                }
                return
            }

            const numberOfItems = binaryReader.getInt32();
            if ( magicNumber === 0 ) {
                if ( onError ) {
                    onError( 'Number of labels is 0 ! Abort parsing...' );
                }
                return
            }

            let labels = new Array( numberOfItems );
            for ( let index = 0 ; index < numberOfItems ; index++ ) {
                labels[ index ] = binaryReader.getUint8();
                if ( onProgress ) {
                    onProgress( index + 1, numberOfItems );
                }
            }

            onSuccess( labels );

        }

    }

    /**
     * @author [Tristan Valcke]{@link https://github.com/Itee}
     * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
     *
     * @file Todo
     *
     * @example Todo
     *
     */

    exports.Neuron = Neuron;
    exports.NeuralLayer = NeuralLayer;
    exports.NeuralNetwork = NeuralNetwork;
    exports.ArtificialIntelligence = ArtificialIntelligence;
    exports.ActivationFunctionFactory = factory;
    exports.getRandom = getRandom;
    exports.makeNeuron = makeNeuron;
    exports.makeLayer = makeLayer;
    exports.makeNeuralNetwork = makeNeuralNetwork;
    exports.makeArtificialIntelligence = makeArtificialIntelligence;
    exports.IDXImagesLoader = IDXImagesLoader;
    exports.IDXLabelsLoader = IDXLabelsLoader;

    return exports;

}({}));
