/**
 * @author [Tristan Valcke]{@link https://github.com/Itee}
 * @license [BSD-3-Clause]{@link https://opensource.org/licenses/BSD-3-Clause}
 *
 * @file Todo
 *
 * @example Todo
 *
 */

import { FileLoader } from './FileLoader'
import {
    BinaryReader,
    Endianness
} from './BinaryReader'


class IDXImagesLoader {

    constructor () {

    }

    load ( url, onSuccess, onProgress, onError ) {

        const scope = this

        const loader = new FileLoader( scope.manager )
        loader.setResponseType( 'arraybuffer' );
        loader.load( url, data => {

            scope.parse( data, onSuccess, onProgress, onError )

        }, onProgress, onError )

    }

    parse ( data, onSuccess, onProgress, onError ) {

        const binaryReader = new BinaryReader( data, 0, data.byteLength, Endianness.Big )

        const magicNumber = binaryReader.getInt32()
        if ( magicNumber !== 2051 ) {
            if ( onError ) {
                onError( 'Invalid magic number flag ! Abort parsing...' )
            }
            return
        }

        const numberOfItems = binaryReader.getInt32()
        if ( magicNumber === 0 ) {
            if ( onError ) {
                onError( 'Number of labels is 0 ! Abort parsing...' )
            }
            return
        }

        const imageHeight = binaryReader.getInt32()
        if ( imageHeight !== 28 ) {
            if ( onError ) {
                onError( 'Invalid image height ! Abort parsing...' )
            }
            return
        }

        const imageWidth = binaryReader.getInt32()
        if ( imageWidth !== 28 ) {
            if ( onError ) {
                onError( 'Invalid image width ! Abort parsing...' )
            }
            return
        }

        const length = imageHeight * imageWidth

        let images = new Array( numberOfItems )
        for ( let index = 0 ; index < numberOfItems ; index++ ) {

            let image = new Uint8Array( length )
            for ( let pixelIndex = 0 ; pixelIndex < length ; pixelIndex++ ) {
                image[ pixelIndex ] = binaryReader.getUint8()
            }
            images[ index ] = image

            if ( onProgress ) {
                onProgress( index + 1, numberOfItems )
            }

        }

        onSuccess( images )

    }

}

export { IDXImagesLoader }
