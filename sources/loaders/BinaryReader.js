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
} )

/**
 *
 * @type {Object}
 */
const Byte = Object.freeze( {
    One:    1,
    Two:    2,
    Four:   4,
    Height: 8
} )

/**
 *
 * @param buffer
 * @param offset
 * @param length
 * @param endianness
 * @constructor
 */
function BinaryReader ( buffer, offset, length, endianness ) {

    this._buffer     = buffer || new ArrayBuffer( 0 )
    this._offset     = offset || 0
    this._length     = length || (buffer) ? buffer.byteLength : 0
    this._endianness = (typeof endianness === 'boolean') ? endianness : Endianness.Little

    this._updateDataView()

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

        const currentOffset = this._offset
        this._offset += increment
        return currentOffset

    },

    /**
     *
     * @private
     */
    _updateDataView () {

        this._dataView = new DataView( this._buffer, this._offset, this._length )

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

        this._offset = offset

        return this

    },

    /**
     *
     * @param nBytes
     * @return {skipOffsetOf}
     */
    skipOffsetOf ( nBytes ) {

        this._offset += nBytes

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

        this._buffer = buffer
        this._offset = offset || 0
        this._length = length || buffer.byteLength

        this._updateDataView()

        return this

    },

    /**
     *
     * @param endianess
     * @return {setEndianess}
     */
    setEndianess ( endianess ) {

        this._endianness = endianess

        return this

    },

    getBoolean () {

        return (( this.getUint8() & 1 ) === 1)

    },

    getBooleanArray ( length ) {

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getBoolean() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getInt8() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getUint8() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getInt16() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getUint16() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getInt32() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getUint32() )

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

        let low  = undefined
        let high = undefined

        if ( this._endianness === Endianness.Little ) {

            low  = this.getUint32()
            high = this.getUint32()

        } else {

            high = this.getUint32()
            low  = this.getUint32()

        }

        // calculate negative value
        if ( high & 0x80000000 ) {

            high = ~high & 0xFFFFFFFF
            low  = ~low & 0xFFFFFFFF

            if ( low === 0xFFFFFFFF ) {
                high = ( high + 1 ) & 0xFFFFFFFF
            }

            low = ( low + 1 ) & 0xFFFFFFFF

            return -( high * 0x100000000 + low )

        }

        return high * 0x100000000 + low

    },

    getInt64Array ( length ) {

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getInt64() )

        }

        return array

    },

    // Note: see getInt64() comment
    getUint64 () {

        let low  = undefined
        let high = undefined

        if ( this._endianness === Endianness.Little ) {

            low  = this.getUint32()
            high = this.getUint32()

        } else {

            high = this.getUint32()
            low  = this.getUint32()

        }

        return high * 0x100000000 + low

    },

    getUint64Array ( length ) {

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getUint64() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getFloat32() )

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

        const array = []

        for ( let i = 0 ; i < length ; i++ ) {

            array.push( this.getFloat64() )

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

        let string   = ''
        let charCode = undefined

        for ( let i = 0 ; i < length ; i++ ) {
            charCode = this.getUInt8()

            if ( charCode === 0 ) {
                continue
            }

            string += String.fromCharCode( charCode )
        }

        if ( trim ) {
            string = string.trim()
        }

        return string

    },

    getArrayBuffer ( size ) {

        const offset = this._getAndUpdateOffsetBy( size )
        return this._dataView.buffer.slice( offset, offset + size )

    },

} )

export {
    BinaryReader,
    Endianness,
    Byte
}
