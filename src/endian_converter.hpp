/*! **************************************************************************************

\class		EndianConverter
\copyright	Paneda Tech 2014
\author		FLa

Cross-platform endian converter functions.

\remark
This code expands on Steve Lorimer's post at 
http://stackoverflow.com/questions/105252/how-do-i-convert-between-big-endian-and-little-endian-values-in-c

\code
	// Example usage:
	// You have a hex value in memory with the following layout.
	char bytes[8] = { 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08 };

	// If you just type-cast it on an x86 compatible CPU, it will give you the value 
	// of 0x0807060504030201  because of the way intergers are stored in memory.
	
	// If you instead use the network to host function, it will grant the desired 
	// result.
	uint64_t myValue = endian::NtoH(*(uint64_t*)bytes);

	// Google endianess and network byte order to learn more about it.

	// Why not use the already existing functions, like ntoh() and hton()?
	// Because they are named a bit different on different platforms and thus not 
	// cross-platform compatible.

	// Another example:
	zmq::message_t msg(8);
	uint64_t seqNr = 1234;

	uint64_t networkByteOrderSeqNr = endian::HtoN(seqNr);
	// You can now cast it to char* and use std::memcpy.
	std::memcpy(msg.data(), (char*)&networkByteOrderSeqNr, 8);

	// Same thing using CZmqString
	auto nboValue = endian::HtoN(seqNr);
	message.push_back(CZmqString((char*)(&nboValue), sizeof(nboValue)));

\endcode

*************************************************************************************** */

// TODO: Optimize to use low-level functions for byte-swap on known platforms and just 
//		 use the bit-shift as a fall-back.

// TODO: Add method that ouputs the result directly into a byte-buffer.

#pragma once
#include <boost/type_traits.hpp>
#include <boost/static_assert.hpp>
#include <boost/detail/endian.hpp>
#include <stdexcept>
#include <stdint.h>

enum endianness
{
	little_endian = 0,
	big_endian = 1,
	network_endian = big_endian,

#if defined(BOOST_LITTLE_ENDIAN)
	host_endian = little_endian
#elif defined(BOOST_BIG_ENDIAN)
	host_endian = big_endian
#else
#error "unable to determine system endianness"
#endif
};

namespace Ofta
{
	namespace endian {
		namespace detail {

			template<typename T, size_t sz>
			struct swap_bytes
			{
				inline T operator()(T val)
				{
					throw std::out_of_range("data size");
				}
			};

			template<typename T>
			struct swap_bytes < T, 1 >
			{
				inline T operator()(T val)
				{
					return val;
				}
			};

			template<typename T>
			struct swap_bytes < T, 2 >
			{
				inline T operator()(T val)
				{
					return ((((val) >> 8) & 0xff) | (((val)& 0xff) << 8));
				}
			};

			template<typename T>
			struct swap_bytes < T, 4 >
			{
				inline T operator()(T val)
				{
					return ((((val)& 0xff000000) >> 24) |
						(((val)& 0x00ff0000) >> 8) |
						(((val)& 0x0000ff00) << 8) |
						(((val)& 0x000000ff) << 24));
				}
			};

			template<>
			struct swap_bytes < float, 4 >
			{
				inline float operator()(float val)
				{
					uint32_t mem = swap_bytes<uint32_t, sizeof(uint32_t)>()(*(uint32_t*)&val);
					return *(float*)&mem;
				}
			};

			template<typename T>
			struct swap_bytes < T, 8 >
			{
				inline T operator()(T val)
				{
					return ((((val)& 0xff00000000000000ull) >> 56) |
						(((val)& 0x00ff000000000000ull) >> 40) |
						(((val)& 0x0000ff0000000000ull) >> 24) |
						(((val)& 0x000000ff00000000ull) >> 8) |
						(((val)& 0x00000000ff000000ull) << 8) |
						(((val)& 0x0000000000ff0000ull) << 24) |
						(((val)& 0x000000000000ff00ull) << 40) |
						(((val)& 0x00000000000000ffull) << 56));
				}
			};

			template<>
			struct swap_bytes < double, 8 >
			{
				inline double operator()(double val)
				{
					uint64_t mem = swap_bytes<uint64_t, sizeof(uint64_t)>()(*(uint64_t*)&val);
					return *(double*)&mem;
				}
			};

			template<endianness from, endianness to, class T>
			struct do_byte_swap
			{
				inline T operator()(T value)
				{
					return swap_bytes<T, sizeof(T)>()(value);
				}
			};
			// specialisations when attempting to swap to the same endianess
			template<class T> struct do_byte_swap < little_endian, little_endian, T > { inline T operator()(T value) { return value; } };
			template<class T> struct do_byte_swap < big_endian, big_endian, T > { inline T operator()(T value) { return value; } };



		} // namespace detail

		template<endianness from, endianness to, class T>
		inline T ByteSwap(T value)
		{
			// ensure the data is only 1, 2, 4 or 8 bytes
			BOOST_STATIC_ASSERT(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8);
			// ensure we're only swapping arithmetic types
			BOOST_STATIC_ASSERT(boost::is_arithmetic<T>::value);

			return detail::do_byte_swap<from, to, T>()(value);
		}

		// Convenience functions
		template<class T>
		inline T NtoH(T value)
		{
			return ByteSwap<big_endian, host_endian, T>(value);
		}

		template<class T>
		inline T HtoN(T value)
		{
			return ByteSwap<host_endian, big_endian, T>(value);
		}
	} // namespace endian
} // namespace Ofta
