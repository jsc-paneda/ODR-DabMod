/*
   Copyright (C) 2005, 2006, 2007, 2008, 2009, 2010, 2011 Her Majesty
   the Queen in Right of Canada (Communications Research Center Canada)

   Copyright (C) 2016
   Matthias P. Braendli, matthias.braendli@mpb.li

    http://opendigitalradio.org
 */
/*
   This file is part of ODR-DabMod.

   ODR-DabMod is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as
   published by the Free Software Foundation, either version 3 of the
   License, or (at your option) any later version.

   ODR-DabMod is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with ODR-DabMod.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "OfdmGenerator.h"
#include "PcDebug.h"
#include "fftw3.h"
#define FFT_TYPE fftwf_complex

#include <stdio.h>
#include <string.h>
#include <stdexcept>
#include <assert.h>
#include <complex>
#include <malloc.h>
#include <cmath>
typedef std::complex<float> complexf;


OfdmGenerator::OfdmGenerator(size_t nbSymbols,
        size_t nbCarriers,
        size_t spacing,
        bool inverse) :
    ModCodec(ModFormat(nbSymbols * nbCarriers * sizeof(FFT_TYPE)),
            ModFormat(nbSymbols * spacing * sizeof(FFT_TYPE))),
    myFftPlan(NULL),
#if USE_GPU_FFT
    
#else
    myFftIn(NULL), myFftOut(NULL),
#endif
    
    myNbSymbols(nbSymbols),
    myNbCarriers(nbCarriers),
    mySpacing(spacing)
{
    PDEBUG("OfdmGenerator::OfdmGenerator(%zu, %zu, %zu, %s) @ %p\n",
            nbSymbols, nbCarriers, spacing, inverse ? "true" : "false", this);

    if (nbCarriers > spacing) {
        throw std::runtime_error(
                "OfdmGenerator::OfdmGenerator nbCarriers > spacing!");
    }

    if (inverse) {
        myPosDst = (nbCarriers & 1 ? 0 : 1);
        myPosSrc = 0;
        myPosSize = (nbCarriers + 1) / 2;
        myNegDst = spacing - (nbCarriers / 2);
        myNegSrc = (nbCarriers + 1) / 2;
        myNegSize = nbCarriers / 2;
    }
    else {
        myPosDst = (nbCarriers & 1 ? 0 : 1);
        myPosSrc = nbCarriers / 2;
        myPosSize = (nbCarriers + 1) / 2;
        myNegDst = spacing - (nbCarriers / 2);
        myNegSrc = 0;
        myNegSize = nbCarriers / 2;
    }
    myZeroDst = myPosDst + myPosSize;
    myZeroSize = myNegDst - myZeroDst;

    PDEBUG("  myPosDst: %u\n", myPosDst);
    PDEBUG("  myPosSrc: %u\n", myPosSrc);
    PDEBUG("  myPosSize: %u\n", myPosSize);
    PDEBUG("  myNegDst: %u\n", myNegDst);
    PDEBUG("  myNegSrc: %u\n", myNegSrc);
    PDEBUG("  myNegSize: %u\n", myNegSize);
    PDEBUG("  myZeroDst: %u\n", myZeroDst);
    PDEBUG("  myZeroSize: %u\n", myZeroSize);

#if USE_GPU_FFT
     mailBox = mbox_open();
     int ret = gpu_fft_prepare(mailBox, log2(mySpacing) /*log2_N*/, GPU_FFT_REV, myNbSymbols/*jobs*/, &fft); // call once
     switch(ret) {
        case -1: printf("Unable to enable V3D. Please check your firmware is up to date.\n");
        case -2: printf("log2_N=%d not supported.  Try between 8 and 22.\n", log2(mySpacing));
        case -3: printf("Out of memory.  Try a smaller batch or increase GPU memory.\n");
        case -4: printf("Unable to map Videocore peripherals into ARM memory space.\n");
        case -5: printf("Can't open libbcm_host.\n");
    }
#else
    const int N = mySpacing; // The size of the FFT
    myFftIn = (FFT_TYPE*)fftwf_malloc(sizeof(FFT_TYPE) * N);
    myFftOut = (FFT_TYPE*)fftwf_malloc(sizeof(FFT_TYPE) * N);
    myFftPlan = fftwf_plan_dft_1d(N,
            myFftIn, myFftOut,
            FFTW_BACKWARD, FFTW_MEASURE);

    if (sizeof(complexf) != sizeof(FFT_TYPE)) {
        printf("sizeof(complexf) %zu\n", sizeof(complexf));
        printf("sizeof(FFT_TYPE) %zu\n", sizeof(FFT_TYPE));
        throw std::runtime_error(
                "OfdmGenerator::process complexf size is not FFT_TYPE size!");
    }
#endif
}


OfdmGenerator::~OfdmGenerator()
{
    PDEBUG("OfdmGenerator::~OfdmGenerator() @ %p\n", this);

#if USE_GPU_FFT
    gpu_fft_release(fft); // Videocore memory lost if not freed !
#else
    if (myFftIn) {
         fftwf_free(myFftIn);
    }

    if (myFftOut) {
         fftwf_free(myFftOut);
    }

    if (myFftPlan) {
        fftwf_destroy_plan(myFftPlan);
    }
#endif
}

int OfdmGenerator::process(Buffer* const dataIn, Buffer* dataOut)
{
    PDEBUG("OfdmGenerator::process(dataIn: %p, dataOut: %p)\n",
            dataIn, dataOut);

    dataOut->setLength(myNbSymbols * mySpacing * sizeof(complexf));

    FFT_TYPE* in = reinterpret_cast<FFT_TYPE*>(dataIn->getData());
    FFT_TYPE* out = reinterpret_cast<FFT_TYPE*>(dataOut->getData());

    size_t sizeIn = dataIn->getLength() / sizeof(complexf);
    size_t sizeOut = dataOut->getLength() / sizeof(complexf);

    if (sizeIn != myNbSymbols * myNbCarriers) {
        PDEBUG("Nb symbols: %zu\n", myNbSymbols);
        PDEBUG("Nb carriers: %zu\n", myNbCarriers);
        PDEBUG("Spacing: %zu\n", mySpacing);
        PDEBUG("\n%zu != %zu\n", sizeIn, myNbSymbols * myNbCarriers);
        throw std::runtime_error(
                "OfdmGenerator::process input size not valid!");
    }
    if (sizeOut != myNbSymbols * mySpacing) {
        PDEBUG("Nb symbols: %zu\n", myNbSymbols);
        PDEBUG("Nb carriers: %zu\n", myNbCarriers);
        PDEBUG("Spacing: %zu\n", mySpacing);
        PDEBUG("\n%zu != %zu\n", sizeIn, myNbSymbols * mySpacing);
        throw std::runtime_error(
                "OfdmGenerator::process output size not valid!");
    }

#if USE_GPU_FFT
    struct GPU_FFT_COMPLEX *gpuIn;
    struct GPU_FFT_COMPLEX *gpuOut;

    for (size_t i = 0; i < myNbSymbols; ++i) {
      gpuIn = fft->in + i*fft->step;
          
      gpuIn[0].re = 0;
      gpuIn[0].im = 0;
      
      bzero(&gpuIn[myZeroDst], myZeroSize * sizeof(GPU_FFT_COMPLEX));
      memcpy(&gpuIn[myPosDst], &in[myPosSrc],
      	     myPosSize * sizeof(GPU_FFT_COMPLEX));
      memcpy(&gpuIn[myNegDst], &in[myNegSrc],
	     myNegSize * sizeof(GPU_FFT_COMPLEX));
      
      in += myNbCarriers;
    }
        
    gpu_fft_execute(fft);
    
    for (size_t i = 0; i < myNbSymbols; ++i) {
      gpuOut = fft->out + i*fft->step;  
      memcpy(out, gpuOut, mySpacing * sizeof(GPU_FFT_COMPLEX));
      
      out += mySpacing;
    }
#else
    for (size_t i = 0; i < myNbSymbols; ++i) {
        myFftIn[0][0] = 0;
        myFftIn[0][1] = 0;

        bzero(&myFftIn[myZeroDst], myZeroSize * sizeof(FFT_TYPE));
        memcpy(&myFftIn[myPosDst], &in[myPosSrc],
                myPosSize * sizeof(FFT_TYPE));
        memcpy(&myFftIn[myNegDst], &in[myNegSrc],
                myNegSize * sizeof(FFT_TYPE));

        fftwf_execute(myFftPlan);

        memcpy(out, myFftOut, mySpacing * sizeof(FFT_TYPE));

        in += myNbCarriers;
        out += mySpacing;
    }
#endif
    return sizeOut;
}

