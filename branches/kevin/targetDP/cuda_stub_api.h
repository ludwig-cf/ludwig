/* Cuda stub interface. */

#ifndef CUDA_STUB_API_H
#define CUDA_STUB_API_H

#include <stdlib.h>

typedef enum cudaMemcpyKind_enum {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4}
  cudaMemcpyKind;

enum cudaError {
  cudaSuccess = 0,
  cudaErrorMissingConfiguration = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorLaunchFailure = 4,
  cudaErrorLaunchTimeout = 6,
  cudaErrorLaunchOutOfResources = 7,
  cudaErrorInvalidDeviceFunction = 8,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidDevice = 10,
  cudaErrorInvalidValue = 11,
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidMemcpyDirection = 21
};


/* geterror can return:
 cudaErrorUnmapBufferObjectFailed, cudaErrorInvalidHostPointer, cudaErrorInvalidDevicePointer, cudaErrorInvalidTexture, cudaErrorInvalidTextureBinding, cudaErrorInvalidChannelDescriptor, cudaErrorInvalidMemcpyDirection, cudaErrorInvalidFilterSetting, cudaErrorInvalidNormSetting, cudaErrorUnknown, cudaErrorInvalidResourceHandle, cudaErrorInsufficientDriver, cudaErrorSetOnActiveProcess, cudaErrorStartupFailure,
*/


/* exhaustive list
enum cudaError {
  cudaErrorPriorLaunchFailure = 5,
  cudaErrorMapBufferObjectFailed = 14,
  cudaErrorUnmapBufferObjectFailed = 15,
  cudaErrorInvalidHostPointer = 16,
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidTextureBinding = 19,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorAddressOfConstant = 22,
  cudaErrorTextureFetchFailed = 23,
  cudaErrorTextureNotBound = 24,
  cudaErrorSynchronizationError = 25,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorMixedDeviceExecution = 28,
  cudaErrorCudartUnloading = 29,
  cudaErrorUnknown = 30,
  cudaErrorNotYetImplemented = 31,
  cudaErrorMemoryValueTooLarge = 32,
  cudaErrorInvalidResourceHandle = 33,
  cudaErrorNotReady = 34,
  cudaErrorInsufficientDriver = 35,
  cudaErrorSetOnActiveProcess = 36,
  cudaErrorInvalidSurface = 37,
  cudaErrorNoDevice = 38,
  cudaErrorECCUncorrectable = 39,
  cudaErrorSharedObjectSymbolNotFound = 40,
  cudaErrorSharedObjectInitFailed = 41,
  cudaErrorUnsupportedLimit = 42,
  cudaErrorDuplicateVariableName = 43,
  cudaErrorDuplicateTextureName = 44,
  cudaErrorDuplicateSurfaceName = 45,
  cudaErrorDevicesUnavailable = 46,
  cudaErrorInvalidKernelImage = 47,
  cudaErrorNoKernelImageForDevice = 48,
  cudaErrorIncompatibleDriverContext = 49,
  cudaErrorPeerAccessAlreadyEnabled = 50,
  cudaErrorPeerAccessNotEnabled = 51,
  cudaErrorDeviceAlreadyInUse = 54,
  cudaErrorProfilerDisabled = 55,
  cudaErrorProfilerNotInitialized = 56,
  cudaErrorProfilerAlreadyStarted = 57,
  cudaErrorProfilerAlreadyStopped = 58,
  cudaErrorAssert = 59,
  cudaErrorTooManyPeers = 60,
  cudaErrorHostMemoryAlreadyRegistered = 61,
  cudaErrorHostMemoryNotRegistered = 62,
  cudaErrorOperatingSystem = 63,
  cudaErrorPeerAccessUnsupported = 64,
  cudaErrorLaunchMaxDepthExceeded = 65,
  cudaErrorLaunchFileScopedTex = 66,
  cudaErrorLaunchFileScopedSurf = 67,
  cudaErrorSyncDepthExceeded = 68,
  cudaErrorLaunchPendingCountExceeded = 69,
  cudaErrorNotPermitted = 70,
  cudaErrorNotSupported = 71,
  cudaErrorHardwareStackError = 72,
  cudaErrorIllegalInstruction = 73,
  cudaErrorMisalignedAddress = 74,
  cudaErrorInvalidAddressSpace = 75,
  cudaErrorInvalidPc = 76,
  cudaErrorIllegalAddress = 77,
  cudaErrorInvalidPtx = 78,
  cudaErrorInvalidGraphicsContext = 79,
  cudaErrorStartupFailure = 0x7f,
  cudaErrorApiFailureBase = 10000
};
*/

typedef enum cudaError cudaError_t;     /* an enum type */
typedef int * cudaStream_t;             /* an opaque handle */

cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaFree(void ** devPtr);
const char *  cudaGetErrorString(cudaError_t error);
cudaError_t cudaGetLastError(void);
cudaError_t cudaMalloc(void ** devRtr, size_t size);
cudaError_t cudaMemcpy(void * dst, const void * src, size_t count,
		       cudaMemcpyKind kind);
cudaError_t cudaMemset(void * devPtr, int value, size_t count);

#endif
