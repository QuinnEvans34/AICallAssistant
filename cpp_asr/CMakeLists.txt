cmake_minimum_required(VERSION 3.20)
project(cpp_asr VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

set(CPP_ASR_SOURCES
    audio_ringbuffer.cpp
    vad_detector.cpp
    whisper_wrapper.cpp
    snapshot_worker.cpp
    heartbeat_logger.cpp
    exception_logger.cpp
    asr_engine.cpp
    c_api_bindings.cpp
)

set(CPP_ASR_HEADERS
    audio_ringbuffer.h
    vad_detector.h
    whisper_wrapper.h
    snapshot_worker.h
    heartbeat_logger.h
    exception_logger.h
    asr_engine.h
    c_api_bindings.h
)

add_library(cpp_asr SHARED ${CPP_ASR_SOURCES} ${CPP_ASR_HEADERS})
target_include_directories(cpp_asr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cpp_asr PUBLIC cxx_std_20)

if (WIN32)
    target_link_libraries(cpp_asr PRIVATE ws2_32 winmm)
endif()

set_target_properties(
    cpp_asr
    PROPERTIES
        OUTPUT_NAME "cpp_asr"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
)

# Optional pybind11 module target. Consumers can -DCPP_ASR_ENABLE_PYBIND11=ON.
option(CPP_ASR_ENABLE_PYBIND11 "Build pybind11 bindings" OFF)
if (CPP_ASR_ENABLE_PYBIND11)
  find_package(Python3 3.12 COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)

  # Build Python extension module (.pyd on Windows)
  pybind11_add_module(cpp_asr_native c_api_bindings.cpp)
  target_link_libraries(cpp_asr_native PRIVATE cpp_asr)
  target_compile_definitions(cpp_asr_native PRIVATE CPP_ASR_BUILD_PYBIND11)
  target_include_directories(cpp_asr_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

  set_target_properties(cpp_asr_native PROPERTIES
    OUTPUT_NAME "cpp_asr_native"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
  )
endif()
