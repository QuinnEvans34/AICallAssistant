cmake_minimum_required(VERSION 3.20)
project(cpp_asr LANGUAGES CXX)

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

add_library(cpp_asr STATIC ${CPP_ASR_SOURCES} ${CPP_ASR_HEADERS})
target_include_directories(cpp_asr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(cpp_asr PUBLIC cxx_std_20)

# Optional pybind11 module target. Consumers can -DCPP_ASR_ENABLE_PYBIND11=ON.
option(CPP_ASR_ENABLE_PYBIND11 "Build pybind11 bindings" OFF)
if (CPP_ASR_ENABLE_PYBIND11)
  find_package(pybind11 REQUIRED)
  add_library(cpp_asr_native MODULE c_api_bindings.cpp)
  target_link_libraries(cpp_asr_native PRIVATE pybind11::module cpp_asr)
  target_compile_definitions(cpp_asr_native PRIVATE CPP_ASR_BUILD_PYBIND11)
  set_target_properties(cpp_asr_native PROPERTIES
    PREFIX ""
    OUTPUT_NAME "cpp_asr_native"
  )
endif()

