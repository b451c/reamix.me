cmake_minimum_required(VERSION 3.22)

if(APPLE)
  project(reaper_reamix VERSION 1.0.0 LANGUAGES C CXX OBJCXX)
else()
  project(reaper_reamix VERSION 1.0.0 LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Dependencies ---

# REAPER SDK
if(EXISTS "${CMAKE_SOURCE_DIR}/vendor/reaper-sdk/sdk/reaper_plugin.h")
  set(REAPER_SDK_DIR "${CMAKE_SOURCE_DIR}/vendor/reaper-sdk/sdk")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/vendor/reaper-sdk/reaper_plugin.h")
  set(REAPER_SDK_DIR "${CMAKE_SOURCE_DIR}/vendor/reaper-sdk")
else()
  message(FATAL_ERROR "REAPER SDK not found. Run: git submodule update --init")
endif()

# WDL
set(WDL_DIR "${CMAKE_SOURCE_DIR}/vendor/WDL/WDL")
if(NOT EXISTS "${WDL_DIR}/wdltypes.h")
  message(FATAL_ERROR "WDL not found. Run: git submodule update --init")
endif()

# JUCE
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/vendor/JUCE/CMakeLists.txt")
  message(FATAL_ERROR "JUCE not found. Run: git submodule update --init")
endif()
add_subdirectory(vendor/JUCE EXCLUDE_FROM_ALL)

# ONNX Runtime
set(ORT_DIR "${CMAKE_SOURCE_DIR}/vendor/onnxruntime")
if(NOT EXISTS "${ORT_DIR}/include/onnxruntime_cxx_api.h")
  message(WARNING "ONNX Runtime not found at ${ORT_DIR}. Inference will not be available.")
  set(ORT_FOUND FALSE)
else()
  set(ORT_FOUND TRUE)
  add_library(onnxruntime SHARED IMPORTED)
  if(APPLE)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ORT_DIR}/lib/libonnxruntime.dylib"
    )
  elseif(WIN32)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ORT_DIR}/lib/onnxruntime.dll"
      IMPORTED_IMPLIB "${ORT_DIR}/lib/onnxruntime.lib"
    )
  else()
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_LOCATION "${ORT_DIR}/lib/libonnxruntime.so"
    )
  endif()
  set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_DIR}/include"
  )
endif()

# PocketFFT (header-only)
set(POCKETFFT_DIR "${CMAKE_SOURCE_DIR}/vendor")

# --- Main target ---

# ADR-097 sesja 107 — dev calibration build flag DROPPED. AdvancedWeightsPanel
# + AdvancedWeightsWindow + DevCalibrationStorage now ship in every build as
# opt-in advanced settings (hidden by default; opened from gear → Advanced...
# button per SettingsPopover). The earlier REAMIX_DEV_CALIBRATION compile-time
# separation per ADR-087 STATUS UPDATE 1 (sesja 98) was superseded by ADR-097
# OSS direction (sesja 106) — community preset sharing requires the panel to
# be part of the shipped dylib.
file(GLOB_RECURSE REAMIX_SOURCES CONFIGURE_DEPENDS "src/*.cpp")
add_library(reaper_reamix SHARED ${REAMIX_SOURCES})

# Bundled UI fonts — Inter (OFL 1.1) + JetBrains Mono (OFL 1.1).
# juce_add_binary_data() generates BinaryData.h/cpp exposing each TTF as a
# (const char*, int) pair. Identifiers follow JUCE's filename → symbol rule
# (strip spaces/dashes, dots → "_"): Inter-Regular.ttf → InterRegular_ttf.
# See src/ui/LookAndFeelReamix.cpp § fontBlobFor().
juce_add_binary_data(reamix_font_binary SOURCES
  assets/fonts/Inter-Regular.ttf
  assets/fonts/Inter-Medium.ttf
  assets/fonts/Inter-SemiBold.ttf
  assets/fonts/Inter-Bold.ttf
  assets/fonts/JetBrainsMono-Regular.ttf
  assets/fonts/JetBrainsMono-Medium.ttf
  assets/fonts/JetBrainsMono-SemiBold.ttf
  # Session 49 — brand logo (replaces the simple rounded-square dot in
  # HeaderBar + WaveformView splash). Filename → JUCE symbol:
  # reamix-logo.png → reamixlogo_png / reamixlogo_pngSize. See
  # LookAndFeelReamix for the Image loader helper.
  assets/logo/reamix-logo.png
)

# JUCE modules
target_link_libraries(reaper_reamix PRIVATE
  reamix_font_binary
  juce::juce_gui_basics
  juce::juce_gui_extra
  juce::juce_audio_formats
  juce::juce_cryptography
)

if(ORT_FOUND)
  target_link_libraries(reaper_reamix PRIVATE onnxruntime)
  target_compile_definitions(reaper_reamix PRIVATE REAMIX_HAS_ONNX=1)
endif()

# BeatDetector::detectFile() is gated on REAMIX_WITH_REAPER_IO (uses
# PCM_Source_CreateFromFile, resolved at plugin load). The main plugin
# target defines it; parity-test targets do not.
# (juce_audio_basics, needed by BeatDetector.cpp for LagrangeInterpolator,
# is already pulled in transitively via juce_audio_formats.)
target_compile_definitions(reaper_reamix PRIVATE REAMIX_WITH_REAPER_IO=1)

target_include_directories(reaper_reamix PRIVATE
  ${REAPER_SDK_DIR}
  ${WDL_DIR}
  ${WDL_DIR}/swell
  ${POCKETFFT_DIR}
  src
)

target_compile_definitions(reaper_reamix PRIVATE
  JUCE_STANDALONE_APPLICATION=0
  JUCE_DISPLAY_SPLASH_SCREEN=0
  JUCE_MODAL_LOOPS_PERMITTED=1
  JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  JUCE_WEB_BROWSER=0
  $<$<PLATFORM_ID:Linux>:JUCE_USE_CURL=1>
  $<$<NOT:$<PLATFORM_ID:Linux>>:JUCE_USE_CURL=0>
  WDL_NO_DEFINE_MINMAX
  REAMIX_VERSION="${PROJECT_VERSION}"
  $<$<NOT:$<PLATFORM_ID:Windows>>:SWELL_PROVIDED_BY_APP>
  $<$<PLATFORM_ID:Windows>:NOMINMAX>
  $<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN>
  $<$<PLATFORM_ID:Windows>:JUCE_REMOVE_COMPONENT_FROM_DESKTOP_ON_WM_DESTROY=1>
  $<$<PLATFORM_ID:Linux>:JUCE_USE_XSHM=0>
  $<$<CONFIG:Debug>:REAMIX_DEBUG>
)

# Compiler warnings
if(MSVC)
  target_compile_options(reaper_reamix PRIVATE /W3)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
  target_compile_options(reaper_reamix PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers
    -fsigned-char  # ARM defaults to unsigned char; WDL/SWELL assumes signed
  )
endif()

# --- Platform-specific ---

if(APPLE)
  # Session 49: drop arch suffix from the output name. User directive —
  # exactly ONE `reaper_reamix.dylib` ever lives in UserPlugins (never a
  # `-arm64` variant shadowing it). Memory `feedback_one_dylib_install_
  # convention.md` + session-49 DEV-023 document the burn on session 49
  # where a stale `-arm64`-less file silently shadowed new builds.
  # Arch info is still recoverable via `file reaper_reamix.dylib` at runtime.
  set_target_properties(reaper_reamix PROPERTIES
    SUFFIX ".dylib"
    PREFIX ""
    OUTPUT_NAME "reaper_reamix"
  )

  # Set rpath to look for dylibs next to the plugin binary
  # SKIP_BUILD_RPATH=OFF + BUILD_RPATH_USE_ORIGIN=OFF + explicit BUILD_RPATH
  # prevents CMake from adding the ORT vendor path as an extra RPATH entry
  set_target_properties(reaper_reamix PROPERTIES
    BUILD_RPATH "@loader_path"
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH ON
  )

  target_link_libraries(reaper_reamix PRIVATE
    "-framework AppKit"
    "-framework CoreFoundation"
  )

  # SWELL mod stub (REAPER provides SWELL implementation)
  target_sources(reaper_reamix PRIVATE
    ${WDL_DIR}/swell/swell-modstub.mm
  )

elseif(WIN32)
  set_target_properties(reaper_reamix PROPERTIES
    SUFFIX ".dll"
    PREFIX ""
    OUTPUT_NAME "reaper_reamix-x64"
  )
  # Delay-load onnxruntime.dll so we can pre-load it from UserPlugins
  # before Windows resolves it from System32 (where v1.17 may exist)
  if(ORT_FOUND)
    target_link_options(reaper_reamix PRIVATE /DELAYLOAD:onnxruntime.dll)
    target_link_libraries(reaper_reamix PRIVATE delayimp shlwapi)
  endif()

elseif(UNIX)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(LINUX_ARCH "aarch64")
  else()
    set(LINUX_ARCH "x86_64")
  endif()
  set_target_properties(reaper_reamix PROPERTIES
    SUFFIX ".so"
    PREFIX ""
    OUTPUT_NAME "reaper_reamix-${LINUX_ARCH}"
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN"
  )
  # Look for libonnxruntime.so in same directory as the plugin
  set_target_properties(reaper_reamix PROPERTIES
    LINK_FLAGS "-Wl,-rpath,$ORIGIN"
  )
  target_link_libraries(reaper_reamix PRIVATE dl pthread curl)

  # SWELL mod stub for Linux (REAPER provides SWELL implementation)
  target_sources(reaper_reamix PRIVATE
    ${WDL_DIR}/swell/swell-modstub-generic.cpp
  )
endif()

# ADR-097 sesja 107 — symbol verification gate DROPPED. The earlier
# tests/symbol_verification.sh post-build hook enforced zero AdvancedWeightsPanel
# / DevCalibrationStorage symbols in production dylib (REAMIX_DEV_CALIBRATION=OFF).
# After ADR-097 (sesja 107) the panel ships in every build, so the gate is
# obsolete.

# --- Install ---

if(APPLE)
  set(REAPER_PLUGINS_DIR "$ENV{HOME}/Library/Application Support/REAPER/UserPlugins")
elseif(WIN32)
  set(REAPER_PLUGINS_DIR "$ENV{APPDATA}/REAPER/UserPlugins")
else()
  set(REAPER_PLUGINS_DIR "$ENV{HOME}/.config/REAPER/UserPlugins")
endif()

install(TARGETS reaper_reamix DESTINATION "${REAPER_PLUGINS_DIR}")

# --- Tests ---

option(REAMIX_BUILD_TESTS "Build parity / unit tests" ON)

if(REAMIX_BUILD_TESTS)
  enable_testing()

  # test_mel — bitwise parity: reamix::MelSpectrogram vs REABeat reference impl.
  # No JUCE, no REAPER, no ORT linkage — just pocketfft + stdlib.
  add_executable(test_mel
    tests/parity/test_mel.cpp
    src/analysis/MelSpectrogram.cpp
    tests/parity/reference/MelSpectrogramReabeat.cpp
  )
  target_include_directories(test_mel PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_mel PRIVATE /W3)
  else()
    target_compile_options(test_mel PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME mel_parity
           COMMAND test_mel
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_stft — librosa.stft parity: reamix::dsp::STFT vs Python dump
  # (tools/dump_python_features.py). Loads y_audio.npy + stft_magnitude.npy
  # per track from references/golden/phase-2/dumps/. Pure stdlib + pocketfft.
  # Formal threshold 1e-3; realized target ≤ 1e-5.
  add_executable(test_stft
    tests/parity/test_stft.cpp
    src/dsp/STFT.cpp
  )
  target_include_directories(test_stft PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_stft PRIVATE /W3)
  else()
    target_compile_options(test_stft PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME stft_parity
           COMMAND test_stft
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_mel_librosa — librosa.feature.melspectrogram + power_to_db parity:
  # reamix::dsp::MelSpectrogramLibrosa vs Python dumps (mel_power.npy +
  # log_mel.npy). Pure stdlib + pocketfft; shares src/dsp/STFT.cpp with
  # test_stft. Formal threshold 1e-3; mel_power realized floor is ~peak×2^-22
  # (|STFT|² ULP doubled by squaring).
  add_executable(test_mel_librosa
    tests/parity/test_mel_librosa.cpp
    src/dsp/STFT.cpp
    src/dsp/MelSpectrogramLibrosa.cpp
  )
  target_include_directories(test_mel_librosa PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_mel_librosa PRIVATE /W3)
  else()
    target_compile_options(test_mel_librosa PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME mel_librosa_parity
           COMMAND test_mel_librosa
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_dct — scipy.fft.dct(type=2, norm='ortho') parity: reamix::dsp::DCT
  # as consumed by librosa.feature.mfcc. Loads log_mel.npy + mfcc_40.npy +
  # mfcc_20.npy per track, runs DCT(128, 40) + DCT(128, 20) and diffs against
  # the reference mfccs. Pure stdlib (no FFT needed — DCT is direct matmul).
  add_executable(test_dct
    tests/parity/test_dct.cpp
    src/dsp/DCT.cpp
  )
  target_include_directories(test_dct PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_dct PRIVATE /W3)
  else()
    target_compile_options(test_dct PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME dct_parity
           COMMAND test_dct
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_mfcc — librosa.feature.mfcc parity: reamix::analysis::Mfcc end-to-end
  # from y_audio.npy through MelSpectrogramLibrosa + DCT. Covers the wiring
  # that test_mel_librosa and test_dct do not (power_to_db ref=1.0 path,
  # DCT matrix size = n_mfcc). Shares src/dsp/STFT.cpp + MelSpectrogramLibrosa.cpp
  # + DCT.cpp with prior phase-2 parity targets; pure stdlib + pocketfft.
  add_executable(test_mfcc
    tests/parity/test_mfcc.cpp
    src/dsp/STFT.cpp
    src/dsp/MelSpectrogramLibrosa.cpp
    src/dsp/DCT.cpp
    src/analysis/Mfcc.cpp
  )
  target_include_directories(test_mfcc PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_mfcc PRIVATE /W3)
  else()
    target_compile_options(test_mfcc PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME mfcc_parity
           COMMAND test_mfcc
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_pitchtrack — librosa.estimate_tuning(S=|STFT|², bpo=12) parity:
  # reamix::dsp::PitchTrack vs Python dump tuning_est.npy. Loads
  # stft_magnitude.npy per track, casts to float32, squares in float32
  # (matches librosa's complex64 → f32 abs → f32 square internal path) and
  # runs estimateTuning. Pure stdlib (no FFT — PitchTrack is pitch + tuning
  # only, STFT comes from the python dump).
  # Formal threshold 1e-4; floor 0.01 (one tuning-resolution bin). ADR-010.
  add_executable(test_pitchtrack
    tests/parity/test_pitchtrack.cpp
    src/dsp/PitchTrack.cpp
  )
  target_include_directories(test_pitchtrack PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_pitchtrack PRIVATE /W3)
  else()
    target_compile_options(test_pitchtrack PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME pitchtrack_parity
           COMMAND test_pitchtrack
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_chroma_stft — librosa.feature.chroma_stft parity: reamix::dsp::ChromaSTFT
  # vs Python dump chroma_stft.npy. Loads stft_magnitude.npy + tuning_est.npy
  # + chroma_stft.npy per track. The Python tuning scalar is fed directly to
  # ChromaSTFT (not re-derived via PitchTrack here) so this test grades chroma
  # alone; the full y→...→PitchTrack→ChromaSTFT stack is graded later at the
  # FeatureExtractor orchestrator layer. Pure stdlib (no FFT — S comes from
  # the Python dump). Formal threshold 1e-3 (VALIDATION.md phase-2).
  add_executable(test_chroma_stft
    tests/parity/test_chroma_stft.cpp
    src/dsp/ChromaSTFT.cpp
  )
  target_include_directories(test_chroma_stft PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_chroma_stft PRIVATE /W3)
  else()
    target_compile_options(test_chroma_stft PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME chroma_stft_parity
           COMMAND test_chroma_stft
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_spectral_contrast — librosa.feature.spectral_contrast parity:
  # reamix::dsp::SpectralContrast vs Python dump spectral_contrast.npy. Loads
  # stft_magnitude.npy + spectral_contrast.npy per track and feeds |STFT|¹
  # amplitude (NOT power — distinct from ChromaSTFT and Mfcc) directly into
  # SpectralContrast::compute. Pure stdlib. Formal threshold 1e-3 (VALIDATION.md
  # phase-2).
  add_executable(test_spectral_contrast
    tests/parity/test_spectral_contrast.cpp
    src/dsp/SpectralContrast.cpp
  )
  target_include_directories(test_spectral_contrast PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_spectral_contrast PRIVATE /W3)
  else()
    target_compile_options(test_spectral_contrast PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME spectral_contrast_parity
           COMMAND test_spectral_contrast
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_onset_strength — librosa.onset.onset_strength parity:
  # reamix::dsp::OnsetStrength vs Python dump onset_strength.npy. Loads
  # mel_power.npy (cast f64 → f32) directly into OnsetStrength::compute,
  # isolating step-6 drift from MelSpectrogramLibrosa::power upstream.
  # Pure stdlib. Formal threshold 1e-3 (VALIDATION.md phase-2).
  add_executable(test_onset_strength
    tests/parity/test_onset_strength.cpp
    src/dsp/OnsetStrength.cpp
  )
  target_include_directories(test_onset_strength PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_onset_strength PRIVATE /W3)
  else()
    target_compile_options(test_onset_strength PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME onset_strength_parity
           COMMAND test_onset_strength
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_rms — librosa.feature.rms parity: reamix::dsp::RMS vs Python dump
  # rms.npy. Loads y_audio.npy directly (time-domain, no STFT); frame_length
  # 2048, hop 512, center=True pad_mode='constant' matches librosa defaults.
  # Pure stdlib. Formal threshold 1e-3 (VALIDATION.md phase-2).
  add_executable(test_rms
    tests/parity/test_rms.cpp
    src/dsp/RMS.cpp
  )
  target_include_directories(test_rms PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_rms PRIVATE /W3)
  else()
    target_compile_options(test_rms PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME rms_parity
           COMMAND test_rms
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_spectral_centroid — librosa.feature.spectral_centroid parity:
  # reamix::dsp::SpectralCentroid vs Python dump spectral_centroid.npy.
  # Loads stft_magnitude.npy (cast f64 → f32) directly into compute(),
  # isolating centroid drift from upstream STFT drift. Same input class as
  # test_spectral_contrast (amplitude, not power).
  # Pure stdlib. Formal threshold 1e-3 Hz (VALIDATION.md phase-2).
  add_executable(test_spectral_centroid
    tests/parity/test_spectral_centroid.cpp
    src/dsp/SpectralCentroid.cpp
  )
  target_include_directories(test_spectral_centroid PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_spectral_centroid PRIVATE /W3)
  else()
    target_compile_options(test_spectral_centroid PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME spectral_centroid_parity
           COMMAND test_spectral_centroid
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_beat_sync — librosa.util.sync(aggregate=mean, pad=True) parity:
  # reamix::analysis::BeatSync vs Python dump beat_sync_mel_power.npy.
  # Input: mel_power.npy [128 × T] + synthetic beat_frames.npy (stride-20
  # ints). Exercises fix_frames pad/dedupe + per-slice mean aggregation.
  # Pure stdlib. Formal threshold 1e-3 (VALIDATION.md phase-2).
  add_executable(test_beat_sync
    tests/parity/test_beat_sync.cpp
    src/analysis/BeatSync.cpp
  )
  target_include_directories(test_beat_sync PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_beat_sync PRIVATE /W3)
  else()
    target_compile_options(test_beat_sync PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME beat_sync_parity
           COMMAND test_beat_sync
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_beat_windows — step-9 per-beat window extractors parity:
  # reamix::analysis::BeatWindows vs feature_extractor.py L317-463 helpers
  # (_extract_edge_features / _extract_edge_rms / _extract_waveform_snippets).
  # Four sub-tests per track:
  #   edge_features (on mel_power cast to float32, n_mels=128 fixture),
  #   edge_rms      (on y + beat_times),
  #   boundary_waveforms   (pre=35 ms, post=120 ms),
  #   transition_waveforms (pre=280 ms, post=320 ms).
  # Pure stdlib. Formal threshold 1e-3 across all sub-tests (VALIDATION.md
  # phase-2).
  add_executable(test_beat_windows
    tests/parity/test_beat_windows.cpp
    src/analysis/BeatWindows.cpp
  )
  target_include_directories(test_beat_windows PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_beat_windows PRIVATE /W3)
  else()
    target_compile_options(test_beat_windows PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME beat_windows_parity
           COMMAND test_beat_windows
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_feature_extractor — step-10 orchestrator parity:
  # reamix::analysis::FeatureExtractor vs feature_extractor.py L189-305.
  # Four fixture groups in one binary:
  #   A. Standard mode × 10 goldens (synthetic beats).
  #   B. Fast mode (billie_jean, 39-dim, transitionWaveforms-empty).
  #   C. Real beats (billie_jean_real via librosa.beat.beat_track) —
  #      validates non-uniform beat spacing through BeatSync + BeatWindows.
  #   D. Unit: timeToFrames round-half-to-even (nearbyint vs round).
  # Links every phase-2 DSP module + BeatSync + BeatWindows + Mfcc.
  # Pure stdlib + pocketfft. Formal threshold 1e-3 per field.
  add_executable(test_feature_extractor
    tests/parity/test_feature_extractor.cpp
    src/analysis/FeatureExtractor.cpp
    src/analysis/BeatSync.cpp
    src/analysis/BeatWindows.cpp
    src/analysis/Mfcc.cpp
    src/analysis/VocalFeatures.cpp
    src/dsp/ChromaSTFT.cpp
    src/dsp/DCT.cpp
    src/dsp/HPSS.cpp
    src/dsp/MelSpectrogramLibrosa.cpp
    src/dsp/OnsetStrength.cpp
    src/dsp/PitchTrack.cpp
    src/dsp/RMS.cpp
    src/dsp/STFT.cpp
    src/dsp/SpectralCentroid.cpp
    src/dsp/SpectralContrast.cpp
    src/dsp/SpectralFlatness.cpp
  )
  target_include_directories(test_feature_extractor PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_feature_extractor PRIVATE /W3)
  else()
    target_compile_options(test_feature_extractor PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME feature_extractor_parity
           COMMAND test_feature_extractor
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_percentile — phase-2b unit: reamix::util::percentile bitwise parity
  # vs np.percentile(a, q, method='linear') on hand-crafted cases (numpy 1.26.4).
  # Header-only utility — no .cpp source. Pure stdlib (std::sort + std::span).
  add_executable(test_percentile
    tests/parity/test_percentile.cpp
  )
  target_include_directories(test_percentile PRIVATE src)
  if(MSVC)
    target_compile_options(test_percentile PRIVATE /W3)
  else()
    target_compile_options(test_percentile PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME percentile_parity
           COMMAND test_percentile
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_hpss — phase-2b step-3 parity: reamix::dsp::HPSS::harmonic vs
  # librosa.effects.hpss(y)[0] on 10-track phase-2 goldens. Loads y_audio.npy
  # + hpss_y_harmonic.npy per track (dumped in phase-2b session 2).
  # Formal threshold 1e-4 on y_harmonic (VALIDATION.md, phase-2b spec.md).
  add_executable(test_hpss
    tests/parity/test_hpss.cpp
    src/dsp/HPSS.cpp
    src/dsp/STFT.cpp
  )
  target_include_directories(test_hpss PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_hpss PRIVATE /W3)
  else()
    target_compile_options(test_hpss PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME hpss_parity
           COMMAND test_hpss
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_spectral_flatness — phase-2b step-4 parity: reamix::dsp::SpectralFlatness
  # vs librosa.feature.spectral_flatness(y=y_harmonic, ...). Loads Python
  # hpss_y_harmonic.npy (not C++ HPSS) to isolate flatness drift. Threshold 1e-6.
  add_executable(test_spectral_flatness
    tests/parity/test_spectral_flatness.cpp
    src/dsp/SpectralFlatness.cpp
  )
  target_include_directories(test_spectral_flatness PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_spectral_flatness PRIVATE /W3)
  else()
    target_compile_options(test_spectral_flatness PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME spectral_flatness_parity
           COMMAND test_spectral_flatness
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_vocal_features — phase-2b step-5 parity: reamix::analysis::VocalFeatures
  # vs Python vocal_features.py::_extract_vocal_features default-mode (ADR-014).
  # Feeds Python HPSS + STFT magnitude to isolate orchestrator drift from
  # upstream DSP drift. 8 output fields × 10 tracks. Threshold 1e-3 on
  # meaningful fields; bitwise 0.0 on ADR-014 zero-stubs.
  add_executable(test_vocal_features
    tests/parity/test_vocal_features.cpp
    src/analysis/VocalFeatures.cpp
    src/analysis/BeatSync.cpp
    src/dsp/RMS.cpp
    src/dsp/SpectralFlatness.cpp
  )
  target_include_directories(test_vocal_features PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_vocal_features PRIVATE /W3)
  else()
    target_compile_options(test_vocal_features PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME vocal_features_parity
           COMMAND test_vocal_features
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_novelty_features — phase-3 step-2 parity: reamix::analysis::NoveltyFeatures
  # (end-to-end y → 25-dim MFCC(13)+chroma(12) z-scored vstack) vs Python dump
  # `novelty_features_25d.npy`, which mirrors structure_analyzer.py:193-210
  # verbatim. Pure stdlib + pocketfft. ADR-019 gate: 1e-3 absolute OR 5e-7
  # ratio (ADR-011 escape precedent).
  add_executable(test_novelty_features
    tests/parity/test_novelty_features.cpp
    src/analysis/NoveltyFeatures.cpp
    src/analysis/Mfcc.cpp
    src/dsp/ChromaSTFT.cpp
    src/dsp/DCT.cpp
    src/dsp/MelSpectrogramLibrosa.cpp
    src/dsp/PitchTrack.cpp
    src/dsp/STFT.cpp
  )
  target_include_directories(test_novelty_features PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_novelty_features PRIVATE /W3)
  else()
    target_compile_options(test_novelty_features PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME novelty_features_parity
           COMMAND test_novelty_features
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_recurrence — phase-3 step-3 parity: reamix::analysis::Recurrence::build
  # vs recurrence.py::build_recurrence_matrix dumped as R_matrix.npy + 3
  # component aids (R_feat / R_chroma / R_homo). Consumes
  # recurrence_feature_matrix.npy (f64, L2-normalized per row, real beats from
  # librosa.beat.beat_track per ADR-018). Gate: L∞ ≤ 1e-3 on R. Components
  # loaded only for bisection print on failure. Pure stdlib — no pocketfft,
  # no sklearn (port uses brute-force kNN on n<500 per recurrence.py spec).
  add_executable(test_recurrence
    tests/parity/test_recurrence.cpp
    src/analysis/Recurrence.cpp
  )
  target_include_directories(test_recurrence PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_recurrence PRIVATE /W3)
  else()
    target_compile_options(test_recurrence PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME recurrence_parity
           COMMAND test_recurrence
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_cbm_segmenter — phase-3 step-4 (session 5) parity: reamix::analysis::
  # CBMSegmenter DP pipeline (computeBarFeatures + computeBarAutosimilarity +
  # cbmSegment) vs cbm_segmenter.py on the 10-track phase-2 corpus. Primary
  # gate: cbm_bar_segments exact integer match on 10/10. Intermediate diffs
  # (bar_features, bar_times, autosim) are printed as bisection aids, not
  # gated. Pure stdlib — no pocketfft, no librosa, no sklearn.
  add_executable(test_cbm_segmenter
    tests/parity/test_cbm_segmenter.cpp
    src/analysis/CBMSegmenter.cpp
  )
  target_include_directories(test_cbm_segmenter PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_cbm_segmenter PRIVATE /W3)
  else()
    target_compile_options(test_cbm_segmenter PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME cbm_segmenter_parity
           COMMAND test_cbm_segmenter
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_novelty_segmenter — phase-3 step-5 session 7 + 8 parity per ADR-021
  # and ADR-022:
  #   Session 7: SSM → Foote novelty → find_peaks → boundaries → segment
  #              embeddings (pre-clustering, strict bitwise gates).
  #   Session 8: clustering (normalized Laplacian + eigh via Accelerate LAPACK
  #              + k-means++ + Lloyd) + labeling (_create_segments) per
  #              ADR-022 Option A (preserve Python silhouette-bug observed
  #              behavior). Hungarian-matched cluster_id ≥ 90 %, exact
  #              label ≥ 90 %.
  # Consumes NoveltyFeatures output (session 2) on one fixture + golden SSM
  # on another. Links Apple Accelerate.framework for LAPACK `dsyevd`.
  add_executable(test_novelty_segmenter
    tests/parity/test_novelty_segmenter.cpp
    src/analysis/NoveltySegmenter.cpp
    src/analysis/NoveltyFeatures.cpp
    src/analysis/Mfcc.cpp
    src/dsp/ChromaSTFT.cpp
    src/dsp/DCT.cpp
    src/dsp/MelSpectrogramLibrosa.cpp
    src/dsp/PitchTrack.cpp
    src/dsp/STFT.cpp
  )
  target_include_directories(test_novelty_segmenter PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(APPLE)
    target_link_libraries(test_novelty_segmenter PRIVATE "-framework Accelerate")
  endif()
  if(MSVC)
    target_compile_options(test_novelty_segmenter PRIVATE /W3)
  else()
    target_compile_options(test_novelty_segmenter PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME novelty_segmenter_parity
           COMMAND test_novelty_segmenter
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_segment_consolidation — phase-3 step-7 session 9 parity per
  # session-9 scope swap (dispatcher was planned for step 6 but calls
  # consolidation internally → consolidation ported first). Strict
  # bitwise gate on all 5 Segment fields × 10 tracks × 2 fixtures
  # (CBM-path input + novelty-path input). Deterministic algorithm —
  # no RNG, no spectral decomposition — so strict bitwise is achievable.
  add_executable(test_segment_consolidation
    tests/parity/test_segment_consolidation.cpp
    src/analysis/SegmentConsolidation.cpp
  )
  target_include_directories(test_segment_consolidation PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_segment_consolidation PRIVATE /W3)
  else()
    target_compile_options(test_segment_consolidation PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME segment_consolidation_parity
           COMMAND test_segment_consolidation
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_structure_analyzer — phase-3 step-6 session 10 parity: full
  # StructureAnalyzer::analyze dispatcher per ADR-018 predicates
  # (fastMode → novelty; !fastMode && downbeats≥4 && beatFeatures &&
  # CBM≥5 → CBM; else → novelty fallback). Both branches run
  # SegmentConsolidation::consolidate. 10-track corpus splits 3 CBM /
  # 7 novelty. Gate per-track: CBM-dispatched strict bitwise on all 5
  # Segment fields + boundaries (session-6 precedent); novelty-dispatched
  # ADR-022 Option B — start/end bitwise, confidence L∞ ≤ 1e-6, cluster
  # Hungarian ≥ 70 %, label exact ≥ 70 %, boundaries bitwise (session-8
  # precedent). Links the full novelty pipeline (Mfcc/ChromaSTFT/STFT
  # /DCT/MelSpectrogramLibrosa/PitchTrack via NoveltyFeatures; Accelerate
  # LAPACK via NoveltySegmenter clustering).
  add_executable(test_structure_analyzer
    tests/parity/test_structure_analyzer.cpp
    src/analysis/StructureAnalyzer.cpp
    src/analysis/CBMSegmenter.cpp
    src/analysis/NoveltyFeatures.cpp
    src/analysis/NoveltySegmenter.cpp
    src/analysis/SegmentConsolidation.cpp
    src/analysis/Mfcc.cpp
    src/dsp/ChromaSTFT.cpp
    src/dsp/DCT.cpp
    src/dsp/MelSpectrogramLibrosa.cpp
    src/dsp/PitchTrack.cpp
    src/dsp/STFT.cpp
  )
  target_include_directories(test_structure_analyzer PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(APPLE)
    target_link_libraries(test_structure_analyzer PRIVATE "-framework Accelerate")
  endif()
  if(MSVC)
    target_compile_options(test_structure_analyzer PRIVATE /W3)
  else()
    target_compile_options(test_structure_analyzer PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME structure_analyzer_parity
           COMMAND test_structure_analyzer
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_repetition_map — phase-3 step-8 session 11 parity: full
  # RepetitionMap::build vs Python `build_repetition_map` on the 10-track
  # corpus. Consumes session-10 dispatched_* segments as input, computes
  # boundary_waveforms via BeatWindows::extractWaveformSnippets (phase-2
  # step-9 validated) on beats_cpp, calls Recurrence internally for
  # phase-2 internal-diagonal scan. Gate: per-track n_sections_scanned /
  # n_pairs_verified / section_pairs equal + ≥ 90 % agreement on
  # high-confidence jumps (waveform_similarity > 0.6) with field tolerance
  # (wf/chroma L∞ ≤ 1e-6, lag ±1); ADR-022 eminem_without_me waived.
  # Links the full Recurrence + BeatWindows dependency chain; no Accelerate
  # (no LAPACK path).
  add_executable(test_repetition_map
    tests/parity/test_repetition_map.cpp
    src/analysis/RepetitionMap.cpp
    src/analysis/Recurrence.cpp
    src/analysis/BeatWindows.cpp
    src/dsp/WaveformXcorr.cpp
  )
  target_include_directories(test_repetition_map PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_repetition_map PRIVATE /W3)
  else()
    target_compile_options(test_repetition_map PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME repetition_map_parity
           COMMAND test_repetition_map
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_median_filter — phase-2b unit: reamix::util::medianFilter2DReflect
  # bitwise parity vs scipy.ndimage.median_filter(mode='reflect', odd kernels).
  # Data dumped by `python tools/dump_median_filter_tests.py` under
  # tests/parity/reference/data/median_filter/. Header-only utility.
  add_executable(test_median_filter
    tests/parity/test_median_filter.cpp
  )
  target_include_directories(test_median_filter PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_median_filter PRIVATE /W3)
  else()
    target_compile_options(test_median_filter PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME median_filter_parity
           COMMAND test_median_filter
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_postprocessor — bitwise parity: reamix::Postprocessor vs REABeat
  # reference. Pure stdlib — no JUCE, no REAPER, no ORT, no model file.
  add_executable(test_postprocessor
    tests/parity/test_postprocessor.cpp
    src/analysis/Postprocessor.cpp
    tests/parity/reference/PostprocessorReabeat.cpp
  )
  target_include_directories(test_postprocessor PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_postprocessor PRIVATE /W3)
  else()
    target_compile_options(test_postprocessor PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME postprocessor_parity
           COMMAND test_postprocessor
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_beat_interpolator — bitwise parity: reamix::BeatInterpolator vs
  # REABeat reference. Pure stdlib — no JUCE, no REAPER, no ORT, no model.
  add_executable(test_beat_interpolator
    tests/parity/test_beat_interpolator.cpp
    src/analysis/BeatInterpolator.cpp
    tests/parity/reference/BeatInterpolatorReabeat.cpp
  )
  target_include_directories(test_beat_interpolator PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_beat_interpolator PRIVATE /W3)
  else()
    target_compile_options(test_beat_interpolator PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME beat_interpolator_parity
           COMMAND test_beat_interpolator
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_tempo_estimator — bitwise parity: reamix::TempoEstimator vs REABeat
  # reference. Pure stdlib + <cmath> — no JUCE, no REAPER, no ORT, no model.
  add_executable(test_tempo_estimator
    tests/parity/test_tempo_estimator.cpp
    src/analysis/TempoEstimator.cpp
    tests/parity/reference/TempoEstimatorReabeat.cpp
  )
  target_include_directories(test_tempo_estimator PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_tempo_estimator PRIVATE /W3)
  else()
    target_compile_options(test_tempo_estimator PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME tempo_estimator_parity
           COMMAND test_tempo_estimator
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_time_sig_detector — bitwise parity: reamix::TimeSigDetector vs REABeat
  # reference. Pure stdlib — no JUCE, no REAPER, no ORT, no model file.
  add_executable(test_time_sig_detector
    tests/parity/test_time_sig_detector.cpp
    src/analysis/TimeSigDetector.cpp
    tests/parity/reference/TimeSigDetectorReabeat.cpp
  )
  target_include_directories(test_time_sig_detector PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_time_sig_detector PRIVATE /W3)
  else()
    target_compile_options(test_time_sig_detector PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME time_sig_detector_parity
           COMMAND test_time_sig_detector
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_downbeat_cleaner — bitwise parity: reamix::DownbeatCleaner vs REABeat
  # reference. Pure stdlib — no JUCE, no REAPER, no ORT, no model file.
  add_executable(test_downbeat_cleaner
    tests/parity/test_downbeat_cleaner.cpp
    src/analysis/DownbeatCleaner.cpp
    tests/parity/reference/DownbeatCleanerReabeat.cpp
  )
  target_include_directories(test_downbeat_cleaner PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_downbeat_cleaner PRIVATE /W3)
  else()
    target_compile_options(test_downbeat_cleaner PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME downbeat_cleaner_parity
           COMMAND test_downbeat_cleaner
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_onset_refinement — bitwise parity: reamix::OnsetRefinement vs REABeat
  # reference. Uses pocketfft (same FFT path as MelSpectrogram) — no JUCE,
  # no REAPER, no ORT, no model file. Deterministic synthetic audio inputs.
  add_executable(test_onset_refinement
    tests/parity/test_onset_refinement.cpp
    src/analysis/OnsetRefinement.cpp
    tests/parity/reference/OnsetRefinementReabeat.cpp
  )
  target_include_directories(test_onset_refinement PRIVATE
    src
    ${POCKETFFT_DIR}
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_onset_refinement PRIVATE /W3)
  else()
    target_compile_options(test_onset_refinement PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME onset_refinement_parity
           COMMAND test_onset_refinement
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_inference — bitwise parity: reamix::InferenceProcessor vs REABeat
  # reference. Links ORT; requires the ONNX model at runtime (env var
  # REAMIX_TEST_ONNX_MODEL or argv[1]; falls back to ~/.reabeat/models/).
  # If the model is absent the test prints SKIPPED and exits 0.
  if(ORT_FOUND)
    add_executable(test_inference
      tests/parity/test_inference.cpp
      src/analysis/InferenceProcessor.cpp
      src/analysis/MelSpectrogram.cpp
      tests/parity/reference/InferenceProcessorReabeat.cpp
    )
    target_include_directories(test_inference PRIVATE
      src
      ${POCKETFFT_DIR}
      tests/parity/reference
    )
    target_link_libraries(test_inference PRIVATE onnxruntime)
    target_compile_definitions(test_inference PRIVATE
      REAMIX_HAS_ONNX=1
      REABEAT_HAS_ONNX=1
    )
    if(APPLE)
      # ORT dylib lives at vendor/onnxruntime/lib — rpath it for the test binary.
      set_target_properties(test_inference PROPERTIES
        BUILD_RPATH "${ORT_DIR}/lib"
      )
    elseif(UNIX)
      set_target_properties(test_inference PROPERTIES
        BUILD_RPATH "${ORT_DIR}/lib"
      )
    endif()
    if(MSVC)
      target_compile_options(test_inference PRIVATE /W3)
    else()
      target_compile_options(test_inference PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
    add_test(NAME inference_parity
             COMMAND test_inference
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endif()

  # test_audio_loader — bitwise parity on resample() vs REABeat reference +
  # semantic round-trip on loadMono() via juce::WavAudioFormat writer.
  # Links juce_audio_basics (LagrangeInterpolator) + juce_audio_formats
  # (WavAudioFormat + AudioFormatManager). No ORT, no REAPER, no model file.
  add_executable(test_audio_loader
    tests/parity/test_audio_loader.cpp
    src/io/AudioLoader.cpp
    tests/parity/reference/AudioLoaderReabeat.cpp
  )
  target_include_directories(test_audio_loader PRIVATE
    src
    tests/parity/reference
  )
  target_link_libraries(test_audio_loader PRIVATE
    juce::juce_audio_basics
    juce::juce_audio_formats
  )
  target_compile_definitions(test_audio_loader PRIVATE
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  )
  if(MSVC)
    target_compile_options(test_audio_loader PRIVATE /W3)
  else()
    target_compile_options(test_audio_loader PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME audio_loader_parity
           COMMAND test_audio_loader
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_model_manager — SHA-256 plumbing (NIST vectors) + cache-path shape
  # (modelDir() under userApplicationDataDirectory). No network, no model
  # file, no REABeat parity (ADR-006 deliberately diverges cache paths).
  add_executable(test_model_manager
    tests/parity/test_model_manager.cpp
    src/analysis/ModelManager.cpp
  )
  target_include_directories(test_model_manager PRIVATE
    src
  )
  target_link_libraries(test_model_manager PRIVATE
    juce::juce_core
    juce::juce_cryptography
  )
  target_compile_definitions(test_model_manager PRIVATE
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  )
  if(MSVC)
    target_compile_options(test_model_manager PRIVATE /W3)
  else()
    target_compile_options(test_model_manager PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME model_manager_parity
           COMMAND test_model_manager
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_beat_detector — end-to-end bitwise parity: reamix::BeatDetector
  # vs REABeat reference. Shared Ort::Session with deterministic opts;
  # synthetic 120 BPM click train @ 22050 Hz (no resampling). Links ORT
  # and juce_audio_basics (for LagrangeInterpolator in resampleTo22050).
  # REAMIX_WITH_REAPER_IO is NOT defined here — detectFile is plugin-only.
  if(ORT_FOUND)
    add_executable(test_beat_detector
      tests/parity/test_beat_detector.cpp
      src/analysis/BeatDetector.cpp
      src/analysis/MelSpectrogram.cpp
      src/analysis/InferenceProcessor.cpp
      src/analysis/Postprocessor.cpp
      src/analysis/BeatInterpolator.cpp
      src/analysis/OnsetRefinement.cpp
      src/analysis/TempoEstimator.cpp
      src/analysis/TimeSigDetector.cpp
      src/analysis/DownbeatCleaner.cpp
      tests/parity/reference/BeatDetectorReabeat.cpp
      tests/parity/reference/MelSpectrogramReabeat.cpp
      tests/parity/reference/InferenceProcessorReabeat.cpp
      tests/parity/reference/PostprocessorReabeat.cpp
      tests/parity/reference/BeatInterpolatorReabeat.cpp
      tests/parity/reference/OnsetRefinementReabeat.cpp
      tests/parity/reference/TempoEstimatorReabeat.cpp
      tests/parity/reference/TimeSigDetectorReabeat.cpp
      tests/parity/reference/DownbeatCleanerReabeat.cpp
    )
    target_include_directories(test_beat_detector PRIVATE
      src
      ${POCKETFFT_DIR}
      tests/parity/reference
    )
    target_link_libraries(test_beat_detector PRIVATE
      onnxruntime
      juce::juce_audio_basics
    )
    target_compile_definitions(test_beat_detector PRIVATE
      REAMIX_HAS_ONNX=1
      REABEAT_HAS_ONNX=1
      JUCE_STANDALONE_APPLICATION=0
      JUCE_MODAL_LOOPS_PERMITTED=0
      JUCE_WEB_BROWSER=0
      JUCE_USE_CURL=0
      JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
    )
    if(APPLE)
      set_target_properties(test_beat_detector PROPERTIES
        BUILD_RPATH "${ORT_DIR}/lib"
      )
    elseif(UNIX)
      set_target_properties(test_beat_detector PROPERTIES
        BUILD_RPATH "${ORT_DIR}/lib"
      )
    endif()
    if(MSVC)
      target_compile_options(test_beat_detector PRIVATE /W3)
    else()
      target_compile_options(test_beat_detector PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
    add_test(NAME beat_detector_parity
             COMMAND test_beat_detector
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  endif()

  # test_quality — phase-4 step-2 session 17 parity: bitwise parity of
  # reamix::remix::{computeQualityScore, computeVocalPenalty,
  # computeOnsetPenalty, qualityLabel, qualityColorKey} vs
  # references/python-source/remix/quality.py. Pure scalar arithmetic on
  # fixed-order weighted sums; any non-bitwise result flags a transcription
  # or accumulation-order bug. 210 quality-score cases (10 edge + 200 random
  # PCG64 seed=42), 110 vocal cases, 57 onset cases, + label / color-key
  # boundary spot checks. Data dumped by tools/dump_quality_tests.py.
  #
  # `-ffp-contract=off` is required. Without it, Clang contracts `a*b + c`
  # into a single FMA instruction (one rounding) while CPython performs the
  # multiply and the add as two separate IEEE-754 operations (two roundings).
  # That 1-ULP drift (~2.22e-16 on values near 1.0) shows up in ~20 % of the
  # weighted-sum cases unless contraction is disabled. MSVC is bitwise-safe
  # by default (no FP contraction unless /fp:fast).
  add_executable(test_quality
    tests/parity/test_quality.cpp
    src/remix/Quality.cpp
  )
  target_include_directories(test_quality PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_quality PRIVATE /W3)
  else()
    target_compile_options(test_quality PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME quality_parity
           COMMAND test_quality
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_sosfilt — phase-4 session 18 parity: scipy.signal.sosfilt vs
  # reamix::dsp::SosFilt (transposed direct form II biquad cascade) on a
  # Butterworth 4th-order bandpass [250, 3400] Hz at fs=44100. The SOS
  # coefficients and reference outputs are dumped by
  # tools/dump_phase4_tests.py via scipy.signal.butter + sosfilt on the
  # boundary-waveform batch for the smoke track.
  #
  # `-ffp-contract=off` per ADR-028: the SOS kernel is `b0*x + s1 ...`,
  # exactly the scalar-MAC pattern FMA contracts on Clang. Consistent with
  # test_quality flag discipline.
  add_executable(test_sosfilt
    tests/parity/test_sosfilt.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_sosfilt PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_sosfilt PRIVATE /W3)
  else()
    target_compile_options(test_sosfilt PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME sosfilt_parity
           COMMAND test_sosfilt
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_butterworth — phase-5 session 30 parity: Butterworth SOS design
  # (scipy.signal.butter bit-exact ≤1e-12 per-element) + zero-phase
  # forward-backward wrapper (scipy.signal.sosfiltfilt round-trip ≤1e-9 on
  # 3 synthetic signals × 3 band configs × 2 sample rates = 18 cases).
  # See ADR-031 session 29 + meta/VALIDATION.md phase-5 row.
  #
  # `-ffp-contract=off` per ADR-028 — the SOS cascade + forward-backward
  # composition amplifies FMA-contraction drift by 2× vs test_sosfilt.
  add_executable(test_butterworth
    tests/parity/test_butterworth.cpp
    src/dsp/Butterworth.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_butterworth PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_butterworth PRIVATE /W3)
  else()
    target_compile_options(test_butterworth PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME butterworth_parity
           COMMAND test_butterworth
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_phase_align — phase-5 session 31 parity:
  # reamix::render::PhaseAlign vs references/python-source/remix/crossfade.py::_phase_align
  # Time-domain brute-force NCC (crossfade.py:86-153) per ADR-031 SB-4.
  # 7 cases: mono shift±k @ sr=44100 / zero-shift passthrough / stereo
  # broadcast / norm-guard trigger / n<32 early return / noise API-default
  # max_shift=64. Two gates: shift-index bit-exact (int) + aligned-output
  # max_abs ≤ 1e-12 (f64 ULP).
  #
  # `-ffp-contract=off` per ADR-028 (13th reuse) — dot product + L2 norm
  # accumulators are the same pure-scalar class as SosFilt (ADR-028 §Scope).
  add_executable(test_phase_align
    tests/parity/test_phase_align.cpp
    src/render/PhaseAlign.cpp
  )
  target_include_directories(test_phase_align PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_phase_align PRIVATE /W3)
  else()
    target_compile_options(test_phase_align PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME phase_align_parity
           COMMAND test_phase_align
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_crossfade — phase-5 session 32 parity:
  # reamix::render::Crossfade::adaptiveCrossfade vs Python-monkey-patched
  # adaptive_crossfade_fixed (ADR-032 Option B.myfix). Composite module —
  # composes Butterworth sosfiltfilt + PhaseAlign NCC + equal-power fades +
  # per-band RMS compensation (±9 dB clamp 0.35..2.83).
  # 4 cases: billie_jean PA=T/F control pair + synthetic bass+treble PA=T
  # EC=F + synthetic RMS-mismatch clamp path. Gate: per-sample max_abs ≤ 1e-9.
  #
  # `-ffp-contract=off` per ADR-028 (14th reuse) — per-band sum + NCC dot +
  # equal-power cos/sin + RMS scalar gain are all pure-scalar accumulator
  # classes covered by ADR-028's FMA-contraction reasoning.
  add_executable(test_crossfade
    tests/parity/test_crossfade.cpp
    src/render/Crossfade.cpp
    src/render/PhaseAlign.cpp
    src/dsp/Butterworth.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_crossfade PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_crossfade PRIVATE /W3)
  else()
    target_compile_options(test_crossfade PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME crossfade_parity
           COMMAND test_crossfade
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_splice — phase-5 session 34 parity:
  # reamix::render::Splice (5 core primitives) vs Python SpliceMixin
  # (references/python-source/remix/splice.py L26-196). Kinds:
  #   find_onset, window_onset    — int bit-exact gate
  #   similarity, score_splice    — f64 max_abs ≤ 1e-9
  # 11 cases: 3 find_onset (default / short-chunk / explicit ms),
  # 2 window_onset (default / short n<16), 3 similarity (identical / BJ real /
  # anti-correlated), 3 score_splice (BJ real / reject branch / no-shift branch).
  #
  # `-ffp-contract=off` per ADR-028 (15th reuse) — NCC dot / norm accumulators,
  # box-convolve multiply-and-accumulate, zero-mean mean-subtract, 20*log10 RMS
  # ratio: all pure-scalar accumulator class covered by ADR-028 reasoning.
  add_executable(test_splice
    tests/parity/test_splice.cpp
    src/render/Splice.cpp
  )
  target_include_directories(test_splice PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_splice PRIVATE /W3)
  else()
    target_compile_options(test_splice PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME splice_parity
           COMMAND test_splice
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_renderer — phase-5 session 36 parity:
  # reamix::render::Renderer (build_edit_plan + _render_edit_plan + render)
  # vs Python RemixRenderer (references/python-source/remix/renderer.py
  # L70-475). In-session smoke seed (3 cases) per HANDOVER session 36; full
  # 48-case corpus per ADR-034 is the actual phase-5 close gate.
  #
  # Cases: billie_jean r=0.5 (canonical) + alice_in_chains_nutshell r=0.3
  # (fallback path_len=12 n_trans=0) + meshuggah_bleed r=0.7 (polyrhythmic).
  # Gate: per-sample audio max_abs ≤ 5e-4 (phase-5 spec L27 starting), edit
  # plan int-bit-exact + sec-fields ≤ 1e-9, transition_times ≤ 1e-9.
  #
  # Sources composed: Renderer + Splice (5 composite methods) + Crossfade
  # (adaptive) + PhaseAlign + Butterworth + SosFilt — full phase-5 stack.
  #
  # `-ffp-contract=off` per ADR-028 (16th reuse) — Renderer composes scalar
  # accumulator modules + per-sample fade-multiply + np.cos/np.sin equal-
  # power ramps; same arithmetic class as the rest of phase-5.
  add_executable(test_renderer
    tests/parity/test_renderer.cpp
    src/render/Renderer.cpp
    src/render/Splice.cpp
    src/render/Crossfade.cpp
    src/render/PhaseAlign.cpp
    src/dsp/Butterworth.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_renderer PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_renderer PRIVATE /W3)
  else()
    target_compile_options(test_renderer PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME renderer_parity
           COMMAND test_renderer
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_transition_cost — phase-4 session 18 parity:
  # reamix::remix::computeTransitionCosts vs Python
  # references/python-source/remix/transition_cost.py::compute_transition_costs
  # on billie_jean (session-18 smoke) + future 48-case corpus (session 19+).
  #
  # Per-signal sub-gates (ADR-026): waveform, chroma, successor, context,
  # label, section, bar_align, energy, edge_energy, centroid — each with
  # its own ULP budget set empirically. Aggregate total_cost L∞ ≤ 0.02.
  # Boolean hard-gate flags must be bitwise equal (hard_gate_energy,
  # hard_gate_chroma, hard_gate_quality).
  #
  # `-ffp-contract=off` per ADR-028 (weighted sums via computeQualityScore).
  add_executable(test_transition_cost
    tests/parity/test_transition_cost.cpp
    src/remix/TransitionCost.cpp
    src/remix/SegmentData.cpp
    src/remix/Quality.cpp
    src/dsp/SosFilt.cpp
    src/dsp/WaveformXcorr.cpp
  )
  target_include_directories(test_transition_cost PRIVATE
    src
    tests/parity/reference
    ${POCKETFFT_DIR}
  )
  if(MSVC)
    target_compile_options(test_transition_cost PRIVATE /W3)
  else()
    target_compile_options(test_transition_cost PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME transition_cost_parity
           COMMAND test_transition_cost
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_viterbi_dp — phase-4 session 20 parity:
  # reamix::remix::{computeDownbeatArrays, buildNeighbors, viterbiDP}
  # vs Python references/python-source/remix/viterbi_dp.py::{
  #   _compute_downbeat_arrays, _build_neighbors, _viterbi_dp}.
  #
  # Per-function sub-gates (HANDOVER-19): downbeat arrays + int-set
  # semantics (bitwise), SegmentData re-use validation, neighbor CSR
  # (int64 bitwise both indices + offsets), DP path (exact int64 match)
  # + total_cost (f64 ULP ≤ 5e-4).
  #
  # Session-20 scope: billie_jean × 0.5 smoke. Session-21 extends to
  # 16 tracks × 3 durations = 48 cases per phase-4 acceptance criterion (a).
  #
  # `-ffp-contract=off` per ADR-028 (3rd reuse after test_quality +
  # test_transition_cost). DP kernel accumulates `dp[t,i] + W[i,j]` +
  # weighted penalty sums — same FMA vulnerability class.
  add_executable(test_viterbi_dp
    tests/parity/test_viterbi_dp.cpp
    src/remix/ViterbiDP.cpp
    src/remix/SegmentData.cpp
  )
  target_include_directories(test_viterbi_dp PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_viterbi_dp PRIVATE /W3)
  else()
    target_compile_options(test_viterbi_dp PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME viterbi_dp_parity
           COMMAND test_viterbi_dp
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_optimizer — phase-4 session 22 parity:
  # reamix::remix::CleanOptimizer vs Python optimizer.py::CleanOptimizer.
  # FIRST end-to-end facade test — includes the label-aware intro/outro
  # + post-DP outro extension + transition_metadata assembly chain that
  # session-20/21 ViterbiDP parity did NOT cover (viterbi dump used
  # simplified compute_dp_params with default intro=4/outro=2).
  # `-ffp-contract=off` per ADR-028 — 4th reuse after Quality / TransitionCost
  # / ViterbiDP. The facade accumulates DP penalty chains + post-DP
  # extension arithmetic + metadata copies; same FMA-vulnerability class.
  add_executable(test_optimizer
    tests/parity/test_optimizer.cpp
    src/remix/Optimizer.cpp
    src/remix/ViterbiDP.cpp
    src/remix/SegmentData.cpp
  )
  target_include_directories(test_optimizer PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_optimizer PRIVATE /W3)
  else()
    target_compile_options(test_optimizer PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME optimizer_parity
           COMMAND test_optimizer
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_optimizer_e2e — phase-4 session 26 ADR-027 integration parity:
  # same source, same bitwise + per-signal gates, but compiled against the
  # C++-BeatDetector grid goldens under tests/parity/reference/data/
  # phase4_e2e/ instead of the librosa-grid goldens under phase4/.
  # Session 26 corpus narrowed to billie_jean for smoke (session 27 expands
  # to full 48 cases). `-ffp-contract=off` per ADR-028 (9th reuse after
  # Quality / TransitionCost / ViterbiDP / Optimizer / RegionOptimizer /
  # RegionCost / Prescreen / BlockAssembly — composes identical weighted-
  # sum chain, inherits FMA vulnerability class).
  add_executable(test_optimizer_e2e
    tests/parity/test_optimizer.cpp
    src/remix/Optimizer.cpp
    src/remix/ViterbiDP.cpp
    src/remix/SegmentData.cpp
  )
  target_include_directories(test_optimizer_e2e PRIVATE
    src
    tests/parity/reference
  )
  target_compile_definitions(test_optimizer_e2e PRIVATE
    PHASE4_DATA_SUBDIR="phase4_e2e"
    PHASE4_E2E_MODE=1
  )
  if(MSVC)
    target_compile_options(test_optimizer_e2e PRIVATE /W3)
  else()
    target_compile_options(test_optimizer_e2e PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME optimizer_e2e_parity
           COMMAND test_optimizer_e2e
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_optimizer_variation — phase-6 session 58 / ADR-048 / DEV-027 parity:
  # reamix::remix::CleanOptimizer::remix_k_best + remix_variation vs Python
  # optimizer.py::remix_k_best (`_remix.py:140-147` server-side semantic for
  # remix_variation). 3 tracks × 2 ratios × 4 variations = 24 lookups.
  # Full 16×4=64 corpus gate deferred to phase-6 close per ADR-034.
  # `-ffp-contract=off` per ADR-028 — 5th reuse on the optimizer chain.
  add_executable(test_optimizer_variation
    tests/parity/test_optimizer_variation.cpp
    src/remix/Optimizer.cpp
    src/remix/ViterbiDP.cpp
    src/remix/SegmentData.cpp
  )
  target_include_directories(test_optimizer_variation PRIVATE
    src
    tests/parity/reference
  )
  if(MSVC)
    target_compile_options(test_optimizer_variation PRIVATE /W3)
  else()
    target_compile_options(test_optimizer_variation PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME optimizer_variation_parity
           COMMAND test_optimizer_variation
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_region_optimizer — phase-4 session 23 parity:
  # reamix::remix::RegionOptimizer + reamix::remix::computeRegionCosts vs
  # Python region_optimizer.py::RegionRemixMixin + region_cost.py.
  # Companion to test_optimizer — same FMA-vulnerability class + same
  # per-signal sub-gate classes for the embedded TransitionCandidate
  # metadata. Corpus 16 × 3 = 48 cases (ratios 0.5, 0.8, 1.2 — exercises
  # is_extending branch at region_optimizer.py:157-158 for 1.2).
  # `-ffp-contract=off` per ADR-028 — 5th reuse after Quality / TransitionCost
  # / ViterbiDP / Optimizer.
  add_executable(test_region_optimizer
    tests/parity/test_region_optimizer.cpp
    src/remix/RegionOptimizer.cpp
    src/remix/RegionCost.cpp
    src/remix/TransitionCost.cpp
    src/remix/Quality.cpp
    src/remix/SegmentData.cpp
    src/dsp/WaveformXcorr.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_region_optimizer PRIVATE
    src
    tests/parity/reference
    ${POCKETFFT_DIR}
  )
  if(MSVC)
    target_compile_options(test_region_optimizer PRIVATE /W3)
  else()
    target_compile_options(test_region_optimizer PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME region_optimizer_parity
           COMMAND test_region_optimizer
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_transition_prescreen — phase-4 session 24 parity:
  # reamix::remix::prescreenTransitions vs
  # Python transition_prescreen.py::prescreen_transitions.
  # `-ffp-contract=off` per ADR-028 — 6th reuse after Quality / TransitionCost
  # / ViterbiDP / Optimizer / RegionOptimizer. Recurrence matrix build +
  # diagonal mean_sim accumulation + WaveformXcorr composition all hit
  # FMA-vulnerability class.
  add_executable(test_transition_prescreen
    tests/parity/test_transition_prescreen.cpp
    src/remix/TransitionPrescreen.cpp
    src/analysis/Recurrence.cpp
    src/dsp/WaveformXcorr.cpp
  )
  target_include_directories(test_transition_prescreen PRIVATE
    src
    tests/parity/reference
    ${POCKETFFT_DIR}
  )
  if(MSVC)
    target_compile_options(test_transition_prescreen PRIVATE /W3)
  else()
    target_compile_options(test_transition_prescreen PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME transition_prescreen_parity
           COMMAND test_transition_prescreen
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_block_assembly — phase-4 session 24 parity:
  # reamix::remix::computeBlockCompatibility + assembleBlocks + buildBlocks
  # vs Python block_assembly.py::{build_blocks, compute_block_compatibility,
  # assemble_blocks}. 3rd consumer of EDGE_ENERGY_SATURATION_DB (session-24
  # ADR-029 consolidation — Hard Rule #1 single-source-of-truth promotion).
  # `-ffp-contract=off` per ADR-028 — 7th reuse.
  add_executable(test_block_assembly
    tests/parity/test_block_assembly.cpp
    src/remix/BlockAssembly.cpp
    src/remix/TransitionCost.cpp
    src/remix/Quality.cpp
    src/remix/SegmentData.cpp
    src/dsp/WaveformXcorr.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_block_assembly PRIVATE
    src
    tests/parity/reference
    ${POCKETFFT_DIR}
  )
  if(MSVC)
    target_compile_options(test_block_assembly PRIVATE /W3)
  else()
    target_compile_options(test_block_assembly PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME block_assembly_parity
           COMMAND test_block_assembly
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_transient_continuity — sesja 75 (DEV-028 ADR-064) self-validation.
  # Validates `computeOnsetNorm` + per-pair transient_continuity composition
  # against hand-computed expected values + mathematical invariants. NO Python
  # reference (per memory feedback_python_no_longer_source_of_truth.md — the
  # cost component is C++-canonical, fix-in-port extension).
  add_executable(test_transient_continuity
    tests/parity/test_transient_continuity.cpp
    src/remix/Quality.cpp
  )
  target_include_directories(test_transient_continuity PRIVATE
    src
  )
  if(MSVC)
    target_compile_options(test_transient_continuity PRIVATE /W3)
  else()
    target_compile_options(test_transient_continuity PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME transient_continuity_parity
           COMMAND test_transient_continuity
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_mfcc_continuity — sesja 77 (DEV-028 ADR-066) self-validation.
  # Validates `computeMfccContinuityMatrix` against hand-computed expected
  # values + mathematical invariants. NO Python reference (C++-canonical
  # extension per ADR-066, mirroring ADR-064 pattern).
  add_executable(test_mfcc_continuity
    tests/parity/test_mfcc_continuity.cpp
    src/remix/Quality.cpp
  )
  target_include_directories(test_mfcc_continuity PRIVATE
    src
  )
  if(MSVC)
    target_compile_options(test_mfcc_continuity PRIVATE /W3)
  else()
    target_compile_options(test_mfcc_continuity PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME mfcc_continuity_parity
           COMMAND test_mfcc_continuity
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_sequential_continuity — sesja 81 (DEV-028 ADR-068 D4) self-validation.
  # Validates the sequential_continuity collapse branch in computeQualityScore
  # against hand-computed mean values. NO Python reference (C++-canonical
  # collapse formula per sesja-80 D1 correlation evidence + ADR-068 D4).
  add_executable(test_sequential_continuity
    tests/parity/test_sequential_continuity.cpp
    src/remix/Quality.cpp
  )
  target_include_directories(test_sequential_continuity PRIVATE
    src
  )
  if(MSVC)
    target_compile_options(test_sequential_continuity PRIVATE /W3)
  else()
    target_compile_options(test_sequential_continuity PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME sequential_continuity_parity
           COMMAND test_sequential_continuity
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_harmonic_vs_timbre_blend — sesja 92 (ADR-080 RESCOPE + ADR-083) self-validation.
  # Validates Tone slider blend logic in computeQualityScore + chroma similarity
  # matrix kernel against hand-computed expected values + algebraic invariants.
  # NO Python reference (C++-canonical extension per ADR-080 RESCOPE STATUS UPDATE 1).
  add_executable(test_harmonic_vs_timbre_blend
    tests/parity/test_harmonic_vs_timbre_blend.cpp
    src/remix/Quality.cpp
  )
  target_include_directories(test_harmonic_vs_timbre_blend PRIVATE
    src
  )
  if(MSVC)
    target_compile_options(test_harmonic_vs_timbre_blend PRIVATE /W3)
  else()
    target_compile_options(test_harmonic_vs_timbre_blend PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME harmonic_vs_timbre_blend_parity
           COMMAND test_harmonic_vs_timbre_blend
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_edit_length — sesja 92 (ADR-083) self-validation.
  # Validates Edit Length per-edge global penalty wiring across ViterbiDP /
  # RegionOptimizer / CleanOptimizer Inputs structs + center-zero map algebra
  # + DP integration (penalty applied only on non-sequential edges). NO Python
  # reference (C++-canonical extension per ADR-083 Option α).
  add_executable(test_edit_length
    tests/parity/test_edit_length.cpp
    src/remix/ViterbiDP.cpp
    src/remix/SegmentData.cpp
  )
  target_include_directories(test_edit_length PRIVATE
    src
  )
  if(MSVC)
    target_compile_options(test_edit_length PRIVATE /W3)
  else()
    target_compile_options(test_edit_length PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME edit_length_parity
           COMMAND test_edit_length
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_region_inner_loop — sesja 94 (ADR-081 STATUS UPDATE 1) self-validation.
  # Validates Region β-model "inner-loop synthesizer" cost-function rebalance
  # (cap 1.0→5.0, backward penalty 0.5→0.05, jump base 0.8→0.3 in beta path).
  # Default region_beta=false preserves bit-exact baseline → existing
  # region_optimizer_parity 48-case corpus PASSES unchanged. Per ADR-065 +
  # memory `feedback_python_no_longer_source_of_truth.md`: hand-computed
  # invariants, no Python ground truth (extension is C++-canonical).
  add_executable(test_region_inner_loop
    tests/parity/test_region_inner_loop.cpp
    src/remix/RegionOptimizer.cpp
    src/remix/SegmentData.cpp
    src/remix/Quality.cpp
    src/remix/TransitionCost.cpp
    src/dsp/WaveformXcorr.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_region_inner_loop PRIVATE
    src
    ${POCKETFFT_DIR}
  )
  if(MSVC)
    target_compile_options(test_region_inner_loop PRIVATE /W3)
  else()
    target_compile_options(test_region_inner_loop PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME region_inner_loop_parity
           COMMAND test_region_inner_loop
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_region_optimizer_k_best — sesja 100 (DEV-032, ADR-091) self-validation.
  # Validates RegionOptimizer K-best variations (Region mirror of ADR-048
  # Duration K-best per CleanOptimizer::remix_k_best). Exposes new public
  # methods remix_k_best + remix_variation + blocked_transitions param on
  # remix(). Default nullptr param preserves bit-exact baseline → existing
  # region_optimizer_parity 48-case corpus PASSES unchanged. Per ADR-065:
  # hand-computed invariants, no Python ground truth (extension is C++-
  # canonical; Region mode never had K-best in Python).
  add_executable(test_region_optimizer_k_best
    tests/parity/test_region_optimizer_k_best.cpp
    src/remix/RegionOptimizer.cpp
    src/remix/SegmentData.cpp
    src/remix/Quality.cpp
    src/remix/TransitionCost.cpp
    src/dsp/WaveformXcorr.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_region_optimizer_k_best PRIVATE
    src
    ${POCKETFFT_DIR}
  )
  if(MSVC)
    target_compile_options(test_region_optimizer_k_best PRIVATE /W3)
  else()
    target_compile_options(test_region_optimizer_k_best PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME region_optimizer_k_best
           COMMAND test_region_optimizer_k_best
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_block_assembly_beta — sesja 96 (ADR-081 STATUS UPDATE 4) self-validation.
  # Validates Block Assembly β-model candidate-space expansion at junctions
  # (DEV-040 § A) + outside-window unclamp in assembleBlocks (DEV-043).
  # Default block_assembly_beta=false preserves bit-exact baseline → existing
  # block_assembly_parity 16-case corpus PASSES unchanged. Per ADR-065 +
  # memory `feedback_python_no_longer_source_of_truth.md`: hand-computed
  # invariants, no Python ground truth (extension is C++-canonical).
  add_executable(test_block_assembly_beta
    tests/parity/test_block_assembly_beta.cpp
    src/remix/BlockAssembly.cpp
    src/remix/Quality.cpp
    src/remix/TransitionCost.cpp
    src/remix/SegmentData.cpp
    src/dsp/WaveformXcorr.cpp
    src/dsp/SosFilt.cpp
  )
  target_include_directories(test_block_assembly_beta PRIVATE
    src
    ${POCKETFFT_DIR}
  )
  if(MSVC)
    target_compile_options(test_block_assembly_beta PRIVATE /W3)
  else()
    target_compile_options(test_block_assembly_beta PRIVATE
      -Wall -Wextra -Wno-unused-parameter
      -ffp-contract=off
    )
  endif()
  add_test(NAME block_assembly_beta_parity
           COMMAND test_block_assembly_beta
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_dev_calibration_persistence — sesja 98 (ADR-087 STATUS UPDATE 1)
  # self-validation. Sesja 106 (ADR-098): perceptual mapping tests removed
  # + schema bumped v1→v2 + testLegacyV1RecordLoads added. Sesja 107
  # (ADR-097): relocated to src/ui/ namespace reamix::ui; compiled in every
  # build (gate dropped). Coverage: JSONL schema round-trip + legacy v1
  # forward-compat + hashQualityWeights atDefault sentinel + storage
  # append/load + Block Assembly β-default helper.
  add_executable(test_dev_calibration_persistence
    tests/parity/test_dev_calibration_persistence.cpp
    src/ui/DevCalibrationStorage.cpp
    src/remix/Quality.cpp
  )
  target_include_directories(test_dev_calibration_persistence PRIVATE
    src
    ${POCKETFFT_DIR}
  )
  target_link_libraries(test_dev_calibration_persistence PRIVATE
    juce::juce_core
  )
  target_compile_definitions(test_dev_calibration_persistence PRIVATE
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  )
  if(MSVC)
    target_compile_options(test_dev_calibration_persistence PRIVATE /W3)
  else()
    target_compile_options(test_dev_calibration_persistence PRIVATE
      -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME dev_calibration_persistence
           COMMAND test_dev_calibration_persistence
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    # test_vocal_continuity — sesja 98 (ADR-088) self-validation.
    # Validates the vocal phrase continuity cost component formula:
    #   q = 1 - max(edge_vocal_release_end[i], edge_vocal_onset_start[j])
    # Hand-computed invariants per ADR-065. Default weight 0.0 preserves
    # bit-exact baseline.
    add_executable(test_vocal_continuity
      tests/parity/test_vocal_continuity.cpp
      src/remix/Quality.cpp
    )
    target_include_directories(test_vocal_continuity PRIVATE
      src
      ${POCKETFFT_DIR}
    )
    target_link_libraries(test_vocal_continuity PRIVATE
      juce::juce_core
    )
    target_compile_definitions(test_vocal_continuity PRIVATE
      JUCE_STANDALONE_APPLICATION=0
      JUCE_MODAL_LOOPS_PERMITTED=0
      JUCE_WEB_BROWSER=0
      JUCE_USE_CURL=0
      JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
    )
    if(MSVC)
      target_compile_options(test_vocal_continuity PRIVATE /W3)
    else()
      target_compile_options(test_vocal_continuity PRIVATE
        -Wall -Wextra -Wno-unused-parameter
        -ffp-contract=off)
    endif()
    add_test(NAME vocal_continuity_parity
             COMMAND test_vocal_continuity
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # test_parse_weights_field_coverage — sesja 86 ADR-076 CI guardrail.
  # Verifies tools/calibration_harness_parse_weights.h reads every JSONL
  # field that orchestrator.py emits + maps it onto the matching
  # QualityWeights struct member. Catches the sesja-85 class of bug:
  # struct field added but parseWeights block forgotten → silent zero-weight
  # axis → bugged calibration. Header-only header + struct, no Quality.cpp
  # dep + no DSP/ONNX dep — pure JSON round-trip.
  add_executable(test_parse_weights_field_coverage
    tests/parity/test_parse_weights_field_coverage.cpp
  )
  target_include_directories(test_parse_weights_field_coverage PRIVATE
    src
    tools
  )
  target_link_libraries(test_parse_weights_field_coverage PRIVATE
    juce::juce_core
  )
  target_compile_definitions(test_parse_weights_field_coverage PRIVATE
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  )
  if(MSVC)
    target_compile_options(test_parse_weights_field_coverage PRIVATE /W3)
  else()
    target_compile_options(test_parse_weights_field_coverage PRIVATE
      -Wall -Wextra -Wno-unused-parameter
    )
  endif()
  add_test(NAME parse_weights_field_coverage
           COMMAND test_parse_weights_field_coverage
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(parse_weights_field_coverage PROPERTIES TIMEOUT 30)

  # test_custom_kind_registry — sesja 100c (ADR-092 DEV-054) self-validation.
  # Validates CustomKindRegistry CRUD + JSON round-trip + ID format invariants
  # + malformed-input rejection + cascade-detection shape. Hand-computed
  # invariants per ADR-065. No DSP / ONNX deps.
  add_executable(test_custom_kind_registry
    tests/parity/test_custom_kind_registry.cpp
    src/ui/CustomKindRegistry.cpp
  )
  target_include_directories(test_custom_kind_registry PRIVATE src)
  target_link_libraries(test_custom_kind_registry PRIVATE
    juce::juce_core
    juce::juce_graphics
  )
  target_compile_definitions(test_custom_kind_registry PRIVATE
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  )
  if(MSVC)
    target_compile_options(test_custom_kind_registry PRIVATE /W3)
  else()
    target_compile_options(test_custom_kind_registry PRIVATE
      -Wall -Wextra -Wno-unused-parameter)
  endif()
  add_test(NAME custom_kind_registry
           COMMAND test_custom_kind_registry
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
  set_tests_properties(custom_kind_registry PROPERTIES TIMEOUT 30)

  # listener_ui_slot_randomization — sesja 88 ADR-078 CI guardrail.
  # Verifies tools/dev/calibration/listener_ui.py per-pair coin-flip slot
  # mapping prevents anchor (highest_mean from EUBO) from sticking in
  # slot A across the entire session. Sesja 88 audit precedent: Drake
  # 14-D N=77 had r01024 in slot A 41/41 times pre-fix. Sesja 85 cross-
  # track Δ matrix had same structural bug. Test runs 1000 forced pairs
  # against patched session.next_pair_from_pool returning fixed
  # (anchor, candidate); asserts each ID appears in slot A 50% ±10%.
  # Python test (no C++ harness needed; logic lives in Python listener_ui).
  find_program(PY_VENV_CALIBRATION
               NAMES python
               PATHS ${CMAKE_SOURCE_DIR}/tools/.venv-calibration/bin
               NO_DEFAULT_PATH)
  if(PY_VENV_CALIBRATION)
    add_test(NAME listener_ui_slot_randomization
             COMMAND ${PY_VENV_CALIBRATION}
                     ${CMAKE_SOURCE_DIR}/tests/parity/test_listener_ui_slot_randomization.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(listener_ui_slot_randomization PROPERTIES TIMEOUT 60)
  endif()
endif()  # REAMIX_BUILD_TESTS

# --- Tooling binaries (CLI runners for parity gates, not shipped with plugin) ---

# beat_detector_cli — phase-1 closing gate driver. Loads an audio file,
# runs reamix::BeatDetector, serializes DetectionResult to JSON. Paired
# with tools/closing_gate.py which runs Python beat_detector.py on the
# same files and diffs the results. Built only when ORT is available.
if(ORT_FOUND)
  add_executable(beat_detector_cli
    tools/beat_detector_cli.cpp
    src/analysis/BeatDetector.cpp
    src/analysis/MelSpectrogram.cpp
    src/analysis/InferenceProcessor.cpp
    src/analysis/Postprocessor.cpp
    src/analysis/BeatInterpolator.cpp
    src/analysis/OnsetRefinement.cpp
    src/analysis/TempoEstimator.cpp
    src/analysis/TimeSigDetector.cpp
    src/analysis/DownbeatCleaner.cpp
    src/io/AudioLoader.cpp
  )
  target_include_directories(beat_detector_cli PRIVATE
    src
    ${POCKETFFT_DIR}
  )
  target_link_libraries(beat_detector_cli PRIVATE
    onnxruntime
    juce::juce_audio_basics
    juce::juce_audio_formats
  )
  target_compile_definitions(beat_detector_cli PRIVATE
    REAMIX_HAS_ONNX=1
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=0
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  )
  if(APPLE)
    set_target_properties(beat_detector_cli PROPERTIES
      BUILD_RPATH "${ORT_DIR}/lib"
    )
  elseif(UNIX)
    set_target_properties(beat_detector_cli PROPERTIES
      BUILD_RPATH "${ORT_DIR}/lib"
    )
  endif()
  if(MSVC)
    target_compile_options(beat_detector_cli PRIVATE /W3)
  else()
    target_compile_options(beat_detector_cli PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endif()

# warm_cache — DEV-028 Stage-1 prerequisite. Drives AnalyzePipeline end-to-end
# headlessly to populate AnalysisDiskCache for the calibration corpus. Built
# only when ORT is available because BeatDetector requires ONNX. Source set
# mirrors what the plugin's analyse path actually needs (analysis + dsp + io +
# remix-cost-side + ui/AnalyzePipeline + ui/AnalysisDiskCache + Theme), so we
# avoid pulling MainComponent / panels / ReaperBridge into a CLI tool.
if(ORT_FOUND)
  file(GLOB WARM_CACHE_ANALYSIS_SOURCES CONFIGURE_DEPENDS "src/analysis/*.cpp")
  file(GLOB WARM_CACHE_DSP_SOURCES      CONFIGURE_DEPENDS "src/dsp/*.cpp")
  add_executable(warm_cache
    tools/warm_cache.cpp
    ${WARM_CACHE_ANALYSIS_SOURCES}
    ${WARM_CACHE_DSP_SOURCES}
    src/io/AudioLoader.cpp
    src/remix/Quality.cpp
    src/remix/SegmentData.cpp
    src/remix/TransitionCost.cpp
    src/ui/AnalyzePipeline.cpp
    src/ui/AnalysisDiskCache.cpp
    src/ui/Theme.cpp
  )
  target_include_directories(warm_cache PRIVATE
    src
    ${POCKETFFT_DIR}
  )
  target_link_libraries(warm_cache PRIVATE
    onnxruntime
    juce::juce_audio_formats
    juce::juce_cryptography
    juce::juce_events
    juce::juce_gui_basics
  )
  target_compile_definitions(warm_cache PRIVATE
    REAMIX_HAS_ONNX=1
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
    JUCE_DISPLAY_SPLASH_SCREEN=0
  )
  if(APPLE OR UNIX)
    set_target_properties(warm_cache PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")
  endif()
  if(MSVC)
    target_compile_options(warm_cache PRIVATE /W3)
  else()
    target_compile_options(warm_cache PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endif()

# calibration_harness — DEV-028 Stage-1 driver. Reads JSONL batch + a
# previously-warmed AnalysisDiskCache, runs RemixPipeline (or rebuilds TC for
# Duration mode) per request, emits WAV + per-splice CSV. Source set extends
# warm_cache's by adding render/* + remix/{RegionCost,RegionOptimizer,
# Optimizer,ViterbiDP,TransitionPrescreen,BlockAssembly} + ui/RemixPipeline.
# Built only when ORT is available because the bundle deserialisation path
# transitively pulls BeatDetector headers.
if(ORT_FOUND)
  file(GLOB CAL_HARNESS_ANALYSIS_SOURCES CONFIGURE_DEPENDS "src/analysis/*.cpp")
  file(GLOB CAL_HARNESS_DSP_SOURCES      CONFIGURE_DEPENDS "src/dsp/*.cpp")
  file(GLOB CAL_HARNESS_REMIX_SOURCES    CONFIGURE_DEPENDS "src/remix/*.cpp")
  file(GLOB CAL_HARNESS_RENDER_SOURCES   CONFIGURE_DEPENDS "src/render/*.cpp")
  add_executable(calibration_harness
    tools/calibration_harness.cpp
    ${CAL_HARNESS_ANALYSIS_SOURCES}
    ${CAL_HARNESS_DSP_SOURCES}
    ${CAL_HARNESS_REMIX_SOURCES}
    ${CAL_HARNESS_RENDER_SOURCES}
    src/io/AudioLoader.cpp
    src/ui/AnalyzePipeline.cpp
    src/ui/AnalysisDiskCache.cpp
    src/ui/RemixPipeline.cpp
    src/ui/Theme.cpp
  )
  target_include_directories(calibration_harness PRIVATE
    src
    ${POCKETFFT_DIR}
  )
  target_link_libraries(calibration_harness PRIVATE
    onnxruntime
    juce::juce_audio_formats
    juce::juce_cryptography
    juce::juce_events
    juce::juce_gui_basics
  )
  target_compile_definitions(calibration_harness PRIVATE
    REAMIX_HAS_ONNX=1
    JUCE_STANDALONE_APPLICATION=0
    JUCE_MODAL_LOOPS_PERMITTED=1
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_GLOBAL_MODULE_SETTINGS_INCLUDED=1
  )
  if(APPLE OR UNIX)
    set_target_properties(calibration_harness PROPERTIES BUILD_RPATH "${ORT_DIR}/lib")
  endif()
  if(MSVC)
    target_compile_options(calibration_harness PRIVATE /W3)
  else()
    target_compile_options(calibration_harness PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endif()

# Install ONNX Runtime shared library next to the plugin for @loader_path/$ORIGIN rpath resolution.
# macOS: the dylib's install_name is @rpath/libonnxruntime.<VERSION>.dylib — copy the
# versioned file (the un-versioned libonnxruntime.dylib is a duplicate, not a symlink,
# in the prebuilt ORT tarball, so loader resolution needs the versioned name).
if(ORT_FOUND)
  if(APPLE)
    file(GLOB ORT_DYLIBS "${ORT_DIR}/lib/libonnxruntime.*.dylib")
    install(FILES ${ORT_DYLIBS} DESTINATION "${REAPER_PLUGINS_DIR}")
  elseif(WIN32)
    install(FILES "${ORT_DIR}/lib/onnxruntime.dll" DESTINATION "${REAPER_PLUGINS_DIR}")
  else()
    install(FILES "${ORT_DIR}/lib/libonnxruntime.so" DESTINATION "${REAPER_PLUGINS_DIR}")
  endif()
endif()
