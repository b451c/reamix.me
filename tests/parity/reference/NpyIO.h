#pragma once

// Minimal reader for numpy .npy v1.0/v2.0 files, little-endian, C-order.
// Supports float32 ("<f4"), float64 ("<f8"), int64 ("<i8"). No pickle,
// no object arrays. Used only by phase-2 parity tests (loading dumps from
// tools/dump_python_features.py). Rejects anything it doesn't handle.

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace reamix::test {

struct NpyHeader {
    std::string dtype;               // e.g. "<f4", "<f8"
    bool fortran_order = false;
    std::vector<std::size_t> shape;
    std::size_t data_offset = 0;
};

inline NpyHeader parseNpyHeader(std::ifstream& in, const std::string& path)
{
    char magic[6];
    in.read(magic, 6);
    if (in.gcount() != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("Not an .npy file: " + path);

    std::uint8_t major = 0, minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    std::size_t headerLen = 0;
    if (major == 1) {
        std::uint16_t h = 0;
        in.read(reinterpret_cast<char*>(&h), 2);
        headerLen = h;
    } else if (major == 2 || major == 3) {
        std::uint32_t h = 0;
        in.read(reinterpret_cast<char*>(&h), 4);
        headerLen = h;
    } else {
        throw std::runtime_error("Unsupported .npy version: " + path);
    }

    std::string header(headerLen, '\0');
    in.read(header.data(), static_cast<std::streamsize>(headerLen));

    NpyHeader out;
    out.data_offset = static_cast<std::size_t>(in.tellg());

    // Very small ad-hoc parser for the header dict: { 'descr': '<f4',
    // 'fortran_order': False, 'shape': (N,) or (A, B), }
    auto findAfter = [&](const std::string& key) -> std::size_t {
        auto p = header.find(key);
        if (p == std::string::npos)
            throw std::runtime_error("Missing key '" + key + "' in .npy header: " + path);
        return p + key.size();
    };

    {
        auto p = findAfter("'descr'");
        p = header.find('\'', p) + 1;
        auto q = header.find('\'', p);
        out.dtype = header.substr(p, q - p);
    }
    {
        auto p = findAfter("'fortran_order'");
        p = header.find_first_of("TF", p);
        out.fortran_order = (header[p] == 'T');
    }
    {
        auto p = findAfter("'shape'");
        p = header.find('(', p) + 1;
        auto q = header.find(')', p);
        std::string dims = header.substr(p, q - p);
        std::size_t i = 0;
        while (i < dims.size()) {
            while (i < dims.size() && !std::isdigit(static_cast<unsigned char>(dims[i]))) ++i;
            if (i >= dims.size()) break;
            std::size_t j = i;
            while (j < dims.size() && std::isdigit(static_cast<unsigned char>(dims[j]))) ++j;
            out.shape.push_back(static_cast<std::size_t>(std::stoull(dims.substr(i, j - i))));
            i = j;
        }
    }

    if (out.fortran_order)
        throw std::runtime_error("Fortran-order .npy not supported: " + path);

    return out;
}

// Load a 1-D float32 array (e.g. y_audio.npy).
inline std::vector<float> loadNpy1DFloat32(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open .npy: " + path);
    auto hdr = parseNpyHeader(in, path);
    if (hdr.dtype != "<f4")
        throw std::runtime_error(path + ": expected dtype <f4, got " + hdr.dtype);
    if (hdr.shape.size() != 1)
        throw std::runtime_error(path + ": expected 1-D, got " + std::to_string(hdr.shape.size()) + "-D");

    std::vector<float> data(hdr.shape[0]);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

// Load a 1-D float64 array (e.g. tuning_est.npy, shape (1,) scalar wrapper).
inline std::vector<double> loadNpy1DFloat64(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open .npy: " + path);
    auto hdr = parseNpyHeader(in, path);
    if (hdr.dtype != "<f8")
        throw std::runtime_error(path + ": expected dtype <f8, got " + hdr.dtype);
    if (hdr.shape.size() != 1)
        throw std::runtime_error(path + ": expected 1-D, got " + std::to_string(hdr.shape.size()) + "-D");

    std::vector<double> data(hdr.shape[0]);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(double)));
    return data;
}

// Load a 1-D int64 array (e.g. beat_frames.npy).
inline std::vector<std::int64_t> loadNpy1DInt64(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open .npy: " + path);
    auto hdr = parseNpyHeader(in, path);
    if (hdr.dtype != "<i8")
        throw std::runtime_error(path + ": expected dtype <i8, got " + hdr.dtype);
    if (hdr.shape.size() != 1)
        throw std::runtime_error(path + ": expected 1-D, got " + std::to_string(hdr.shape.size()) + "-D");

    std::vector<std::int64_t> data(hdr.shape[0]);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(std::int64_t)));
    return data;
}

// Load a 2-D float64 array stored C-order (row-major). Returns rows×cols flat,
// plus the shape. Matrices from librosa (e.g. stft_magnitude of shape
// (bins, frames)) are saved as C-order arrays shape[0]=bins, shape[1]=frames.
struct NpyMatrixF64 {
    std::vector<double> data;
    std::size_t rows = 0;
    std::size_t cols = 0;
    double at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

inline NpyMatrixF64 loadNpy2DFloat64(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open .npy: " + path);
    auto hdr = parseNpyHeader(in, path);
    if (hdr.dtype != "<f8")
        throw std::runtime_error(path + ": expected dtype <f8, got " + hdr.dtype);
    if (hdr.shape.size() != 2)
        throw std::runtime_error(path + ": expected 2-D, got " + std::to_string(hdr.shape.size()) + "-D");

    NpyMatrixF64 out;
    out.rows = hdr.shape[0];
    out.cols = hdr.shape[1];
    out.data.resize(out.rows * out.cols);
    in.read(reinterpret_cast<char*>(out.data.data()),
            static_cast<std::streamsize>(out.data.size() * sizeof(double)));
    return out;
}

// Load a 2-D float32 array stored C-order (row-major). Symmetric with
// loadNpy2DFloat64 — used by step-9 tests for boundary_waveforms.npy and
// transition_waveforms.npy, which the Python reference emits as float32
// (Hann-windowed, RMS-normalized mono snippets).
struct NpyMatrixF32 {
    std::vector<float> data;
    std::size_t rows = 0;
    std::size_t cols = 0;
    float at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

inline NpyMatrixF32 loadNpy2DFloat32(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open .npy: " + path);
    auto hdr = parseNpyHeader(in, path);
    if (hdr.dtype != "<f4")
        throw std::runtime_error(path + ": expected dtype <f4, got " + hdr.dtype);
    if (hdr.shape.size() != 2)
        throw std::runtime_error(path + ": expected 2-D, got " + std::to_string(hdr.shape.size()) + "-D");

    NpyMatrixF32 out;
    out.rows = hdr.shape[0];
    out.cols = hdr.shape[1];
    out.data.resize(out.rows * out.cols);
    in.read(reinterpret_cast<char*>(out.data.data()),
            static_cast<std::streamsize>(out.data.size() * sizeof(float)));
    return out;
}

// Load a 2-D int64 array stored C-order (row-major). Used by phase-3 session-5
// CBMSegmenter parity test for cbm_bar_segments.npy (shape [n_segs, 2]).
struct NpyMatrixI64 {
    std::vector<std::int64_t> data;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::int64_t at(std::size_t r, std::size_t c) const { return data[r * cols + c]; }
};

inline NpyMatrixI64 loadNpy2DInt64(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Cannot open .npy: " + path);
    auto hdr = parseNpyHeader(in, path);
    if (hdr.dtype != "<i8")
        throw std::runtime_error(path + ": expected dtype <i8, got " + hdr.dtype);
    if (hdr.shape.size() != 2)
        throw std::runtime_error(path + ": expected 2-D, got " + std::to_string(hdr.shape.size()) + "-D");

    NpyMatrixI64 out;
    out.rows = hdr.shape[0];
    out.cols = hdr.shape[1];
    out.data.resize(out.rows * out.cols);
    in.read(reinterpret_cast<char*>(out.data.data()),
            static_cast<std::streamsize>(out.data.size() * sizeof(std::int64_t)));
    return out;
}

} // namespace reamix::test
