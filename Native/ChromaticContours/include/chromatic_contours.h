#pragma once
#include <cstdint>

#if defined(_WIN32)
  #if defined(CHROMATIC_CONTOURS_EXPORTS)
    #define CHROMATIC_API __declspec(dllexport)
  #else
    #define CHROMATIC_API __declspec(dllimport)
  #endif
#else
  #define CHROMATIC_API __attribute__((visibility("default")))
#endif

extern "C" {
    struct CvEdge {
        std::uint32_t region_id;
        std::int32_t x0;
        std::int32_t y0;
        std::int32_t x1;
        std::int32_t y1;
    };

    CHROMATIC_API std::uint32_t cv_abi_version();

    // For width*height row-major unsigned region IDs, emit both directed sides
    // of each shared boundary, plus outer canvas edges. The region interior
    // is always to the right of each directed edge (screen coordinates).
    // Returns required count; if out is null/too small, no writes occur.
    CHROMATIC_API std::uint64_t cv_extract_shared_edges(
        const std::uint32_t* region_ids,
        std::uint32_t width,
        std::uint32_t height,
        CvEdge* out,
        std::uint64_t capacity);
}

// Standalone adaptation of Unipolar-Gambit's polyline filter.
// Coordinates are input-image pixels; closed input omits a duplicated last point.
extern "C" {
    struct CvPoint { float x; float y; };
    CHROMATIC_API std::uint64_t cv_adaptive_filter_polyline(
        const CvPoint* points,
        std::uint64_t count,
        std::uint8_t closed,
        float drop_tolerance,
        float smooth_strength,
        float sharp_turn_threshold,
        CvPoint* out,
        std::uint64_t capacity);
}
