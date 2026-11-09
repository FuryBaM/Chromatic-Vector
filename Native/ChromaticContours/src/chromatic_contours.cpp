#define CHROMATIC_CONTOURS_EXPORTS
#include "chromatic_contours.h"
#include <cstddef>
#include <limits>

static_assert(sizeof(CvEdge) == 20, "CvEdge ABI must be tightly packed");

extern "C" std::uint32_t cv_abi_version() { return 2; }

extern "C" std::uint64_t cv_extract_shared_edges(
    const std::uint32_t* ids,
    const std::uint32_t width,
    const std::uint32_t height,
    CvEdge* out,
    const std::uint64_t capacity)
{
    if (ids == nullptr || width == 0 || height == 0 ||
        width >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        height >= static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()))
        return 0;
    std::uint64_t count = 0;
    auto emit = [&](std::uint32_t region, int x0, int y0, int x1, int y1) {
        if (out != nullptr && count < capacity)
            out[count] = CvEdge{region, x0, y0, x1, y1};
        ++count;
    };
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::size_t row = static_cast<std::size_t>(y) * width;
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t i = row + x;
            const std::uint32_t id = ids[i];
            if (x+1 < width && id != ids[i+1]) {
                emit(id, x+1, y, x+1, y+1);
                emit(ids[i+1], x+1, y+1, x+1, y);
            }
            if (y+1 < height && id != ids[i+width]) {
                emit(id, x+1, y+1, x, y+1);
                emit(ids[i+width], x, y+1, x+1, y+1);
            }
            if (x == 0) emit(id, 0, y+1, 0, y);
            if (x+1 == width) emit(id, x+1, y, x+1, y+1);
            if (y == 0) emit(id, x, 0, x+1, 0);
            if (y+1 == height) emit(id, x+1, y+1, x, y+1);
        }
    }
    return count;
}

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
    float distance(CvPoint a, CvPoint b) {
        return std::hypot(a.x-b.x, a.y-b.y);
    }
    float distance_to_line(CvPoint a, CvPoint b, CvPoint p) {
        const float dx=b.x-a.x,dy=b.y-a.y;
        const float denom=std::hypot(dx,dy);
        return denom < 1.0e-6f ? distance(a,p) :
            std::abs((p.x-a.x)*dy-(p.y-a.y)*dx)/denom;
    }
    float calculate_turn(CvPoint a, CvPoint b, CvPoint c) {
        const float dx1=b.x-a.x, dy1=b.y-a.y;
        const float dx2=c.x-b.x, dy2=c.y-b.y;
        const float len1=std::hypot(dx1,dy1),len2=std::hypot(dx2,dy2);
        if (len1 < 1.0e-6f || len2 < 1.0e-6f) return 0.0f;
        const float dot=std::clamp((dx1*dx2+dy1*dy2)/(len1*len2),-1.0f,1.0f);
        return 1.0f-dot;
    }
    CvPoint relax_point(CvPoint a, CvPoint b, CvPoint c,
                       float turn, float strength, float sharp_threshold) {
        if (turn >= sharp_threshold || strength <= 0.0f) return b;
        const float len_ab=distance(a,b), len_bc=distance(b,c);
        if (len_ab+len_bc < 1.0e-6f) return b;
        const float fraction=len_ab/(len_ab+len_bc);
        const CvPoint target={a.x+(c.x-a.x)*fraction,a.y+(c.y-a.y)*fraction};
        const float sharpness=std::clamp(turn/std::max(sharp_threshold,1.0e-6f),0.0f,1.0f);
        const float k=strength*(1.0f-sharpness);
        return {b.x+(target.x-b.x)*k, b.y+(target.y-b.y)*k};
    }
}

extern "C" std::uint64_t cv_adaptive_filter_polyline(
    const CvPoint* points, std::uint64_t count, std::uint8_t closed,
    float drop_tolerance, float smooth_strength, float sharp_turn_threshold,
    CvPoint* out, std::uint64_t capacity)
{
    if (points == nullptr || count == 0 ||
        !std::isfinite(drop_tolerance) || !std::isfinite(smooth_strength) ||
        !std::isfinite(sharp_turn_threshold) || drop_tolerance < 0 ||
        smooth_strength < 0 || smooth_strength > 1 || sharp_turn_threshold <= 0)
        return 0;
    std::vector<CvPoint> reduced;
    reduced.reserve(static_cast<std::size_t>(count));
    if (count <= 2) {
        reduced.insert(reduced.end(), points, points+count);
    } else {
        CvPoint anchor=points[0];
        reduced.push_back(anchor);
        const std::uint64_t last=closed ? count : count-1;
        for (std::uint64_t i=1;i<last;++i) {
            const CvPoint current=points[i], next=points[(i+1)%count];
            const float turn=calculate_turn(anchor,current,next);
            const float deviation=distance_to_line(anchor,next,current);
            if (turn < sharp_turn_threshold && deviation <= drop_tolerance)
                continue;
            anchor=relax_point(anchor,current,next,turn,smooth_strength,sharp_turn_threshold);
            reduced.push_back(anchor);
        }
        if (!closed) reduced.push_back(points[count-1]);
    }
    if (out != nullptr && capacity >= reduced.size())
        std::copy(reduced.begin(),reduced.end(),out);
    return reduced.size();
}
