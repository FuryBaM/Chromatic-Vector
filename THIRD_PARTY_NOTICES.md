# Contour method attribution

The optional Unipolar contour adapter is derived from first-party algorithms in
FuryBaM/Unipolar-Gambit, commit `6a826d823555baf789d68fced3825d5a26037527`:

- `Assets/Modules/Game.Rendering/BorderBuilder.cs`: `AdaptiveFilterPolyline`,
  `RelaxPoint`, `SimplifyOpenPolyline`, `SimplifyClosedPolyline`, `SmoothPolyline`.
- `Native/ProvinceEdges/ug_province_edges.cpp`: the matching adaptive kernel.

Source: https://github.com/FuryBaM/Unipolar-Gambit/tree/6a826d823555baf789d68fced3825d5a26037527

The source repository's MIT license text is reproduced in
`licenses/Unipolar-Gambit-MIT.txt`. No game files, game assets, Unity runtime or
binaries are required. The game repository is read only during verification.

The source's currently active path uses adaptive filtering. RDP + Chaikin is an
older alternative retained there for visual comparison. Chromatic Vector adds
canonical SVG boundaries, bounded Bézier fitting, explicit junction protection,
alpha/correction locks and intersection fallback around those contour methods.
Thus this is an adaptation of verified methods, not a claim that a complete game
renderer or its distance-field shader has been ported into an SVG converter.
