"""Validated illustrative configuration; legacy edge_protection stays unmodified."""
from dataclasses import asdict, dataclass
import math

POSTER_GEOMETRY = dict(simplify=1.65, smooth=.95, max_error=1.7)

PRESETS = {
    'draft': dict(base_colors=10, target_shapes=140, detail_budget=35, boundary_preservation=.82,
                  residual_threshold=.15, max_shape_levels=3, primitive_fit='auto', refine_passes=1),
    'detail': dict(base_colors=16, target_shapes=250, detail_budget=100, boundary_preservation=.90,
                   residual_threshold=.08, max_shape_levels=3, primitive_fit='auto', refine_passes=2),
    'poster': dict(base_colors=14, target_shapes=165, detail_budget=44, boundary_preservation=.78,
                   residual_threshold=.095, max_shape_levels=3, primitive_fit='auto', refine_passes=2),
}

@dataclass(frozen=True)
class IllustrativeConfig:
    preset: str = 'detail'
    base_colors: int = 16
    target_shapes: int = 250
    detail_budget: int = 100
    boundary_preservation: float = .90
    residual_threshold: float = .08
    max_shape_levels: int = 3
    primitive_fit: str = 'auto'
    refine_passes: int = 2
    seed: int = 23
    contour_method: str = 'bounded'
    neural_model: str | None = None
    neural_optimize_steps: int = 5
    neural_merge: bool = False

    def __post_init__(self):
        from .unipolar import METHODS
        if self.contour_method not in (*METHODS,'neural'):raise ValueError('--contour-method must be '+', '.join((*METHODS,'neural')))
        if not 0<=self.neural_optimize_steps<=20:raise ValueError('--neural-optimize-steps must be 0..20')
        if self.neural_merge and self.contour_method!='neural':raise ValueError('--neural-merge requires --contour-method neural')
        if self.preset not in PRESETS:
            raise ValueError('--preset must be one of: '+', '.join(PRESETS))
        if not 2 <= self.base_colors <= 256:
            raise ValueError('--base-colors must be 2..256')
        if self.target_shapes < 0 or self.detail_budget < 0:
            raise ValueError('--target-shapes and --detail-budget must be >=0')
        for name in ('boundary_preservation','residual_threshold'):
            if not math.isfinite(getattr(self,name)) or not 0 <= getattr(self,name) <= 1:
                raise ValueError('--'+name.replace('_','-')+' must be finite and 0..1')
        if self.max_shape_levels not in (1,2,3) or not 0 <= self.refine_passes <= 5:
            raise ValueError('--max-shape-levels must be 1..3; --refine-passes must be 0..5')
        if self.primitive_fit not in ('auto','strict','off'):
            raise ValueError('--primitive-fit must be auto, strict or off')

    @classmethod
    def resolve(cls, preset='detail', **overrides):
        if preset not in PRESETS:
            raise ValueError('--preset must be one of: '+', '.join(PRESETS))
        values = {**PRESETS[preset], **{k:v for k,v in overrides.items() if v is not None}}
        return cls(preset=preset, **values)

    def to_dict(self):
        values=asdict(self)
        if values['neural_model'] is not None:values['neural_model']=str(values['neural_model'])
        return values
