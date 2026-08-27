from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class BoxSpecification(BaseModel):
    lengths: tuple[float, float, float] = (60.0, 60.0, 60.0)
    periodic: tuple[bool, bool, bool] = (True, True, True)

    @model_validator(mode="after")
    def positive_lengths(self):
        if any(value <= 0 for value in self.lengths):
            raise ValueError("All box lengths must be positive")
        return self


class PhaseSpecification(BaseModel):
    name: str
    structure: Literal["fcc", "bcc", "hcp", "custom"]
    lattice_parameter: float = Field(gt=0)
    c_over_a: float = Field(default=1.633, gt=0)
    template_path: str | None = None
    fraction: float = Field(gt=0, le=1)
    composition: dict[str, float]

    @model_validator(mode="after")
    def valid_composition(self):
        if self.structure == "custom" and not self.template_path:
            raise ValueError("A custom phase requires template_path")
        if not self.composition:
            raise ValueError("A phase composition cannot be empty")
        if any(value < 0 for value in self.composition.values()):
            raise ValueError("Element fractions cannot be negative")
        total = sum(self.composition.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Composition of phase {self.name!r} must sum to 1")
        return self


class CrystallographicOrientation(BaseModel):
    x_direction: tuple[int, int, int]
    z_direction: tuple[int, int, int]

    @model_validator(mode="after")
    def independent_axes(self):
        if not any(self.x_direction) or not any(self.z_direction):
            raise ValueError("Crystallographic directions cannot be zero")
        if abs(sum(a * b for a, b in zip(self.x_direction, self.z_direction))) > 0:
            raise ValueError("x_direction and z_direction must be perpendicular")
        return self


class GrainSpecification(BaseModel):
    number: int = Field(default=8, ge=1, le=500)
    method: Literal["voronoi"] = "voronoi"
    orientations: Literal["random", "identity", "explicit"] = "random"
    seeds: list[tuple[float, float, float]] | None = None
    orientation_matrices: list[list[list[float]]] | None = None
    euler_angles_deg: list[tuple[float, float, float]] | None = None
    crystallographic_orientations: list[CrystallographicOrientation] | None = None

    @model_validator(mode="after")
    def valid_seeds(self):
        if self.seeds is not None and len(self.seeds) != self.number:
            raise ValueError("The number of explicit seeds must match grains.number")
        supplied = (
            int(self.orientation_matrices is not None)
            + int(self.euler_angles_deg is not None)
            + int(self.crystallographic_orientations is not None)
        )
        if supplied > 1:
            raise ValueError("Specify orientation_matrices or euler_angles_deg, not both")
        if self.orientations == "explicit" and supplied != 1:
            raise ValueError("Explicit orientations require matrices or Euler angles")
        values = (
            self.orientation_matrices
            or self.euler_angles_deg
            or self.crystallographic_orientations
        )
        if values is not None and len(values) != self.number:
            raise ValueError("The number of explicit orientations must match grains.number")
        return self


class VacancySpecification(BaseModel):
    fraction: float = Field(gt=0, lt=1)
    phase: str | None = None


class InterstitialSpecification(BaseModel):
    element: str
    position: tuple[float, float, float]
    fractional: bool = True


class DefectSpecification(BaseModel):
    vacancies: list[VacancySpecification] = []
    interstitials: list[InterstitialSpecification] = []


class InclusionSpecification(BaseModel):
    name: str
    phase: str
    shape: Literal["sphere", "ellipsoid", "box", "cylinder", "slab"]
    center: tuple[float, float, float] = (0.5, 0.5, 0.5)
    size: tuple[float, float, float]
    fractional_center: bool = True
    periodic: bool = True
    orientation_matrix: list[list[float]] | None = None
    euler_angles_deg: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def valid_geometry(self):
        if any(value <= 0 for value in self.size):
            raise ValueError("Inclusion size values must be positive")
        if self.orientation_matrix is not None and self.euler_angles_deg is not None:
            raise ValueError("Specify one inclusion orientation representation")
        return self


class SegregationSpecification(BaseModel):
    phase: str
    region: Literal["grain_boundary", "phase_boundary"] = "grain_boundary"
    target_composition: dict[str, float]

    @model_validator(mode="after")
    def valid_target(self):
        if not self.target_composition or any(v < 0 for v in self.target_composition.values()):
            raise ValueError("Segregation target composition must be non-negative and non-empty")
        if abs(sum(self.target_composition.values()) - 1.0) > 1e-6:
            raise ValueError("Segregation target composition must sum to 1")
        return self


class RegionConstraintSpecification(BaseModel):
    """Chemical-disorder target applied to a spatial/microstructural region."""

    name: str
    phase: str | None = None
    grain_ids: list[int] | None = None
    region: Literal[
        "all", "interior", "grain_boundary", "phase_boundary", "triple_junction",
        "mixed_boundary",
    ] = "all"
    target_composition: dict[str, float] | None = None
    pair_weight: float = Field(default=1.0, ge=0)
    composition_weight: float = Field(default=10.0, ge=0)
    pair_scope: Literal["internal", "touching", "crossing"] = "internal"
    target_pair_factors: dict[int, dict[str, float]] = {}

    @model_validator(mode="after")
    def valid_constraint(self):
        if self.grain_ids is not None and (not self.grain_ids or min(self.grain_ids) < 0):
            raise ValueError("grain_ids must contain non-negative grain identifiers")
        if self.target_composition is not None:
            if not self.target_composition or any(value < 0 for value in self.target_composition.values()):
                raise ValueError("Region target composition must be non-negative and non-empty")
            if abs(sum(self.target_composition.values()) - 1.0) > 1e-6:
                raise ValueError("Region target composition must sum to 1")
        if self.pair_weight == 0 and self.composition_weight == 0:
            raise ValueError("A region constraint must have a non-zero objective weight")
        for shell, factors in self.target_pair_factors.items():
            if shell < 1 or any(value < 0 for value in factors.values()):
                raise ValueError("target_pair_factors require positive shell indices and non-negative values")
            if any("-" not in pair for pair in factors):
                raise ValueError("target_pair_factors keys must use the form Element-Element")
        return self


class SpatialSelectorSpecification(BaseModel):
    phase: str | None = None
    grain_ids: list[int] | None = None
    region: Literal[
        "all", "interior", "grain_boundary", "phase_boundary", "triple_junction",
        "mixed_boundary",
    ] = "all"


class DirectionalSROConstraintSpecification(BaseModel):
    """Directional source-to-target Warren-Cowley objective."""

    name: str
    source: SpatialSelectorSpecification
    target: SpatialSelectorSpecification
    shell: int = Field(default=1, ge=1, le=5)
    target_sro: dict[str, float]
    weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def valid_directional_target(self):
        if not self.target_sro or any("->" not in key for key in self.target_sro):
            raise ValueError("target_sro keys must use the form Source->Target")
        if any(value >= 1.0 or value < -10.0 for value in self.target_sro.values()):
            raise ValueError("Directional SRO targets must satisfy -10 <= alpha < 1")
        return self


class TripletConstraintSpecification(BaseModel):
    """Restricted center-neighbor-neighbor angular motif objective."""

    name: str
    selector: SpatialSelectorSpecification
    shell: int = Field(default=1, ge=1, le=2)
    angle_bins: list[float] = [60.0, 90.0, 120.0, 180.0]
    weight: float = Field(default=0.2, gt=0)

    @model_validator(mode="after")
    def valid_angles(self):
        if not self.angle_bins or self.angle_bins != sorted(set(self.angle_bins)):
            raise ValueError("angle_bins must be non-empty, unique, and increasing")
        if self.angle_bins[0] <= 0 or self.angle_bins[-1] > 180:
            raise ValueError("angle_bins must lie in (0, 180]")
        return self


class ChemistrySpecification(BaseModel):
    method: Literal["random", "native_sqs", "icet", "sqsgenerator"] = "random"
    neighbor_shells: int = Field(default=2, ge=1, le=5)
    steps: int = Field(default=50_000, ge=1)
    exclude_boundaries: bool = True
    random_temperature: float = Field(default=0.05, gt=0)
    history_interval: int = Field(default=100, ge=1)
    segregation: list[SegregationSpecification] = []
    region_constraints: list[RegionConstraintSpecification] = []
    directional_sro_constraints: list[DirectionalSROConstraintSpecification] = []
    triplet_constraints: list[TripletConstraintSpecification] = []
    adaptive_temperature: bool = True
    target_acceptance_rate: float = Field(default=0.35, gt=0.05, lt=0.95)
    early_stop_patience: int | None = Field(default=None, ge=100)
    objective_tolerance: float = Field(default=0.0, ge=0)
    restarts: int = Field(default=1, ge=1, le=64)
    require_exact_region_composition: bool = False
    exact_benchmark_max_sites: int | None = Field(default=None, ge=2, le=24)
    exact_benchmark_max_configurations: int = Field(default=1_000_000, ge=1, le=10_000_000)


class AnalysisSpecification(BaseModel):
    ovito_enabled: bool = True
    methods: list[Literal["cna", "ptm"]] = ["cna", "ptm"]
    ptm_rmsd_cutoff: float = Field(default=0.15, gt=0)
    rdf_enabled: bool = True
    rdf_cutoff: float = Field(default=8.0, gt=0)
    rdf_bins: int = Field(default=80, ge=10, le=1000)


class CleanupSpecification(BaseModel):
    minimum_distance_factor: float = Field(default=0.68, gt=0.3, lt=1.0)
    maximum_deleted_fraction: float = Field(default=0.15, ge=0, le=0.5)
    boundary_width: float = Field(default=3.5, gt=0)
    maximum_phase_fraction_deviation: float = Field(default=0.02, ge=0, lt=1)
    maximum_composition_deviation: float = Field(default=0.02, ge=0, lt=1)
    maximum_boundary_fraction: float = Field(default=0.40, ge=0, le=1)


class OutputSpecification(BaseModel):
    formats: list[Literal["extxyz", "lammps-data", "vasp", "cif"]] = [
        "extxyz",
        "lammps-data",
    ]


class ModelSpecification(BaseModel):
    name: str = "AlloyModelForge model"
    random_seed: int = 42
    box: BoxSpecification = BoxSpecification()
    phases: list[PhaseSpecification]
    grains: GrainSpecification = GrainSpecification()
    chemistry: ChemistrySpecification = ChemistrySpecification()
    analysis: AnalysisSpecification = AnalysisSpecification()
    cleanup: CleanupSpecification = CleanupSpecification()
    defects: DefectSpecification = DefectSpecification()
    inclusions: list[InclusionSpecification] = []
    output: OutputSpecification = OutputSpecification()

    @model_validator(mode="after")
    def phase_fractions_sum_to_one(self):
        if not self.phases:
            raise ValueError("At least one phase is required")
        total = sum(phase.fraction for phase in self.phases)
        if abs(total - 1.0) > 1e-6:
            raise ValueError("Phase fractions must sum to 1")
        if len({phase.name for phase in self.phases}) != len(self.phases):
            raise ValueError("Phase names must be unique")
        inclusion_phase_names = {inclusion.phase for inclusion in self.inclusions}
        unknown = inclusion_phase_names - {phase.name for phase in self.phases}
        if unknown:
            raise ValueError(f"Inclusions reference unknown phases: {', '.join(sorted(unknown))}")
        matrix_phases = [phase for phase in self.phases if phase.name not in inclusion_phase_names]
        if not matrix_phases:
            raise ValueError("At least one phase must remain assigned to matrix grains")
        if self.grains.number < len(matrix_phases):
            raise ValueError(
                "grains.number must be at least the number of matrix phases"
            )
        if len({inclusion.name for inclusion in self.inclusions}) != len(self.inclusions):
            raise ValueError("Inclusion names must be unique")
        phase_names = {phase.name for phase in self.phases}
        constraint_names = [constraint.name for constraint in self.chemistry.region_constraints]
        if len(set(constraint_names)) != len(constraint_names):
            raise ValueError("Region constraint names must be unique")
        for constraint in self.chemistry.region_constraints:
            if constraint.phase is not None and constraint.phase not in phase_names:
                raise ValueError(f"Region constraint references unknown phase: {constraint.phase}")
            if constraint.grain_ids and max(constraint.grain_ids) >= self.grains.number:
                raise ValueError("Region constraint grain_ids exceed grains.number")
            if constraint.target_composition is not None:
                if constraint.phase is None and len(self.phases) > 1:
                    raise ValueError(
                        f"Region constraint {constraint.name!r} with a target composition "
                        "must name a phase in a multiphase model"
                    )
                allowed = (
                    set(next(phase.composition for phase in self.phases if phase.name == constraint.phase))
                    if constraint.phase is not None
                    else set().union(*(phase.composition for phase in self.phases))
                )
                if not set(constraint.target_composition).issubset(allowed):
                    raise ValueError(
                        f"Region constraint {constraint.name!r} contains elements absent from the model"
                    )
        directional_names = [c.name for c in self.chemistry.directional_sro_constraints]
        triplet_names = [c.name for c in self.chemistry.triplet_constraints]
        all_objective_names = constraint_names + directional_names + triplet_names
        if len(set(all_objective_names)) != len(all_objective_names):
            raise ValueError("Native SQS constraint names must be unique")
        for constraint in self.chemistry.directional_sro_constraints:
            if constraint.source.phase is None or constraint.target.phase is None:
                raise ValueError("Directional SRO source and target must explicitly name phases")
            for selector in (constraint.source, constraint.target):
                if selector.phase is not None and selector.phase not in phase_names:
                    raise ValueError(f"Directional SRO references unknown phase: {selector.phase}")
                if selector.grain_ids and max(selector.grain_ids) >= self.grains.number:
                    raise ValueError("Directional SRO grain_ids exceed grains.number")
            allowed_source = (set(next(p.composition for p in self.phases
                                       if p.name == constraint.source.phase))
                              if constraint.source.phase else set().union(*(p.composition for p in self.phases)))
            allowed_target = (set(next(p.composition for p in self.phases
                                       if p.name == constraint.target.phase))
                              if constraint.target.phase else set().union(*(p.composition for p in self.phases)))
            for pair in constraint.target_sro:
                source_element, target_element = pair.split("->", 1)
                if source_element not in allowed_source or target_element not in allowed_target:
                    raise ValueError(f"Directional SRO pair {pair!r} is incompatible with its selectors")
        for constraint in self.chemistry.triplet_constraints:
            selector = constraint.selector
            if selector.phase is None and len(self.phases) > 1:
                raise ValueError("Triplet constraints must name a phase in a multiphase model")
            if selector.phase is not None and selector.phase not in phase_names:
                raise ValueError(f"Triplet constraint references unknown phase: {selector.phase}")
            if selector.grain_ids and max(selector.grain_ids) >= self.grains.number:
                raise ValueError("Triplet constraint grain_ids exceed grains.number")
        return self


def load_specification(path: str | Path) -> ModelSpecification:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ModelSpecification.model_validate(data)
