"""Fixed research geometry policy; importable without scientific dependencies."""

VERSION = "vessel-branch-geometry/1.0.0"
POLICY = {
    "reference_disc_diameter": 128,
    "junction_merge_disc_diameters": 0.05,
    "spur_max_disc_diameters": 0.08,
    "root_max_disc_radii": 2.0,
    "root_tie_disc_diameters": 0.05,
    "sample_start_disc_diameters": 0.10,
    "sample_end_disc_diameters": 0.25,
    "direction_fit": "total_least_squares_13_uniform_arclength_samples",
    "roi_disc_radii": [1.0, None],
    "roi_order": "disc_circle_on_fixed_normalized_grid; full_mask_input",
    "canonical_orientation": "minimum_relative_disc_xy_then_sha256_of_four_rotations",
    "parent": "unique_bridge_towards_unique_disc_nearest_terminal",
    "angle": "acos(dot(daughter_unit_vectors)); native-grid fitted direction lines",
    "crossing": "four_arms_with_two_antiparallel_pairs_within_25_degrees",
    "cycles": "junction_on_cycle_rejected; independent_branches_retained",
    "maximum_pixels": 4194304,
    "scale_check_factors": [0.75, 1.5],
    "scale_angle_tolerance_degrees": 5.0,
    "scale_match_disc_diameters": 0.08,
    "scale_resampling_orientation": "disc_normalized_canonical_grid_before_resampling",
}
REASONS = {
    "crossing_candidate",
    "complex_four_arm_junction",
    "multiple_branches",
    "loop_ambiguous",
    "no_disc_connected_root",
    "ambiguous_disc_root",
    "short_or_folded_arm",
    "parent_not_unique",
    "scale_topology_or_angle_unstable",
}
