#!/usr/bin/env python3

"""ArtVIP/OpenUSD to URDF converter."""

import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path
from xml.dom import minidom
import xml.etree.ElementTree as ET

from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


# ============================================================
# CONFIG
# ============================================================

USD_PATH = (
    "/home/tracy/ArtVIP/Scenes/kitchen/kitchen.usd"
)

OUTPUT_DIR = "/home/tracy/converted/kitchen"

DEFAULT_EFFORT = 100.0
DEFAULT_VELOCITY = 1.0

ADD_WORLD_LINK = True
AUTO_GROUND = True

# Visual filtering. ArtVIP geometry is normally purpose=default/render.
EXPORT_ONLY_VISIBLE_VISUALS = True

# Visual: OBJ + MTL + textures. Collision: STL.
EXPORT_MATERIALS = True
COPY_TEXTURES = True

# Diagnostics.
DEBUG_JOINTS = True
DEBUG_AFFINE = True
DEBUG_MESH_FILTER = False

ANCHOR_WARNING_TOLERANCE_M = 1e-3
RIGID_NORM_TOL = 1e-4
RIGID_ORTHO_TOL = 1e-4
RIGID_DET_TOL = 1e-4


# ============================================================
# BASIC HELPERS
# ============================================================

def sanitize_name(name):
    """Make a USD prim name safe for URDF/file names."""
    name = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    if not name:
        name = "link"
    if name[0].isdigit():
        name = "_" + name
    return name


def axis_to_index(axis):
    axis = str(axis).upper()
    if axis == "X":
        return 0
    if axis == "Y":
        return 1
    if axis == "Z":
        return 2
    raise RuntimeError(f"Unsupported joint axis: {axis}")


def axis_to_xyz(axis):
    i = axis_to_index(axis)
    v = [0.0, 0.0, 0.0]
    v[i] = 1.0
    return tuple(v)


def up_axis_vector(up_axis):
    up_axis = str(up_axis).upper()
    if up_axis == "X":
        return (1.0, 0.0, 0.0)
    if up_axis == "Y":
        return (0.0, 1.0, 0.0)
    if up_axis == "Z":
        return (0.0, 0.0, 1.0)
    raise RuntimeError(f"Unsupported stage upAxis: {up_axis}")


# ============================================================
# SMALL VECTOR / MATRIX MATH (row-vector convention)
# ============================================================

def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def cross(a, b):
    return (
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    )


def norm(v):
    return math.sqrt(dot(v, v))


def normalize(v):
    n = norm(v)
    if n < 1e-15:
        raise RuntimeError(f"Cannot normalize near-zero vector: {v}")
    return (v[0]/n, v[1]/n, v[2]/n)


def safe_normalize(v):
    n = norm(v)
    if n < 1e-15:
        return None
    return (v[0]/n, v[1]/n, v[2]/n)


def add(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def subtract(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def scale(v, s):
    return (v[0]*s, v[1]*s, v[2]*s)


def transpose3(M):
    return [
        [M[j][i] for j in range(3)]
        for i in range(3)
    ]


def matmul3(A, B):
    return [
        [
            sum(A[i][k] * B[k][j] for k in range(3))
            for j in range(3)
        ]
        for i in range(3)
    ]


def row_vec_mul(v, M):
    """Row vector v times 3x3 matrix M."""
    return (
        v[0]*M[0][0] + v[1]*M[1][0] + v[2]*M[2][0],
        v[0]*M[0][1] + v[1]*M[1][1] + v[2]*M[2][1],
        v[0]*M[0][2] + v[1]*M[1][2] + v[2]*M[2][2],
    )


def determinant3(M):
    return (
        M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1])
        - M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0])
        + M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0])
    )


def gf_linear_rows(matrix):
    """
    Return the 3 transformed local basis vectors from a Gf.Matrix4d.

    Gf uses row-vector transform convention, so rows 0..2 are the transformed
    local X/Y/Z basis directions (including scale/shear/reflection).
    """
    return [
        [float(matrix[i][j]) for j in range(3)]
        for i in range(3)
    ]


def gf_translation(matrix):
    t = matrix.ExtractTranslation()
    return (float(t[0]), float(t[1]), float(t[2]))


# ============================================================
# RIGID FRAME REPRESENTATION
# ============================================================
# A frame is a dict:
#   frame["R"] : 3x3 proper row-rotation, local -> world
#   frame["t"] : translation in USD stage units, local origin in world
#
# For a local point p (row vector):
#   p_world = p * R + t
# ============================================================

def choose_up_preserving_axis(raw_rows, up_axis):
    """Choose the local basis row most aligned with the stage world-up axis."""
    world_up = up_axis_vector(up_axis)
    best_idx = None
    best_score = -1.0

    for i, row in enumerate(raw_rows):
        u = safe_normalize(row)
        if u is None:
            continue
        score = abs(dot(u, world_up))
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None:
        raise RuntimeError("Affine matrix has no usable basis vector.")

    return best_idx


def fallback_perpendicular(primary):
    """Return a stable unit vector perpendicular to primary."""
    candidates = [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ]

    # Pick the global axis least parallel to primary.
    base = min(candidates, key=lambda a: abs(dot(a, primary)))
    projected = subtract(base, scale(primary, dot(base, primary)))
    return normalize(projected)


def orthonormalize_rows(raw_rows, preferred_axis):
    """Create a proper rotation while preserving one source basis axis."""
    p = int(preferred_axis)
    q = [None, None, None]

    primary = safe_normalize(raw_rows[p])
    if primary is None:
        raise RuntimeError(
            f"Preferred affine basis axis {p} has zero length; cannot rigidize."
        )
    q[p] = primary

    remaining = [i for i in range(3) if i != p]

    # Choose the remaining raw axis that contains the most independent
    # information after removing its component along the preferred axis.
    best_s = None
    best_residual = None
    best_norm = -1.0

    for s in remaining:
        r = tuple(raw_rows[s])
        residual = subtract(r, scale(primary, dot(r, primary)))
        residual_norm = norm(residual)
        if residual_norm > best_norm:
            best_norm = residual_norm
            best_s = s
            best_residual = residual

    s = best_s
    if best_residual is None or norm(best_residual) < 1e-12:
        secondary = fallback_perpendicular(primary)
    else:
        secondary = normalize(best_residual)

    q[s] = secondary
    missing = next(i for i in range(3) if q[i] is None)

    # Enforce right-handed local basis: q0 x q1 = q2.
    if missing == 2:
        q[2] = normalize(cross(q[0], q[1]))
    elif missing == 1:
        q[1] = normalize(cross(q[2], q[0]))
    else:  # missing == 0
        q[0] = normalize(cross(q[1], q[2]))

    # Recompute the secondary axis once from the other two to reduce numerical
    # error while keeping the preferred axis exactly fixed.
    if s == 0:
        q[0] = normalize(cross(q[1], q[2]))
    elif s == 1:
        q[1] = normalize(cross(q[2], q[0]))
    else:
        q[2] = normalize(cross(q[0], q[1]))

    R = [list(q[0]), list(q[1]), list(q[2])]

    det = determinant3(R)
    if det < 0.999999:
        raise RuntimeError(
            f"Rigidization failed to create a proper rotation (det={det})."
        )

    return R


def analyze_affine(matrix):
    rows = gf_linear_rows(matrix)
    scales = [norm(r) for r in rows]
    det = determinant3(rows)

    unit_rows = [safe_normalize(r) for r in rows]
    orth_errors = []
    for i in range(3):
        for j in range(i + 1, 3):
            if unit_rows[i] is None or unit_rows[j] is None:
                orth_errors.append(float("inf"))
            else:
                orth_errors.append(abs(dot(unit_rows[i], unit_rows[j])))

    max_orth_error = max(orth_errors) if orth_errors else 0.0
    max_norm_error = max(abs(s - 1.0) for s in scales)

    is_rigid = (
        max_norm_error <= RIGID_NORM_TOL
        and max_orth_error <= RIGID_ORTHO_TOL
        and abs(det - 1.0) <= RIGID_DET_TOL
    )

    return {
        "linear_rows": rows,
        "basis_norms": scales,
        "determinant": det,
        "max_normalized_orthogonality_error": max_orth_error,
        "max_basis_norm_error": max_norm_error,
        "has_reflection": det < 0.0,
        "is_rigid": is_rigid,
    }


def rigid_frame_from_affine(matrix, up_axis, preferred_axis=None):
    """Project a USD affine frame to translation plus proper rotation."""
    raw_rows = gf_linear_rows(matrix)

    if preferred_axis is None:
        preferred_axis = choose_up_preserving_axis(raw_rows, up_axis)

    R = orthonormalize_rows(raw_rows, preferred_axis)
    t = gf_translation(matrix)

    return {
        "R": R,
        "t": t,
        "preferred_axis": int(preferred_axis),
        "affine": analyze_affine(matrix),
    }


def world_point_to_frame(p_world, frame):
    """World point -> local coordinates of a PURE RIGID frame."""
    d = subtract(p_world, frame["t"])
    return row_vec_mul(d, transpose3(frame["R"]))


def relative_rigid_frame(parent_frame, child_frame):
    """
    Express child pure-rigid frame in parent pure-rigid coordinates.

    Row-vector convention:
      R_rel = R_child * R_parent^T
      t_rel = (t_child - t_parent) * R_parent^T
    """
    Rp_T = transpose3(parent_frame["R"])
    R_rel = matmul3(child_frame["R"], Rp_T)
    t_rel = row_vec_mul(
        subtract(child_frame["t"], parent_frame["t"]),
        Rp_T,
    )
    return R_rel, t_rel


def matrix_to_rpy(R_column):
    """
    Column-vector 3x3 rotation -> URDF roll/pitch/yaw.
    URDF uses R = Rz(yaw) * Ry(pitch) * Rx(roll).
    """
    value = max(-1.0, min(1.0, -R_column[2][0]))
    pitch = math.asin(value)

    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(R_column[2][1], R_column[2][2])
        yaw = math.atan2(R_column[1][0], R_column[0][0])
    else:
        roll = math.atan2(-R_column[1][2], R_column[1][1])
        yaw = 0.0

    return roll, pitch, yaw


def relative_frame_to_xyz_rpy(parent_frame, child_frame, meters_per_unit):
    R_row, t_stage = relative_rigid_frame(parent_frame, child_frame)

    # Row-vector Gf rotation -> conventional column-vector URDF matrix.
    R_column = transpose3(R_row)
    rpy = matrix_to_rpy(R_column)

    xyz = tuple(float(x) * meters_per_unit for x in t_stage)
    return xyz, rpy


# ============================================================
# VISUAL MESH FILTER
# ============================================================

def is_visual_mesh(prim):
    if not prim.IsA(UsdGeom.Mesh):
        return False

    if not EXPORT_ONLY_VISIBLE_VISUALS:
        return True

    imageable = UsdGeom.Imageable(prim)

    visibility = imageable.ComputeVisibility()
    if visibility == UsdGeom.Tokens.invisible:
        return False

    purpose = imageable.ComputePurpose()
    if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
        return False

    return True


def mesh_filter_reason(prim):
    if not prim.IsA(UsdGeom.Mesh):
        return "not_mesh"

    if not EXPORT_ONLY_VISIBLE_VISUALS:
        return "visual"

    imageable = UsdGeom.Imageable(prim)
    visibility = imageable.ComputeVisibility()
    purpose = imageable.ComputePurpose()

    if visibility == UsdGeom.Tokens.invisible:
        return "invisible"
    if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
        return f"purpose={purpose}"
    return "visual"


# ============================================================
# USD JOINT HELPERS
# ============================================================

def get_joint_type(prim):
    if prim.IsA(UsdPhysics.RevoluteJoint):
        return "revolute"
    if prim.IsA(UsdPhysics.PrismaticJoint):
        return "prismatic"
    if prim.IsA(UsdPhysics.FixedJoint):
        return "fixed"
    if prim.IsA(UsdPhysics.Joint):
        return "unsupported"
    return None


def get_joint_data(prim):
    joint_type = get_joint_type(prim)
    joint = UsdPhysics.Joint(prim)

    body0_targets = joint.GetBody0Rel().GetTargets()
    body1_targets = joint.GetBody1Rel().GetTargets()

    if len(body0_targets) > 1 or len(body1_targets) > 1:
        raise RuntimeError(
            f"Joint {prim.GetPath()} has multiple body targets."
        )

    body0 = body0_targets[0] if body0_targets else None
    body1 = body1_targets[0] if body1_targets else None

    pos0 = joint.GetLocalPos0Attr().Get()
    pos1 = joint.GetLocalPos1Attr().Get()
    rot0 = joint.GetLocalRot0Attr().Get()
    rot1 = joint.GetLocalRot1Attr().Get()

    if pos0 is None:
        pos0 = Gf.Vec3f(0, 0, 0)
    if pos1 is None:
        pos1 = Gf.Vec3f(0, 0, 0)
    if rot0 is None:
        rot0 = Gf.Quatf(1, Gf.Vec3f(0, 0, 0))
    if rot1 is None:
        rot1 = Gf.Quatf(1, Gf.Vec3f(0, 0, 0))

    data = {
        "name": prim.GetName(),
        "path": str(prim.GetPath()),
        "type": joint_type,
        "body0": body0,
        "body1": body1,
        "pos0": pos0,
        "pos1": pos1,
        "rot0": rot0,
        "rot1": rot1,
    }

    if joint_type == "revolute":
        rj = UsdPhysics.RevoluteJoint(prim)
        data["axis"] = str(rj.GetAxisAttr().Get())
        data["lower"] = rj.GetLowerLimitAttr().Get()
        data["upper"] = rj.GetUpperLimitAttr().Get()

    elif joint_type == "prismatic":
        pj = UsdPhysics.PrismaticJoint(prim)
        data["axis"] = str(pj.GetAxisAttr().Get())
        data["lower"] = pj.GetLowerLimitAttr().Get()
        data["upper"] = pj.GetUpperLimitAttr().Get()

    return data


def make_local_frame_matrix(pos, quat):
    """Joint-local -> rigid-body-local Gf affine matrix."""
    imag = quat.GetImaginary()
    qd = Gf.Quatd(
        float(quat.GetReal()),
        Gf.Vec3d(
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
        ),
    )

    m = Gf.Matrix4d(1.0)
    m.SetRotateOnly(qd)
    m.SetTranslateOnly(
        Gf.Vec3d(
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )
    )
    return m


def get_joint_side_world_matrix(stage, body_path, pos, rot, xform_cache):
    """Return FULL AFFINE authored joint-side frame in USD world coordinates."""
    body_prim = stage.GetPrimAtPath(body_path)

    if not body_prim or not body_prim.IsValid():
        raise RuntimeError(f"Invalid rigid body path: {body_path}")

    body_world = xform_cache.GetLocalToWorldTransform(body_prim)
    local_joint = make_local_frame_matrix(pos, rot)

    # Gf row-vector convention: joint local -> body -> world.
    return local_joint * body_world


def compute_anchor_error(stage, joint, meters_per_unit, xform_cache):
    if joint["body0"] is None or joint["body1"] is None:
        return None

    f0 = get_joint_side_world_matrix(
        stage,
        joint["body0"],
        joint["pos0"],
        joint["rot0"],
        xform_cache,
    )
    f1 = get_joint_side_world_matrix(
        stage,
        joint["body1"],
        joint["pos1"],
        joint["rot1"],
        xform_cache,
    )

    p0 = gf_translation(f0)
    p1 = gf_translation(f1)

    delta = tuple((p0[i] - p1[i]) * meters_per_unit for i in range(3))
    error = norm(delta)

    return {
        "error_m": error,
        "delta_m": list(delta),
    }


def compute_urdf_joint_axis(stage, joint, child_frame, xform_cache):
    local_axis = axis_to_xyz(joint["axis"])

    side0 = get_joint_side_world_matrix(
        stage,
        joint["body0"],
        joint["pos0"],
        joint["rot0"],
        xform_cache,
    )
    side1 = get_joint_side_world_matrix(
        stage,
        joint["body1"],
        joint["pos1"],
        joint["rot1"],
        xform_cache,
    )

    axis0_world = normalize(row_vec_mul(local_axis, gf_linear_rows(side0)))
    axis1_world = normalize(row_vec_mul(local_axis, gf_linear_rows(side1)))
    alignment = dot(axis0_world, axis1_world)

    axis_world = axis0_world
    axis_urdf = normalize(
        row_vec_mul(axis_world, transpose3(child_frame["R"]))
    )

    canonical = axis_to_xyz(joint["axis"])
    canonical_dot = dot(axis_urdf, canonical)
    if abs(canonical_dot) > 0.999999:
        axis_urdf = scale(canonical, 1.0 if canonical_dot >= 0.0 else -1.0)

    return axis_urdf, {
        "usd_axis": list(local_axis),
        "side_alignment": alignment,
        "axis0_world": list(axis0_world),
        "axis1_world": list(axis1_world),
    }


# ============================================================
# LINK FRAMES
# ============================================================

def build_link_rigid_frames(stage, link_paths, incoming_joint, root_paths, up_axis):
    cache = UsdGeom.XformCache()
    result = {}
    root_paths = set(root_paths)

    for path in link_paths:
        if path in root_paths:
            prim = stage.GetPrimAtPath(path)
            affine = cache.GetLocalToWorldTransform(prim)
            result[path] = rigid_frame_from_affine(
                affine,
                up_axis=up_axis,
                preferred_axis=None,
            )
            result[path]["source"] = "component_root"
            continue

        joint = incoming_joint.get(path)
        if joint is None:
            raise RuntimeError(f"No incoming joint found for link {path}")

        joint1_affine = get_joint_side_world_matrix(
            stage,
            path,
            joint["pos1"],
            joint["rot1"],
            cache,
        )

        preferred_axis = None
        if joint["type"] in ("revolute", "prismatic"):
            preferred_axis = axis_to_index(joint["axis"])

        result[path] = rigid_frame_from_affine(
            joint1_affine,
            up_axis=up_axis,
            preferred_axis=preferred_axis,
        )
        result[path]["source"] = "body1_joint_frame"
        result[path]["incoming_joint"] = joint["name"]

    return result


# ============================================================
# USD MATERIAL / TEXTURE HELPERS
# ============================================================

def _first_connected_source(shade_input):
    """Return first UsdShadeConnectionSourceInfo for an input, if any."""
    if not shade_input:
        return None
    try:
        result = shade_input.GetConnectedSources()
        infos = result[0] if isinstance(result, tuple) else result
        return infos[0] if infos else None
    except Exception:
        return None


def _connected_shader(shade_input):
    info = _first_connected_source(shade_input)
    if info is None:
        return None
    try:
        prim = info.source.GetPrim()
        shader = UsdShade.Shader(prim)
        if shader and shader.GetPrim().IsValid():
            return shader
    except Exception:
        pass
    return None


def _shader_id(shader):
    if not shader:
        return ""
    try:
        value = shader.GetIdAttr().Get()
        return str(value) if value is not None else ""
    except Exception:
        return ""


def _input_value(shader, name, default=None):
    try:
        inp = shader.GetInput(name)
        value = inp.Get() if inp else None
        return default if value is None else value
    except Exception:
        return default


def _to_float_list(value, n, default):
    if value is None:
        return list(default)
    try:
        return [float(value[i]) for i in range(n)]
    except Exception:
        return list(default)


def _resolve_material_input_value(connectable, input_name, depth=0):
    """Resolve a simple material/interface input connection such as stPrimvarName."""
    if depth > 8 or not connectable:
        return None
    try:
        inp = connectable.GetInput(input_name)
    except Exception:
        inp = None
    if not inp:
        return None

    try:
        value = inp.Get()
        if value is not None:
            return value
    except Exception:
        pass

    info = _first_connected_source(inp)
    if info is None:
        return None
    try:
        return _resolve_material_input_value(
            info.source,
            str(info.sourceName),
            depth + 1,
        )
    except Exception:
        return None


def _resolve_primvar_reader_name(reader_shader):
    if not reader_shader:
        return "st"
    try:
        var_in = reader_shader.GetInput("varname")
        value = var_in.Get() if var_in else None
        if value is not None:
            return str(value)
        info = _first_connected_source(var_in)
        if info is not None:
            value = _resolve_material_input_value(
                info.source,
                str(info.sourceName),
            )
            if value is not None:
                return str(value)
    except Exception:
        pass
    return "st"


def _uv_spec_from_texture_shader(texture_shader):
    """
    Read the common UsdUVTexture <- UsdTransform2d <- UsdPrimvarReader chain.
    The transform is baked into exported OBJ vt coordinates.
    """
    result = {
        "primvar": "st",
        "scale": [1.0, 1.0],
        "rotation_deg": 0.0,
        "translation": [0.0, 0.0],
        "wrapS": str(_input_value(texture_shader, "wrapS", "repeat")),
        "wrapT": str(_input_value(texture_shader, "wrapT", "repeat")),
    }

    try:
        st_input = texture_shader.GetInput("st")
        upstream = _connected_shader(st_input)
    except Exception:
        upstream = None

    if upstream is None:
        return result

    upstream_id = _shader_id(upstream)

    if upstream_id == "UsdTransform2d":
        result["scale"] = _to_float_list(
            _input_value(upstream, "scale", (1.0, 1.0)),
            2,
            (1.0, 1.0),
        )
        result["rotation_deg"] = float(
            _input_value(upstream, "rotation", 0.0) or 0.0
        )
        result["translation"] = _to_float_list(
            _input_value(upstream, "translation", (0.0, 0.0)),
            2,
            (0.0, 0.0),
        )
        reader = _connected_shader(upstream.GetInput("in"))
        if reader and _shader_id(reader).startswith("UsdPrimvarReader"):
            result["primvar"] = _resolve_primvar_reader_name(reader)

    elif upstream_id.startswith("UsdPrimvarReader"):
        result["primvar"] = _resolve_primvar_reader_name(upstream)

    return result


def apply_uv_transform(uv, spec):
    """UsdTransform2d: result = in * scale * rotate + translation."""
    u = float(uv[0]) * float(spec["scale"][0])
    v = float(uv[1]) * float(spec["scale"][1])

    theta = math.radians(float(spec["rotation_deg"]))
    c = math.cos(theta)
    s = math.sin(theta)

    ur = u * c - v * s
    vr = u * s + v * c

    return (
        ur + float(spec["translation"][0]),
        vr + float(spec["translation"][1]),
    )


def _resolve_asset_path(asset_value):
    if asset_value is None:
        return None

    if isinstance(asset_value, Sdf.AssetPath):
        if asset_value.resolvedPath:
            return str(asset_value.resolvedPath)
        raw = str(asset_value.path)
    else:
        raw = str(asset_value)

    if not raw:
        return None

    try:
        resolved = Ar.GetResolver().Resolve(raw)
        if resolved:
            return str(resolved)
    except Exception:
        pass

    p = Path(raw).expanduser()
    return str(p.resolve()) if p.exists() else raw


def _copy_texture(source_path, texture_dir):
    if not source_path:
        return None

    if "<UDIM>" in source_path:
        # Keep the source path in the report. Standard URDF/RViz has no
        # first-class UDIM material model, so do not pretend it is supported.
        return None

    source = Path(source_path)
    if not source.exists() or not source.is_file():
        return None

    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    dest = texture_dir / f"{sanitize_name(source.stem)}_{digest}{source.suffix.lower()}"
    if COPY_TEXTURES and not dest.exists():
        shutil.copy2(source, dest)
    return dest if dest.exists() else source


def _extract_texture_input(preview_shader, input_name, texture_dir):
    inp = preview_shader.GetInput(input_name)
    tex_shader = _connected_shader(inp)
    if not tex_shader or _shader_id(tex_shader) != "UsdUVTexture":
        return None

    source_path = _resolve_asset_path(_input_value(tex_shader, "file", None))
    copied = _copy_texture(source_path, texture_dir)

    scale_value = _input_value(tex_shader, "scale", None)
    bias_value = _input_value(tex_shader, "bias", None)

    return {
        "source": source_path,
        "copied": str(copied) if copied else None,
        "uri": copied.resolve().as_uri() if copied and copied.exists() else None,
        "uv": _uv_spec_from_texture_shader(tex_shader),
        "texture_scale": _to_float_list(scale_value, 4, (1.0, 1.0, 1.0, 1.0))
            if scale_value is not None else None,
        "texture_bias": _to_float_list(bias_value, 4, (0.0, 0.0, 0.0, 0.0))
            if bias_value is not None else None,
    }


def get_bound_material(prim):
    try:
        result = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        material = result[0] if isinstance(result, tuple) else result
        if material and material.GetPrim().IsValid():
            return material
    except Exception:
        pass
    return None


def material_key(material):
    if material and material.GetPrim().IsValid():
        return str(material.GetPath())
    return "__default_material__"


def extract_material_data(material, texture_dir):
    """
    Extract the UsdPreviewSurface subset that standard URDF/RViz can use.

    Visible in standard URDF/RViz:
      - diffuse/base color
      - opacity
      - diffuse texture + UVs

    Also preserved in conversion_report.json when present:
      metallic, roughness, clearcoat, clearcoatRoughness, ior,
      specularColor, emissiveColor, normal/roughness/metallic textures, etc.
    """
    if material is None or not material.GetPrim().IsValid():
        return {
            "key": "__default_material__",
            "name": "default_material",
            "usd_path": None,
            "shader_id": None,
            "rgba": [0.65, 0.65, 0.65, 1.0],
            "diffuse_texture": None,
            "pbr": {},
            "textures": {},
            "warnings": ["No bound USD material; using gray fallback."],
        }

    key = str(material.GetPath())
    name = sanitize_name(material.GetPrim().GetName())
    warnings = []

    try:
        surface_result = material.ComputeSurfaceSource()
        shader = surface_result[0] if isinstance(surface_result, tuple) else surface_result
    except Exception:
        shader = None

    if not shader or not shader.GetPrim().IsValid():
        return {
            "key": key,
            "name": name,
            "usd_path": key,
            "shader_id": None,
            "rgba": [0.65, 0.65, 0.65, 1.0],
            "diffuse_texture": None,
            "pbr": {},
            "textures": {},
            "warnings": ["Material has no resolvable surface shader."],
        }

    shader_id = _shader_id(shader)
    if shader_id != "UsdPreviewSurface":
        warnings.append(
            f"Surface shader {shader_id!r} is not UsdPreviewSurface; "
            "only fallback color can be represented reliably in URDF/RViz."
        )

    diffuse = _to_float_list(
        _input_value(shader, "diffuseColor", (0.65, 0.65, 0.65)),
        3,
        (0.65, 0.65, 0.65),
    )
    opacity = float(_input_value(shader, "opacity", 1.0) or 1.0)

    pbr = {
        "emissiveColor": _to_float_list(
            _input_value(shader, "emissiveColor", (0.0, 0.0, 0.0)),
            3,
            (0.0, 0.0, 0.0),
        ),
        "metallic": float(_input_value(shader, "metallic", 0.0) or 0.0),
        "roughness": float(_input_value(shader, "roughness", 0.5) or 0.5),
        "clearcoat": float(_input_value(shader, "clearcoat", 0.0) or 0.0),
        "clearcoatRoughness": float(
            _input_value(shader, "clearcoatRoughness", 0.01) or 0.01
        ),
        "ior": float(_input_value(shader, "ior", 1.5) or 1.5),
        "specularColor": _to_float_list(
            _input_value(shader, "specularColor", (0.0, 0.0, 0.0)),
            3,
            (0.0, 0.0, 0.0),
        ),
        "useSpecularWorkflow": int(_input_value(shader, "useSpecularWorkflow", 0) or 0),
    }

    texture_inputs = [
        "diffuseColor",
        "normal",
        "roughness",
        "metallic",
        "emissiveColor",
        "opacity",
        "occlusion",
    ]
    textures = {}
    for input_name in texture_inputs:
        try:
            tex = _extract_texture_input(shader, input_name, texture_dir)
        except Exception as exc:
            tex = None
            warnings.append(f"Failed reading texture input {input_name}: {exc}")
        if tex:
            textures[input_name] = tex
            if tex["source"] and tex["copied"] is None:
                warnings.append(
                    f"Texture for {input_name} could not be copied/resolved: {tex['source']}"
                )

    diffuse_texture = textures.get("diffuseColor")

    # When diffuseColor is driven directly by a texture, using the authored
    # constant diffuse fallback as a tint is usually wrong. Keep white RGB and
    # preserve only opacity in the URDF material.
    rgba = [1.0, 1.0, 1.0, opacity] if diffuse_texture else [*diffuse, opacity]

    return {
        "key": key,
        "name": name,
        "usd_path": key,
        "shader_id": shader_id,
        "rgba": rgba,
        "authored_diffuseColor": diffuse,
        "diffuse_texture": diffuse_texture,
        "pbr": pbr,
        "textures": textures,
        "warnings": warnings,
    }


def get_face_materials(mesh_prim, face_count):
    """Return one bound material per polygon face, including materialBind subsets."""
    default_material = get_bound_material(mesh_prim)
    result = [default_material] * int(face_count)

    for child in mesh_prim.GetChildren():
        if not child.IsA(UsdGeom.Subset):
            continue
        subset = UsdGeom.Subset(child)
        try:
            family = str(subset.GetFamilyNameAttr().Get())
            element_type = str(subset.GetElementTypeAttr().Get())
        except Exception:
            continue

        if family != "materialBind" or element_type != "face":
            continue

        subset_material = get_bound_material(child)
        try:
            face_indices = subset.GetIndicesAttr().Get() or []
        except Exception:
            face_indices = []

        for face_index in face_indices:
            i = int(face_index)
            if 0 <= i < len(result):
                result[i] = subset_material

    return result


def get_uv_primvar(mesh_prim, primvar_name):
    api = UsdGeom.PrimvarsAPI(mesh_prim)
    primvar = None
    try:
        primvar = api.FindPrimvarWithInheritance(primvar_name)
    except Exception:
        pass
    if not primvar:
        try:
            primvar = api.GetPrimvar(primvar_name)
        except Exception:
            primvar = None
    if not primvar:
        return None, None, None

    try:
        values = primvar.ComputeFlattened()
    except Exception:
        values = primvar.Get()
    if not values:
        return None, None, None

    try:
        interpolation = str(primvar.GetInterpolation())
    except Exception:
        interpolation = "vertex"

    return primvar, values, interpolation


def uv_for_face_corner(values, interpolation, face_index, corner_index, point_index):
    if values is None:
        return None
    try:
        if interpolation == "constant":
            value = values[0]
        elif interpolation == "uniform":
            value = values[face_index]
        elif interpolation in ("vertex", "varying"):
            value = values[point_index]
        elif interpolation == "faceVarying":
            value = values[corner_index]
        else:
            value = values[point_index]
        return (float(value[0]), float(value[1]))
    except Exception:
        return None



def write_obj_mtl(mtl_path, material_name, mat_data):
    """Write a minimal Wavefront MTL that Assimp/RViz can consume reliably."""
    rgba = mat_data.get("rgba", [0.65, 0.65, 0.65, 1.0])
    kd = [float(rgba[0]), float(rgba[1]), float(rgba[2])]
    alpha = max(0.0, min(1.0, float(rgba[3])))

    diffuse_tex = mat_data.get("diffuse_texture")
    texture_path = None
    if diffuse_tex:
        copied = diffuse_tex.get("copied")
        if copied:
            p = Path(copied)
            if p.exists():
                texture_path = p.resolve()

    # When a diffuse texture is present, keep Kd white so the texture is not
    # unintentionally tinted by a gray fallback color.
    if texture_path is not None:
        kd = [1.0, 1.0, 1.0]

    with open(mtl_path, "w", encoding="utf-8") as out:
        out.write("# material exported from USD\n")
        out.write(f"newmtl {material_name}\n")
        out.write("Ka 0 0 0\n")
        out.write(f"Kd {kd[0]:.9g} {kd[1]:.9g} {kd[2]:.9g}\n")
        out.write("Ks 0 0 0\n")
        out.write(f"d {alpha:.9g}\n")
        out.write("illum 1\n")

        if texture_path is not None:
            rel = os.path.relpath(texture_path, start=mtl_path.parent.resolve())
            rel = Path(rel).as_posix()
            out.write(f"map_Kd {rel}\n")

    return texture_path is not None


def export_link_visual_objs(
    stage,
    mesh_prims,
    link_rigid_frame,
    meters_per_unit,
    visual_dir,
    texture_dir,
    link_name,
    material_cache,
    warning_sink,
):
    """Export OBJ+MTL visuals grouped by material."""
    cache = UsdGeom.XformCache()
    states = {}
    total_triangles = 0
    warned_missing_uv = set()

    def ensure_material(material):
        key = material_key(material)
        if key not in material_cache:
            material_cache[key] = extract_material_data(material, texture_dir)
        return material_cache[key]

    def ensure_state(mat_data):
        key = mat_data["key"]
        if key in states:
            return states[key]

        mat_token = sanitize_name(mat_data["name"])
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]
        obj_path = visual_dir / f"{link_name}__{mat_token}_{digest}.obj"
        mtl_path = obj_path.with_suffix(".mtl")
        mtl_name = f"usd_{mat_token}_{digest}"
        has_mtl_texture = write_obj_mtl(mtl_path, mtl_name, mat_data)

        fh = open(obj_path, "w", encoding="utf-8")
        fh.write(f"# visual for {link_name}\n")
        fh.write(f"mtllib {mtl_path.name}\n")
        fh.write(f"o {link_name}__{mat_token}\n")
        fh.write(f"usemtl {mtl_name}\n")
        state = {
            "file": fh,
            "path": obj_path,
            "mtl_path": mtl_path,
            "mtl_name": mtl_name,
            "has_mtl_texture": has_mtl_texture,
            "v": 0,
            "vt": 0,
            "vn": 0,
            "triangles": 0,
            "material": mat_data,
        }
        states[key] = state
        return state

    for mesh_prim in mesh_prims:
        mesh = UsdGeom.Mesh(mesh_prim)
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices:
            continue

        mesh_world = cache.GetLocalToWorldTransform(mesh_prim)
        transformed_points = []
        for p in points:
            p_local = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
            pw = mesh_world.Transform(p_local)
            p_world = (float(pw[0]), float(pw[1]), float(pw[2]))
            p_link = world_point_to_frame(p_world, link_rigid_frame)
            transformed_points.append(tuple(x * meters_per_unit for x in p_link))

            # Keep winding consistent after reflection.
        reflected = determinant3(gf_linear_rows(mesh_world)) < 0.0

        face_materials = get_face_materials(mesh_prim, len(counts))
        uv_cache = {}
        cursor = 0

        for face_index, count_value in enumerate(counts):
            count = int(count_value)
            face_start = cursor
            face = [int(indices[cursor + i]) for i in range(count)]
            cursor += count
            if count < 3:
                continue

            mat_data = ensure_material(face_materials[face_index])
            state = ensure_state(mat_data)
            diffuse_tex = mat_data.get("diffuse_texture")

            uv_values = uv_interp = uv_spec = None
            if diffuse_tex:
                uv_spec = diffuse_tex["uv"]
                pv_name = uv_spec.get("primvar", "st")
                if pv_name not in uv_cache:
                    _, values, interp = get_uv_primvar(mesh_prim, pv_name)
                    uv_cache[pv_name] = (values, interp)
                uv_values, uv_interp = uv_cache[pv_name]

            for fan_i in range(1, count - 1):
                local_corners = [0, fan_i, fan_i + 1]
                point_indices = [face[c] for c in local_corners]
                corner_indices = [face_start + c for c in local_corners]

                if reflected:
                    point_indices[1], point_indices[2] = point_indices[2], point_indices[1]
                    corner_indices[1], corner_indices[2] = corner_indices[2], corner_indices[1]

                verts = [transformed_points[i] for i in point_indices]
                e1 = subtract(verts[1], verts[0])
                e2 = subtract(verts[2], verts[0])
                n_unit = safe_normalize(cross(e1, e2)) or (0.0, 0.0, 0.0)

                uvs = None
                if diffuse_tex and uv_values is not None:
                    raw_uvs = [
                        uv_for_face_corner(
                            uv_values,
                            uv_interp,
                            face_index,
                            corner_indices[i],
                            point_indices[i],
                        )
                        for i in range(3)
                    ]
                    if all(x is not None for x in raw_uvs):
                        uvs = [apply_uv_transform(x, uv_spec) for x in raw_uvs]

                if diffuse_tex and uvs is None:
                    warn_key = (str(mesh_prim.GetPath()), mat_data["key"])
                    if warn_key not in warned_missing_uv:
                        warning_sink.append(
                            f"Texture material {mat_data['name']} on {mesh_prim.GetPath()} "
                            "has no usable UV primvar; texture may not render correctly."
                        )
                        warned_missing_uv.add(warn_key)

                fh = state["file"]
                v_start = state["v"] + 1
                for v in verts:
                    fh.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
                state["v"] += 3

                vt_start = None
                if uvs is not None:
                    vt_start = state["vt"] + 1
                    for uv in uvs:
                        fh.write(f"vt {uv[0]:.9g} {uv[1]:.9g}\n")
                    state["vt"] += 3

                vn_index = state["vn"] + 1
                fh.write(f"vn {n_unit[0]:.9g} {n_unit[1]:.9g} {n_unit[2]:.9g}\n")
                state["vn"] += 1

                if vt_start is not None:
                    fh.write(
                        "f "
                        f"{v_start}/{vt_start}/{vn_index} "
                        f"{v_start+1}/{vt_start+1}/{vn_index} "
                        f"{v_start+2}/{vt_start+2}/{vn_index}\n"
                    )
                else:
                    fh.write(
                        "f "
                        f"{v_start}//{vn_index} "
                        f"{v_start+1}//{vn_index} "
                        f"{v_start+2}//{vn_index}\n"
                    )

                state["triangles"] += 1
                total_triangles += 1

    visuals = []
    for state in states.values():
        state["file"].close()
        if state["triangles"] <= 0:
            try:
                state["path"].unlink()
            except Exception:
                pass
            try:
                state["mtl_path"].unlink()
            except Exception:
                pass
            continue
        visuals.append({
            "mesh": str(state["path"]),
            "mesh_uri": state["path"].resolve().as_uri(),
            "mtl": str(state["mtl_path"]),
            "mtl_uri": state["mtl_path"].resolve().as_uri(),
            "has_mtl_texture": bool(state["has_mtl_texture"]),
            "triangle_count": state["triangles"],
            "material": state["material"],
        })

    return visuals, total_triangles


# ============================================================
# MESH OWNERSHIP / STL EXPORT
# ============================================================

def path_under(child_path, parent_path):
    if child_path == parent_path:
        return True
    return child_path.HasPrefix(parent_path)


def assign_meshes_to_links(stage, link_paths):
    result = {path: [] for path in link_paths}
    unowned = []
    skipped = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue

        if not is_visual_mesh(prim):
            skipped.append((str(prim.GetPath()), mesh_filter_reason(prim)))
            continue

        mesh_path = prim.GetPath()
        candidates = [
            link_path
            for link_path in link_paths
            if path_under(mesh_path, link_path)
        ]

        if not candidates:
            unowned.append(prim)
            continue

        owner = max(candidates, key=lambda p: len(str(p)))
        result[owner].append(prim)

    if DEBUG_MESH_FILTER and skipped:
        print("\n================ SKIPPED NON-VISUAL MESHES ================")
        for path, reason in skipped:
            print(f"{reason:22s} {path}")
        print("===========================================================\n")

    return result, unowned


def export_link_collision_stl(
    stage,
    mesh_prims,
    link_rigid_frame,
    meters_per_unit,
    output_path,
    solid_name,
):
    """Export collision meshes in the URDF link frame."""
    cache = UsdGeom.XformCache()
    triangle_count = 0

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(f"solid {solid_name}\n")

        for mesh_prim in mesh_prims:
            mesh = UsdGeom.Mesh(mesh_prim)

            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()

            if not points or not counts or not indices:
                continue

            mesh_world = cache.GetLocalToWorldTransform(mesh_prim)
            reflected = determinant3(gf_linear_rows(mesh_world)) < 0.0
            transformed_points = []

            for p in points:
                p_local = Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
                p_world_gf = mesh_world.Transform(p_local)
                p_world = (
                    float(p_world_gf[0]),
                    float(p_world_gf[1]),
                    float(p_world_gf[2]),
                )

                p_link = world_point_to_frame(p_world, link_rigid_frame)
                transformed_points.append(
                    tuple(x * meters_per_unit for x in p_link)
                )

            cursor = 0

            for count_value in counts:
                count = int(count_value)
                face = [int(indices[cursor + i]) for i in range(count)]
                cursor += count

                if count < 3:
                    continue

                # Triangulate polygon fan.
                for i in range(1, count - 1):
                    v0 = transformed_points[face[0]]
                    v1 = transformed_points[face[i]]
                    v2 = transformed_points[face[i + 1]]
                    if reflected:
                        v1, v2 = v2, v1

                    e1 = subtract(v1, v0)
                    e2 = subtract(v2, v0)
                    n = cross(e1, e2)
                    n_unit = safe_normalize(n) or (0.0, 0.0, 0.0)

                    out.write(
                        "  facet normal "
                        f"{n_unit[0]:.9g} {n_unit[1]:.9g} {n_unit[2]:.9g}\n"
                    )
                    out.write("    outer loop\n")

                    for v in (v0, v1, v2):
                        out.write(
                            "      vertex "
                            f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n"
                        )

                    out.write("    endloop\n")
                    out.write("  endfacet\n")
                    triangle_count += 1

        out.write(f"endsolid {solid_name}\n")

    return triangle_count


def compute_min_z_in_world(stage, meters_per_unit):
    cache = UsdGeom.XformCache()
    min_z = None

    for prim in stage.Traverse():
        if not is_visual_mesh(prim):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        if not points:
            continue

        mesh_world = cache.GetLocalToWorldTransform(prim)
        for p in points:
            pw = mesh_world.Transform(
                Gf.Vec3d(float(p[0]), float(p[1]), float(p[2]))
            )
            z_m = float(pw[2]) * meters_per_unit
            if min_z is None or z_m < min_z:
                min_z = z_m

    return 0.0 if min_z is None else min_z


# ============================================================
# URDF XML HELPERS
# ============================================================

def add_link_xml(robot, link_name, visual_records, collision_mesh_uri):
    link = ET.SubElement(robot, "link", {"name": link_name})

    for i, record in enumerate(visual_records):
        visual = ET.SubElement(link, "visual", {"name": f"visual_{i}"})
        ET.SubElement(visual, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(
            geometry,
            "mesh",
            {"filename": record["mesh_uri"], "scale": "1 1 1"},
        )

        mat = record["material"]
        diffuse_tex = mat.get("diffuse_texture")

        # Textured OBJ uses its MTL. Plain colors use URDF material.
        if not (diffuse_tex and record.get("has_mtl_texture")):
            material_xml = ET.SubElement(
                visual,
                "material",
                {"name": f"{link_name}_{sanitize_name(mat['name'])}_{i}"},
            )
            rgba = mat.get("rgba", [0.65, 0.65, 0.65, 1.0])
            ET.SubElement(
                material_xml,
                "color",
                {"rgba": " ".join(f"{float(x):.9g}" for x in rgba)},
            )

    if collision_mesh_uri:
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        collision_geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(
            collision_geometry,
            "mesh",
            {"filename": collision_mesh_uri, "scale": "1 1 1"},
        )


def pretty_xml(element):
    rough = ET.tostring(element, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ")


# ============================================================
# STATIC USD FALLBACK
# ============================================================

def choose_static_root(stage):
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsValid():
        return default_prim.GetPath()

    children = list(stage.GetPseudoRoot().GetChildren())
    if not children:
        raise RuntimeError("USD stage contains no usable prims.")

    return children[0].GetPath()


# ============================================================
# REPORT HELPERS
# ============================================================

def affine_summary_for_json(frame):
    a = frame["affine"]
    return {
        "basis_norms": [float(x) for x in a["basis_norms"]],
        "determinant": float(a["determinant"]),
        "max_normalized_orthogonality_error": float(
            a["max_normalized_orthogonality_error"]
        ),
        "max_basis_norm_error": float(a["max_basis_norm_error"]),
        "has_reflection": bool(a["has_reflection"]),
        "is_rigid": bool(a["is_rigid"]),
        "preferred_axis": int(frame["preferred_axis"]),
        "source": frame.get("source", "unknown"),
    }


def print_affine_diagnostics(link_names, link_frames):
    if not DEBUG_AFFINE:
        return

    print("\n================ AFFINE / RIGIDIZATION CHECK ===============")
    for path in sorted(link_frames.keys(), key=lambda p: str(p)):
        frame = link_frames[path]
        a = frame["affine"]
        name = link_names.get(path, str(path))
        status = "RIGID" if a["is_rigid"] else "AFFINE"
        refl = " reflection" if a["has_reflection"] else ""
        s = a["basis_norms"]
        print(
            f"{name:30s} {status:6s}{refl:12s} "
            f"det={a['determinant']:+.6f} "
            f"norms=({s[0]:.6f}, {s[1]:.6f}, {s[2]:.6f}) "
            f"orth={a['max_normalized_orthogonality_error']:.3e}"
        )
    print("===========================================================\n")


# ============================================================
# CONVERTER
# ============================================================

def convert(usd_path, output_dir, default_effort, default_velocity):
    usd_path = Path(usd_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    mesh_dir = output_dir / "meshes"
    visual_dir = mesh_dir / "visual"
    collision_dir = mesh_dir / "collision"
    texture_dir = output_dir / "textures"

    if not usd_path.exists() or not usd_path.is_file():
        raise RuntimeError(f"Invalid USD path: {usd_path}")
    if usd_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
        raise RuntimeError(f"Unsupported USD input: {usd_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir.mkdir(parents=True, exist_ok=True)
    collision_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Cannot open USD: {usd_path}")

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    if up_axis != "Z":
        raise RuntimeError(f"Expected Z-up stage, got {up_axis}")

    print(f"[INFO] USD: {usd_path}")
    print(f"[INFO] metersPerUnit = {meters_per_unit}")
    print(f"[INFO] upAxis = {up_axis}")

    joints = []
    unsupported_joints = []
    usd_world_joints = []
    link_paths = set()

    for prim in stage.Traverse():
        try:
            rb = UsdPhysics.RigidBodyAPI(prim)
            if rb:
                enabled = rb.GetRigidBodyEnabledAttr().Get()
                if enabled is not False:
                    link_paths.add(prim.GetPath())
        except Exception:
            pass

        joint_type = get_joint_type(prim)
        if joint_type is None:
            continue
        if joint_type == "unsupported":
            unsupported_joints.append(str(prim.GetPath()))
            continue

        data = get_joint_data(prim)
        if data["body0"] is not None:
            link_paths.add(data["body0"])
        if data["body1"] is not None:
            link_paths.add(data["body1"])

        if data["body0"] is None or data["body1"] is None:
            usd_world_joints.append(data)
            continue

        joints.append(data)

    if unsupported_joints:
        print("\n[ERROR] Unsupported USD joints:")
        for path in unsupported_joints:
            print("   ", path)
        raise RuntimeError("USD contains unsupported joint types.")

    incoming_joint = {}
    children_by_parent = {}

    for joint in joints:
        parent = joint["body0"]
        child = joint["body1"]

        if child in incoming_joint:
            raise RuntimeError(
                f"Link {child} has multiple parent joints. URDF requires a tree."
            )

        incoming_joint[child] = joint
        children_by_parent.setdefault(parent, []).append(child)

    root_paths = sorted(
        [path for path in link_paths if path not in incoming_joint],
        key=lambda p: str(p),
    )

    if link_paths and not root_paths:
        raise RuntimeError("No articulation root found. The rigid-body graph may contain a cycle.")

    visited = set()
    stack = list(root_paths)
    while stack:
        path = stack.pop()
        if path in visited:
            continue
        visited.add(path)
        stack.extend(children_by_parent.get(path, []))

    if visited != link_paths:
        missing = sorted([str(p) for p in (link_paths - visited)])
        raise RuntimeError(f"Rigid-body graph is not a URDF forest: {missing}")

    print(f"[INFO] Rigid-body links: {len(link_paths)}")
    print(f"[INFO] Articulation roots: {len(root_paths)}")

    link_names = {}
    used_link_names = {"world"}

    for path in sorted(link_paths, key=lambda p: str(p)):
        prim = stage.GetPrimAtPath(path)
        base = sanitize_name(prim.GetName())
        name = base
        counter = 2
        while name in used_link_names:
            name = f"{base}_{counter}"
            counter += 1
        used_link_names.add(name)
        link_names[path] = name

    link_frames = build_link_rigid_frames(
        stage,
        link_paths,
        incoming_joint,
        root_paths,
        up_axis,
    ) if link_paths else {}

    meshes_by_link, unowned_meshes = assign_meshes_to_links(stage, link_paths)

    static_link_name = None
    static_frame = None
    if unowned_meshes:
        base = "scene_static"
        static_link_name = base
        counter = 2
        while static_link_name in used_link_names:
            static_link_name = f"{base}_{counter}"
            counter += 1
        used_link_names.add(static_link_name)
        static_frame = rigid_frame_from_affine(
            Gf.Matrix4d(1.0),
            up_axis=up_axis,
            preferred_axis=2,
        )
        static_frame["source"] = "usd_world"

    diag_names = dict(link_names)
    diag_frames = dict(link_frames)
    if static_link_name:
        static_key = Sdf.Path("/__scene_static__")
        diag_names[static_key] = static_link_name
        diag_frames[static_key] = static_frame
    print_affine_diagnostics(diag_names, diag_frames)

    xform_cache = UsdGeom.XformCache()
    anchor_report = []

    if DEBUG_JOINTS and joints:
        print("\n================ JOINT ANCHOR CHECK ================")

    for joint in joints:
        result = compute_anchor_error(
            stage,
            joint,
            meters_per_unit,
            xform_cache,
        )
        if result is None:
            continue
        anchor_report.append({"joint": sanitize_name(joint["name"]), **result})
        if DEBUG_JOINTS:
            print(
                f"{joint['name']:45s} "
                f"error={result['error_m']:.6f} m  "
                f"delta=({result['delta_m'][0]:.6f}, "
                f"{result['delta_m'][1]:.6f}, "
                f"{result['delta_m'][2]:.6f})"
            )

    if DEBUG_JOINTS and joints:
        print("====================================================\n")

    min_z = compute_min_z_in_world(stage, meters_per_unit)
    ground_offset_z = -min_z if AUTO_GROUND else 0.0

    print(f"[INFO] Unowned/static visual meshes: {len(unowned_meshes)}")
    print(f"[INFO] Lowest geometry Z = {min_z:.6f} m")
    print(f"[INFO] Ground offset Z   = {ground_offset_z:.6f} m")

    robot_name = sanitize_name(usd_path.stem)
    robot = ET.Element("robot", {"name": robot_name})

    if not ADD_WORLD_LINK and (len(root_paths) + (1 if static_link_name else 0)) != 1:
        raise RuntimeError("Multi-root scenes require ADD_WORLD_LINK = True.")

    world_frame = rigid_frame_from_affine(
        Gf.Matrix4d(1.0),
        up_axis=up_axis,
        preferred_axis=2,
    )

    if ADD_WORLD_LINK:
        ET.SubElement(robot, "link", {"name": "world"})

    report = {
        "converter": "multi-root-affine-obj-mtl-material-aware",
        "input_usd": str(usd_path),
        "meters_per_unit": meters_per_unit,
        "up_axis": up_axis,
        "frame_strategy": "pure_rigid_urdf_frames_plus_full_affine_mesh_baking",
        "zero_pose_policy": "authored_usd_body1_joint_pose",
        "scene_mode": True,
        "root_link": "world" if ADD_WORLD_LINK else None,
        "component_roots": [link_names[p] for p in root_paths],
        "static_scene_link": static_link_name,
        "ground_offset_z": ground_offset_z,
        "links": {},
        "materials": {},
        "joints": [],
        "world_joints_skipped": [j["path"] for j in usd_world_joints],
        "joint_anchor_errors": anchor_report,
        "warnings": [],
        "defaults": {
            "effort": default_effort,
            "velocity": default_velocity,
        },
    }

    for anchor in anchor_report:
        if anchor["error_m"] > ANCHOR_WARNING_TOLERANCE_M:
            report["warnings"].append(
                f"USD joint anchor mismatch on {anchor['joint']}: "
                f"{anchor['error_m']:.6f} m."
            )

    material_cache = {}

    export_items = [
        (path, link_names[path], meshes_by_link.get(path, []), link_frames[path])
        for path in sorted(link_paths, key=lambda p: str(p))
    ]
    if static_link_name:
        export_items.append((None, static_link_name, unowned_meshes, static_frame))

    for link_path, link_name, mesh_prims, frame in export_items:
        visual_records, visual_triangle_count = export_link_visual_objs(
            stage=stage,
            mesh_prims=mesh_prims,
            link_rigid_frame=frame,
            meters_per_unit=meters_per_unit,
            visual_dir=visual_dir,
            texture_dir=texture_dir,
            link_name=link_name,
            material_cache=material_cache,
            warning_sink=report["warnings"],
        )

        collision_uri = None
        collision_path = None
        collision_triangle_count = 0

        if mesh_prims:
            collision_path = collision_dir / f"{link_name}.stl"
            collision_triangle_count = export_link_collision_stl(
                stage=stage,
                mesh_prims=mesh_prims,
                link_rigid_frame=frame,
                meters_per_unit=meters_per_unit,
                output_path=str(collision_path),
                solid_name=link_name,
            )
            collision_uri = collision_path.resolve().as_uri()

        add_link_xml(robot, link_name, visual_records, collision_uri)

        affine_info = affine_summary_for_json(frame)
        print(
            f"[LINK] {link_name}: {len(mesh_prims)} mesh prim(s), "
            f"{visual_triangle_count} visual triangles, "
            f"{len(visual_records)} material visual(s)"
        )

        report["links"][link_name] = {
            "usd_path": str(link_path) if link_path is not None else None,
            "visual_meshes": [
                {
                    "mesh": rec["mesh"],
                    "triangle_count": rec["triangle_count"],
                    "material": rec["material"]["key"],
                }
                for rec in visual_records
            ],
            "collision_mesh": str(collision_path) if collision_path else None,
            "mesh_prim_count": len(mesh_prims),
            "visual_triangle_count": visual_triangle_count,
            "collision_triangle_count": collision_triangle_count,
            "source_frame_affine": affine_info,
        }

        if not affine_info["is_rigid"]:
            report["warnings"].append(
                f"Link {link_name} source frame is affine/non-rigid; residual affine was baked into mesh vertices."
            )

    for key, mat in material_cache.items():
        report["materials"][key] = mat
        for warning in mat.get("warnings", []):
            report["warnings"].append(f"Material {mat['name']}: {warning}")

    used_joint_names = set()

    def unique_joint_name(base):
        base = sanitize_name(base)
        name = base
        counter = 2
        while name in used_joint_names:
            name = f"{base}_{counter}"
            counter += 1
        used_joint_names.add(name)
        return name

    if ADD_WORLD_LINK:
        for root_path in root_paths:
            root_name = link_names[root_path]
            joint_name = unique_joint_name(f"world_to_{root_name}")
            joint_xml = ET.SubElement(robot, "joint", {"name": joint_name, "type": "fixed"})
            ET.SubElement(joint_xml, "parent", {"link": "world"})
            ET.SubElement(joint_xml, "child", {"link": root_name})

            xyz, rpy = relative_frame_to_xyz_rpy(
                world_frame,
                link_frames[root_path],
                meters_per_unit,
            )
            xyz = (xyz[0], xyz[1], xyz[2] + ground_offset_z)
            ET.SubElement(
                joint_xml,
                "origin",
                {
                    "xyz": f"{xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g}",
                    "rpy": f"{rpy[0]:.12g} {rpy[1]:.12g} {rpy[2]:.12g}",
                },
            )
            report["joints"].append({
                "name": joint_name,
                "type": "fixed",
                "parent": "world",
                "child": root_name,
                "origin_xyz_m": list(xyz),
                "origin_rpy_rad": list(rpy),
                "source": "scene_root",
            })

        if static_link_name:
            joint_name = unique_joint_name(f"world_to_{static_link_name}")
            joint_xml = ET.SubElement(robot, "joint", {"name": joint_name, "type": "fixed"})
            ET.SubElement(joint_xml, "parent", {"link": "world"})
            ET.SubElement(joint_xml, "child", {"link": static_link_name})
            ET.SubElement(
                joint_xml,
                "origin",
                {
                    "xyz": f"0 0 {ground_offset_z:.12g}",
                    "rpy": "0 0 0",
                },
            )
            report["joints"].append({
                "name": joint_name,
                "type": "fixed",
                "parent": "world",
                "child": static_link_name,
                "origin_xyz_m": [0.0, 0.0, ground_offset_z],
                "origin_rpy_rad": [0.0, 0.0, 0.0],
                "source": "scene_static",
            })

    for joint in joints:
        parent_path = joint["body0"]
        child_path = joint["body1"]
        parent_name = link_names[parent_path]
        child_name = link_names[child_path]
        joint_name = unique_joint_name(joint["name"])

        joint_xml = ET.SubElement(
            robot,
            "joint",
            {"name": joint_name, "type": joint["type"]},
        )
        ET.SubElement(joint_xml, "parent", {"link": parent_name})
        ET.SubElement(joint_xml, "child", {"link": child_name})

        xyz, rpy = relative_frame_to_xyz_rpy(
            link_frames[parent_path],
            link_frames[child_path],
            meters_per_unit,
        )
        ET.SubElement(
            joint_xml,
            "origin",
            {
                "xyz": f"{xyz[0]:.12g} {xyz[1]:.12g} {xyz[2]:.12g}",
                "rpy": f"{rpy[0]:.12g} {rpy[1]:.12g} {rpy[2]:.12g}",
            },
        )

        report_joint = {
            "name": joint_name,
            "type": joint["type"],
            "parent": parent_name,
            "child": child_name,
            "origin_xyz_m": list(xyz),
            "origin_rpy_rad": list(rpy),
            "zero_pose_policy": "authored_usd_pose",
        }

        if joint["type"] in ("revolute", "prismatic"):
            axis, axis_info = compute_urdf_joint_axis(
                stage,
                joint,
                link_frames[child_path],
                xform_cache,
            )
            ET.SubElement(
                joint_xml,
                "axis",
                {"xyz": f"{axis[0]:.12g} {axis[1]:.12g} {axis[2]:.12g}"},
            )

            lower = float(joint.get("lower", 0.0) or 0.0)
            upper = float(joint.get("upper", 0.0) or 0.0)
            if joint["type"] == "revolute":
                lower = math.radians(lower)
                upper = math.radians(upper)
            else:
                lower *= meters_per_unit
                upper *= meters_per_unit

            ET.SubElement(
                joint_xml,
                "limit",
                {
                    "lower": f"{lower:.12g}",
                    "upper": f"{upper:.12g}",
                    "effort": f"{default_effort:.12g}",
                    "velocity": f"{default_velocity:.12g}",
                },
            )

            report_joint["axis"] = list(axis)
            report_joint["axis_source"] = axis_info
            report_joint["lower"] = lower
            report_joint["upper"] = upper

            if DEBUG_JOINTS and dot(axis, axis_to_xyz(joint["axis"])) < 0.0:
                print(f"[AXIS] {joint_name}: {joint['axis']} -> {axis}")

        report["joints"].append(report_joint)
        print(f"[JOINT] {joint_name}: {parent_name} -> {child_name} ({joint['type']})")

    urdf_path = output_dir / "model.urdf"
    report_path = output_dir / "conversion_report.json"

    with open(urdf_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml(robot))

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    print(f"URDF      : {urdf_path}")
    print(f"Visuals   : {visual_dir}")
    print(f"Collision : {collision_dir}")
    print(f"Textures  : {texture_dir}")
    print(f"Report    : {report_path}")


# ============================================================
# RUN
# ============================================================

def main():
    convert(
        usd_path=USD_PATH,
        output_dir=OUTPUT_DIR,
        default_effort=DEFAULT_EFFORT,
        default_velocity=DEFAULT_VELOCITY,
    )


if __name__ == "__main__":
    main()
