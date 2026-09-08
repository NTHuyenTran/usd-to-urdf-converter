#!/usr/bin/env python3

from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


# ============================================================
# CONFIG
# ============================================================

USD_PATH = ("/path/to/input/model.usd")

OUTPUT_DIR = Path("/path/to/output")

# ============================================================
# HELPERS
# ============================================================

def get_visibility(prim):
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)

        try:
            return str(
                imageable.ComputeVisibility(
                    Usd.TimeCode.Default()
                )
            )
        except Exception:
            pass

    return "-"


def get_purpose(prim):
    if prim.IsA(UsdGeom.Imageable):
        imageable = UsdGeom.Imageable(prim)

        try:
            return str(
                imageable.ComputePurpose()
            )
        except Exception:
            pass

    return "-"


def get_xform(stage, prim):
    cache = UsdGeom.XformCache(
        Usd.TimeCode.Default()
    )

    try:
        return cache.GetLocalToWorldTransform(prim)
    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main():

    usd_path = Path(USD_PATH).resolve()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("USD INSPECTOR")
    print("=" * 80)

    print("Input:")
    print(usd_path)

    stage = Usd.Stage.Open(
        str(usd_path)
    )

    if stage is None:
        raise RuntimeError(
            f"Cannot open USD: {usd_path}"
        )

    # --------------------------------------------------------
    # Stage metadata
    # --------------------------------------------------------

    print()
    print("STAGE METADATA")
    print("-" * 80)

    print(
        "metersPerUnit:",
        UsdGeom.GetStageMetersPerUnit(stage),
    )

    print(
        "upAxis:",
        UsdGeom.GetStageUpAxis(stage),
    )

    print(
        "defaultPrim:",
        stage.GetDefaultPrim().GetPath()
        if stage.GetDefaultPrim()
        else None,
    )

    print(
        "rootLayer:",
        stage.GetRootLayer().identifier,
    )

    # --------------------------------------------------------
    # Export ROOT LAYER
    # --------------------------------------------------------

    root_layer_path = (
        OUTPUT_DIR /
        "01_root_layer.usda"
    )

    stage.GetRootLayer().Export(
        str(root_layer_path)
    )

    print()
    print(
        "[SAVED] Root layer:",
        root_layer_path,
    )

    # --------------------------------------------------------
    # FLATTEN composed USD
    # --------------------------------------------------------

    flattened_path = (
        OUTPUT_DIR /
        "02_flattened_stage.usda"
    )

    flattened = stage.Flatten()

    flattened.Export(
        str(flattened_path)
    )

    print(
        "[SAVED] Flattened stage:",
        flattened_path,
    )

    # --------------------------------------------------------
    # PRIM TREE
    # --------------------------------------------------------

    report_path = (
        OUTPUT_DIR /
        "03_prim_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as out:

        for prim in stage.Traverse():

            path = prim.GetPath()
            type_name = prim.GetTypeName()

            visibility = get_visibility(
                prim
            )

            purpose = get_purpose(
                prim
            )

            out.write(
                "=" * 100 + "\n"
            )

            out.write(
                f"PATH       : {path}\n"
            )

            out.write(
                f"TYPE       : {type_name}\n"
            )

            out.write(
                f"ACTIVE     : {prim.IsActive()}\n"
            )

            out.write(
                f"DEFINED    : {prim.IsDefined()}\n"
            )

            out.write(
                f"INSTANCE   : {prim.IsInstance()}\n"
            )

            out.write(
                f"VISIBILITY : {visibility}\n"
            )

            out.write(
                f"PURPOSE    : {purpose}\n"
            )

            # ----------------------------------------------
            # Rigid body
            # ----------------------------------------------

            rigid_api = UsdPhysics.RigidBodyAPI(
                prim
            )

            if rigid_api:

                enabled = (
                    rigid_api
                    .GetRigidBodyEnabledAttr()
                    .Get()
                )

                out.write(
                    "RIGID BODY : YES\n"
                )

                out.write(
                    f"RB ENABLED : {enabled}\n"
                )

            # ----------------------------------------------
            # Collision
            # ----------------------------------------------

            collision_api = (
                UsdPhysics.CollisionAPI(
                    prim
                )
            )

            if collision_api:

                enabled = (
                    collision_api
                    .GetCollisionEnabledAttr()
                    .Get()
                )

                out.write(
                    "COLLISION  : YES\n"
                )

                out.write(
                    f"COL ENABLED: {enabled}\n"
                )

            # ----------------------------------------------
            # Joint
            # ----------------------------------------------

            if prim.IsA(
                UsdPhysics.Joint
            ):

                joint = UsdPhysics.Joint(
                    prim
                )

                out.write(
                    "JOINT      : YES\n"
                )

                out.write(
                    f"BODY0      : "
                    f"{joint.GetBody0Rel().GetTargets()}\n"
                )

                out.write(
                    f"BODY1      : "
                    f"{joint.GetBody1Rel().GetTargets()}\n"
                )

                out.write(
                    f"LOCAL POS0 : "
                    f"{joint.GetLocalPos0Attr().Get()}\n"
                )

                out.write(
                    f"LOCAL POS1 : "
                    f"{joint.GetLocalPos1Attr().Get()}\n"
                )

                out.write(
                    f"LOCAL ROT0 : "
                    f"{joint.GetLocalRot0Attr().Get()}\n"
                )

                out.write(
                    f"LOCAL ROT1 : "
                    f"{joint.GetLocalRot1Attr().Get()}\n"
                )

                if prim.IsA(
                    UsdPhysics.RevoluteJoint
                ):

                    j = (
                        UsdPhysics
                        .RevoluteJoint(prim)
                    )

                    out.write(
                        f"JOINT TYPE : REVOLUTE\n"
                    )

                    out.write(
                        f"AXIS       : "
                        f"{j.GetAxisAttr().Get()}\n"
                    )

                    out.write(
                        f"LOWER      : "
                        f"{j.GetLowerLimitAttr().Get()}\n"
                    )

                    out.write(
                        f"UPPER      : "
                        f"{j.GetUpperLimitAttr().Get()}\n"
                    )

                elif prim.IsA(
                    UsdPhysics.PrismaticJoint
                ):

                    j = (
                        UsdPhysics
                        .PrismaticJoint(prim)
                    )

                    out.write(
                        "JOINT TYPE : PRISMATIC\n"
                    )

                    out.write(
                        f"AXIS       : "
                        f"{j.GetAxisAttr().Get()}\n"
                    )

                    out.write(
                        f"LOWER      : "
                        f"{j.GetLowerLimitAttr().Get()}\n"
                    )

                    out.write(
                        f"UPPER      : "
                        f"{j.GetUpperLimitAttr().Get()}\n"
                    )

                elif prim.IsA(
                    UsdPhysics.FixedJoint
                ):

                    out.write(
                        "JOINT TYPE : FIXED\n"
                    )

            # ----------------------------------------------
            # Mesh
            # ----------------------------------------------

            if prim.IsA(
                UsdGeom.Mesh
            ):

                mesh = UsdGeom.Mesh(
                    prim
                )

                points = (
                    mesh
                    .GetPointsAttr()
                    .Get()
                )

                counts = (
                    mesh
                    .GetFaceVertexCountsAttr()
                    .Get()
                )

                out.write(
                    "MESH       : YES\n"
                )

                out.write(
                    f"POINTS     : "
                    f"{len(points) if points else 0}\n"
                )

                out.write(
                    f"FACES      : "
                    f"{len(counts) if counts else 0}\n"
                )

            # ----------------------------------------------
            # WORLD TRANSFORM
            # ----------------------------------------------

            M = get_xform(
                stage,
                prim,
            )

            if M is not None:

                out.write(
                    "WORLD MATRIX:\n"
                )

                out.write(
                    str(M) + "\n"
                )

            # ----------------------------------------------
            # Composition stack
            # ----------------------------------------------

            stack = prim.GetPrimStack()

            if stack:

                out.write(
                    "PRIM STACK:\n"
                )

                for spec in stack:

                    out.write(
                        "  layer = "
                        f"{spec.layer.identifier}\n"
                    )

                    out.write(
                        "  path  = "
                        f"{spec.path}\n"
                    )

    print(
        "[SAVED] Prim report:",
        report_path,
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()