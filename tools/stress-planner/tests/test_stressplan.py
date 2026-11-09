import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stressplan import fea  # noqa: E402
from stressplan.assembly import Assembly, Mate, Part, frame_from_axes, mesh_from_stl_bytes, write_stl  # noqa: E402
from stressplan.demo import box  # noqa: E402
from stressplan.export import write_3mf  # noqa: E402
from stressplan.onshape import ElementRef, OnshapeClient, assembly_from_definition  # noqa: E402
from stressplan.planner import PlanOptions, Surface, failure_index, plan_part  # noqa: E402
from stressplan.statics import LoadCase, Material, PointLoad, Wrench, solve_statics  # noqa: E402

PLA = Material.preset("PLA")


def cantilever(load=100.0, res=45):
    V, F = box((0, -10, -10), (200, 10, 10))
    mesh = fea.tet_mesh(V, F, resolution=res)
    M_root = np.cross([200.0, 0, 0], [0, 0, -load])
    ws = [Wrench(np.array([200.0, 0, 0]), np.array([0, 0, -load]), np.zeros(3), "load:tip", 15.0),
          Wrench(np.zeros(3), np.array([0, 0, load]), -M_root, "ground", 15.0)]
    return mesh, fea.solve(mesh, PLA, ws)


def test_cantilever_bending_stress_matches_beam_theory():
    mesh, r = cantilever()
    assert r.imbalance < 1e-6
    I = 20 ** 4 / 12
    slab = (r.centroids[:, 0] > 80) & (r.centroids[:, 0] < 120)
    z, sxx, xm = r.centroids[slab, 2], r.stress[slab, 0, 0], r.centroids[slab, 0]
    # sigma_xx = M(x) z / I with M(x) = F (200 - x)
    expected = 100 * (200 - xm) * z / I
    slope = np.polyfit(expected, sxx, 1)[0]
    assert 0.85 < slope < 1.1, slope
    # Away from the load patches the stress field is tension on top, compression underneath.
    assert sxx[z > 7].mean() > 0 > sxx[z < -7].mean()


def test_failure_index_prefers_layers_along_tension():
    s = np.zeros((1, 3, 3))
    s[0, 0, 0] = 20.0  # 20 MPa uniaxial tension along X
    fi = failure_index(s, np.array([20.0]), np.array([[1.0, 0, 0], [0, 0, 1.0]]), PLA)[0]
    assert fi[0] == pytest.approx(20 / PLA.S_z)    # building along X: tension peels layers
    assert fi[1] == pytest.approx(20 / PLA.S_xy)   # building along Z: tension stays in-layer


def test_statics_reactions_and_mechanism_detection():
    pV, pF = box((-5, -5, -20), (5, 5, 0))
    aV, aF = box((0, -5, -5), (100, 5, 5))
    post = Part("post", "Post", pV, pF, fixed=True)
    arm = Part("arm", "Arm", aV, aF)
    fastened = Mate("joint", "FASTENED", "post", "arm", np.zeros(3), frame_from_axes([0, 1, 0]))
    asm = Assembly([post, arm], [fastened])
    lc = LoadCase(gravity=None, loads=[PointLoad("arm", np.array([100.0, 0, 0]),
                                                 np.array([0, 0, -100.0]), np.zeros(3))])
    st = solve_statics(asm, lc, log=lambda *a: None)
    on_arm = [w for w in st.wrenches["arm"] if w.source == "mate:joint"][0]
    np.testing.assert_allclose(on_arm.force, [0, 0, 100], atol=1e-6)
    np.testing.assert_allclose(on_arm.moment, [0, -10000, 0], atol=1e-4)
    assert not st.unbalanced

    # A revolute about Y cannot carry that moment: the arm swings, and we must say so.
    asm.mates[0] = Mate("joint", "REVOLUTE", "post", "arm", np.zeros(3), frame_from_axes([0, 1, 0]))
    st = solve_statics(asm, lc, log=lambda *a: None)
    assert "Arm" in st.unbalanced


def test_planner_reorients_upright_beam_in_bending():
    # A post modelled standing up, pushed sideways at the top: printed as modelled the bending
    # tension crosses every layer.
    V, F = box((-6, -6, 0), (6, 6, 80))
    mesh = fea.tet_mesh(V, F, resolution=35)
    top = np.array([0, 0, 80.0])
    Fx = np.array([30.0, 0, 0])
    ws = [Wrench(top, Fx, np.zeros(3), "load:push", 6.0),
          Wrench(np.zeros(3), -Fx, -np.cross(top, Fx), "ground", None)]
    r = fea.solve(mesh, PLA, ws)
    plan = plan_part(r, Surface.from_mesh(V, F), PLA, PlanOptions(max_planes=0), log=lambda *a: None)
    d = np.array(plan.metrics["build_dir_asm"])
    assert abs(d[2]) < 0.3, d                                   # lies the post down
    assert plan.metrics["sf_best_3axis"] > 1.5 * plan.metrics["sf_as_modelled"]


def test_upright_constraint_yields_cut_planes_that_keep_bed_on_root():
    from stressplan.demo import DEMO_LOADS, demo_assembly

    asm = demo_assembly()
    lc = LoadCase.from_dict(DEMO_LOADS, asm)
    st = solve_statics(asm, lc, log=lambda *a: None)
    p = asm.part("Bracket")
    mesh = fea.tet_mesh(p.vertices, p.faces, resolution=25)
    r = fea.solve(mesh, lc.material, st.wrenches[p.id], st.gravity[p.id])
    plan = plan_part(r, Surface.from_mesh(p.vertices, p.faces), lc.material,
                     PlanOptions(max_planes=2, keep_orientation=True, target_sf=0), log=lambda *a: None)
    assert plan.planes, "expected at least one cut plane"
    assert plan.metrics["sf_plan"] > plan.metrics["sf_as_modelled"] * 1.1
    V = p.vertices @ plan.R.T
    bed = V[V[:, 2] <= V[:, 2].min() + 0.05]
    for pl in plan.planes:
        assert ((bed - pl["point"]) @ pl["normal"] < 0).all()  # lowest vertices stay on the root
        assert pl["normal"][2] > 0                              # builds upward-ish


def test_3mf_cut_planes_match_slicer_schema(tmp_path):
    V, F = box((-5, -5, -5), (5, 5, 5))
    planes = [{"point": np.array([0, 0, 1.0]), "normal": np.array([0, 0.6, 0.8]),
               "build_normal": np.array([0, 0.6, 0.8])},
              {"point": np.array([0, 0, 2.0]), "normal": np.array([0.6, 0, 0.8]),
               "build_normal": np.array([0, 0, 1.0])}]
    out = write_3mf(tmp_path / "t.3mf", [
        {"name": "A", "vertices": V, "faces": F, "planes": planes, "position": (10, 10, 5)},
        {"name": "B", "vertices": V, "faces": F, "planes": [], "position": (40, 10, 5)},
    ])
    with zipfile.ZipFile(out) as z:
        names = set(z.namelist())
        assert {"[Content_Types].xml", "_rels/.rels", "3D/3dmodel.model", "Metadata/cut_planes.xml"} <= names
        cp = ET.fromstring(z.read("Metadata/cut_planes.xml"))
        model = ET.fromstring(z.read("3D/3dmodel.model"))
    objs = cp.findall("object")
    assert [o.get("id") for o in objs] == ["1"]          # object B has no planes -> omitted
    assert objs[0].get("enabled") == "1" and objs[0].get("user_ordered") == "1"
    pls = objs[0].findall("plane")
    assert float(pls[0].get("pz")) == 2.0                # newest plane first (peels last)
    assert pls[0].get("bnz") == "1" and pls[1].get("bnx") is None
    ns = {"m": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}
    assert len(model.findall(".//m:object", ns)) == 2
    assert len(model.findall(".//m:item", ns)) == 2


def _stl_bytes(V, F, tmp_path):
    write_stl(tmp_path / "x.stl", V, F)
    return (tmp_path / "x.stl").read_bytes()


def test_onshape_definition_to_assembly(tmp_path):
    V, F = box((0, 0, 0), (10, 10, 10))
    stl = _stl_bytes(V, F, tmp_path)
    ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    shifted = [1, 0, 0, 0.1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]  # +100 mm in X
    d = dict(documentId="d" * 24, documentMicroversion="m" * 24, elementId="e" * 24)
    defn = {
        "rootAssembly": {
            "instances": [
                {"id": "IA", "type": "Part", "name": "Base", "partId": "JHD", **d},
                {"id": "IS", "type": "Assembly", "name": "Sub", **d, "elementId": "s" * 24},
            ],
            "occurrences": [
                {"path": ["IA"], "transform": ident, "fixed": True},
                {"path": ["IS"], "transform": shifted},
                {"path": ["IS", "IB"], "transform": shifted},
            ],
            "features": [{
                "featureType": "mate", "id": "f1",
                "featureData": {"name": "Pin", "mateType": "REVOLUTE", "matedEntities": [
                    {"matedOccurrence": ["IA"], "matedCS": {"origin": [0.01, 0, 0], "xAxis": [1, 0, 0],
                                                            "yAxis": [0, 1, 0], "zAxis": [0, 0, 1]}},
                    {"matedOccurrence": ["IS", "IB"], "matedCS": {"origin": [-0.09, 0, 0], "xAxis": [1, 0, 0],
                                                                  "yAxis": [0, 1, 0], "zAxis": [0, 0, 1]}},
                ]},
            }],
        },
        "subAssemblies": [{**d, "elementId": "s" * 24, "instances": [
            {"id": "IB", "type": "Part", "name": "Arm", "partId": "JKD", **d}], "features": []}],
        "parts": [],
    }
    asm = assembly_from_definition(defn, lambda p: stl, log=lambda *a: None)
    base, arm = asm.part("Base"), asm.part("Arm")
    assert base.fixed and not arm.fixed
    np.testing.assert_allclose(arm.vertices.min(0), [100, 0, 0], atol=1e-4)
    m = asm.mates[0]
    assert m.type == "REVOLUTE" and {m.part_a, m.part_b} == {base.id, arm.id}
    np.testing.assert_allclose(m.origin, [10, 0, 0], atol=1e-6)


def test_url_parsing_and_key_scoping():
    ref = ElementRef.from_url("https://cad.onshape.com/documents/0123456789abcdef01234567/w/"
                              "89abcdef0123456789abcdef/e/fedcba9876543210fedcba98")
    assert ref.wvm == "w" and ref.path().startswith("/d/0123")
    with pytest.raises(ValueError):
        ElementRef.from_url("https://example.com/foo")
    c = OnshapeClient("a", "b")
    assert c._trusted("https://cad.onshape.com/api/x")
    assert c._trusted("https://some-region.onshape.com/x")
    assert not c._trusted("https://evil.example/x")
    assert not c._trusted("https://onshape.com.evil.example/x")


def test_bundle_roundtrip(tmp_path):
    from stressplan.demo import demo_assembly

    asm = demo_assembly()
    asm.save(tmp_path / "b")
    back = Assembly.load(tmp_path / "b")
    assert [p.name for p in back.parts] == [p.name for p in asm.parts]
    assert [m.type for m in back.mates] == [m.type for m in asm.mates]
    np.testing.assert_allclose(back.part("Bracket").volume(), asm.part("Bracket").volume(), rtol=1e-5)
