"""Actuator grouping is pure model introspection - no GL, no server."""

import mujoco
import pytest

from mujoco_visualizer.serve.controls import actuator_group_map, build_control_tree

# Two legs' worth of T-segment naming plus a wing, an abdomen and an unsided actuator,
# so every branch of the grouping rule is exercised without the 139 MB fly assets.
_XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 2"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="root" pos="0 0 0.5">
      <joint name="j_coxa_T1_left"  type="hinge" axis="0 1 0"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
      <body name="b1" pos="0.1 0 0">
        <joint name="j_coxa_T1_right" type="hinge" axis="0 1 0"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
        <body name="b2" pos="0.1 0 0">
          <joint name="j_femur_T2_left" type="hinge" axis="0 1 0"/>
          <geom name="g2" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
          <body name="b3" pos="0.1 0 0">
            <joint name="j_wing" type="hinge" axis="0 1 0"/>
            <geom name="g3" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
            <body name="b4" pos="0.1 0 0">
              <joint name="j_abd" type="hinge" axis="0 1 0"/>
              <geom name="g4" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
              <body name="b5" pos="0.1 0 0">
                <joint name="j_misc" type="hinge" axis="0 1 0"/>
                <geom name="g5" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="coxa_T1_left"   joint="j_coxa_T1_left"   ctrlrange="-1 1"/>
    <motor name="coxa_T1_right"  joint="j_coxa_T1_right"  ctrlrange="-1 1"/>
    <motor name="femur_T2_left"  joint="j_femur_T2_left"  ctrlrange="-2 2"/>
    <motor name="wing_yaw_left"  joint="j_wing"           ctrlrange="-1 1"/>
    <motor name="abdomen_abduct" joint="j_abd"            ctrlrange="-1 1"/>
    <motor name="adhere_labrum_right" joint="j_misc"      ctrlrange="0 1"/>
    <motor name="sensor_left"    joint="j_coxa_T1_left"   />
  </actuator>
</mujoco>
"""


@pytest.fixture
def model():
    return mujoco.MjModel.from_xml_string(_XML)


def test_every_actuator_appears_exactly_once(model):
    tree = build_control_tree(model)
    ids = [a["id"] for g in tree["groups"] for a in g["actuators"]]
    assert sorted(ids) == list(range(model.nu))


def test_group_ids_follow_the_scheme(model):
    tree = build_control_tree(model)
    got = {g["id"]: [a["name"] for a in g["actuators"]] for g in tree["groups"]}
    assert got["leg.T1.left"] == ["coxa_T1_left"]
    assert got["leg.T1.right"] == ["coxa_T1_right"]
    assert got["leg.T2.left"] == ["femur_T2_left"]
    assert got["wing.left"] == ["wing_yaw_left"]
    assert got["abdomen"] == ["abdomen_abduct"]
    assert got["other.right"] == ["adhere_labrum_right"]
    assert got["other.left"] == ["sensor_left"]


def test_ctrlrange_is_carried_through(model):
    tree = build_control_tree(model)
    by_name = {a["name"]: a for g in tree["groups"] for a in g["actuators"]}
    assert by_name["femur_T2_left"]["lo"] == pytest.approx(-2.0)
    assert by_name["femur_T2_left"]["hi"] == pytest.approx(2.0)
    assert by_name["femur_T2_left"]["limited"] is True


def test_ctrlrange_fallback_when_unlimited(model):
    """When an actuator has no ctrlrange, lo/hi default to -1/1 and limited is False."""
    tree = build_control_tree(model)
    by_name = {a["name"]: a for g in tree["groups"] for a in g["actuators"]}
    assert by_name["sensor_left"]["lo"] == pytest.approx(-1.0)
    assert by_name["sensor_left"]["hi"] == pytest.approx(1.0)
    assert by_name["sensor_left"]["limited"] is False


def test_group_map_inverts_the_tree(model):
    tree = build_control_tree(model)
    gmap = actuator_group_map(tree)
    assert len(gmap) == model.nu
    assert gmap[0] == "leg.T1.left"


def test_empty_groups_are_omitted(model):
    tree = build_control_tree(model)
    assert all(g["actuators"] for g in tree["groups"])
    assert "leg.T3.left" not in {g["id"] for g in tree["groups"]}
