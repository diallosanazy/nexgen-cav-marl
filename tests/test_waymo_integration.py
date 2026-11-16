import unittest

from nexgen.waymo import MotionState, Scenario, VehicleTrack
from nexgen_sim import Simulator, WAIT
from nexgen_sim.mobility import ShortestPathReference, schedule_snapshots
from nexgen_sim.waymo_adapter import waymo_snapshots
from nexgen_sim.waymo_run import run_scenario
from nexgen_sim.traffic import Stream


def fixture(timestamps=(100.0, 100.1, 100.23)):
    return Scenario("fixture", timestamps,
                    (VehicleTrack(12, tuple(MotionState(True, 0, 0, 0, 0) for _ in timestamps)),
                     VehicleTrack(99, tuple(MotionState(True, x, 0, 0, 0) for x in (20, 5, 5)))),
                    2, 3, 3, 3, 0)


class WaymoIntegrationTests(unittest.TestCase):
    def test_actual_timestamps_and_position_metadata(self):
        frames = waymo_snapshots(fixture(), [12, 99], radius=10)
        self.assertEqual([s.at for s in frames], [0, 100000, 230000])
        self.assertEqual(frames[0].topology.neighbors("12"), ())
        self.assertEqual(frames[1].topology.neighbors("12"), ("99",))
        self.assertEqual(frames[1].topology.node_attributes("99"),
                         {"position": [5.0, 0.0], "waymo_track_id": 99})

    def test_waymo_movement_drives_delivery(self):
        frames = waymo_snapshots(fixture(), [12, 99], radius=10)
        sim = Simulator.from_topology(frames[0].topology, lambda _: WAIT, wait_delay=50000)
        sim.policy = ShortestPathReference(lambda: sim.topology)
        schedule_snapshots(sim, frames)
        sim.generate("M", "12", "99", deadline=200000)
        sim.run(230000)
        self.assertEqual(sim.messages["M"].delivered_at, 101000)
        self.assertEqual(sim.messages["M"].history, ["12", "99"])

    def test_timestamp_collapse_rejected(self):
        with self.assertRaises(ValueError):
            waymo_snapshots(fixture((0, 1e-9, 1e-8)), [12, 99], radius=10)

    def test_runner_reproducibility_and_fixed_deadlines(self):
        options = dict(cohort_size=2, radius=10,
                       streams=[Stream("s", "12", "99", 100, 200000, 0, 230000)])
        first = run_scenario(fixture(), **options)
        second = run_scenario(fixture(), **options)
        self.assertEqual(first[0].log, second[0].log)
        self.assertEqual(first[0].metrics(), second[0].metrics())
        self.assertTrue(first[4]["positions_changed"])
        self.assertEqual(first[4]["directed_link_changes"], 2)
        for message in first[0].messages.values():
            self.assertEqual(message.deadline - message.generated_at, 200000)

    def test_traffic_cannot_outlive_trajectory_window(self):
        with self.assertRaises(ValueError):
            run_scenario(fixture(), cohort_size=2,
                         streams=[Stream("s", "12", "99", 1, 500, 0, 500000)])
