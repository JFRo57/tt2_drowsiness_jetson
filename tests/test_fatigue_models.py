import unittest
import numpy as np
from src.fatigue_models import FatigueModelPackage

class FakeNet(object):
    def __init__(self, output): self.output, self.input_shape = np.asarray(output, dtype=np.float32), None
    def setInput(self, value): self.input_shape = value.shape
    def forward(self): return self.output

class FatigueModelPackageTests(unittest.TestCase):
    def test_manifest_shapes_and_outputs(self):
        package = FatigueModelPackage({"fatigue_models": {"enabled": False}}); package.enabled = True
        package.nets = {"eye_state": FakeNet([[3., 0.]]), "yawn_state": FakeNet([[0., 3.]]), "head_pose": FakeNet([[10., 20., 30.]])}
        frame = np.full((160, 200, 3), 127, dtype=np.uint8); landmarks = np.zeros((68, 2), dtype=np.int32)
        landmarks[36:42] = [[40,50],[45,45],[55,45],[60,50],[55,55],[45,55]]
        landmarks[42:48] = [[90,50],[95,45],[105,45],[110,50],[105,55],[95,55]]
        landmarks[48:68] = [[70+i*2, 90+(i%3)] for i in range(20)]
        result = package.analyze(frame, landmarks, (25,20,135,140))
        self.assertTrue(result["model_eyes_closed"]); self.assertTrue(result["model_yawning"])
        self.assertEqual((1,1,64,64), package.nets["eye_state"].input_shape)
        self.assertEqual((1,1,64,96), package.nets["yawn_state"].input_shape)
        self.assertEqual((1,3,128,128), package.nets["head_pose"].input_shape)
        self.assertEqual(20., result["model_yaw"])

    def test_failure_disables_only_failed_model(self):
        class BrokenNet(FakeNet):
            def forward(self): raise RuntimeError("modelo incompatible")
        package = FatigueModelPackage({"fatigue_models": {"enabled": False}}); package.enabled = True
        package.nets = {"eye_state": BrokenNet([[0., 0.]])}
        result = package.analyze(np.zeros((100,100,3), dtype=np.uint8), np.tile([50,50], (68,1)), (10,10,90,90))
        self.assertNotIn("eye_state", result["fatigue_models_available"])
        self.assertIn("eye_state", result["fatigue_models_errors"])

if __name__ == "__main__": unittest.main()
