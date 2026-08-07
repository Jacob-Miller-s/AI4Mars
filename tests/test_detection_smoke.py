import unittest

from src.detection_smoke import run_detection_smoke


class DetectionSmokeTests(unittest.TestCase):
    def test_torchvision_ops_and_maskrcnn_forward_work_without_weights(self) -> None:
        report = run_detection_smoke(run_forward=True)

        self.assertEqual(report["nms_status"], "ok")
        self.assertEqual(report["roi_align_status"], "ok")
        self.assertEqual(report["roi_align_shape"], [1, 2, 2, 2])
        self.assertEqual(report["maskrcnn_construction_status"], "ok")
        self.assertEqual(report["minimal_forward_status"], "ok")
        self.assertEqual(set(report["minimal_forward_keys"]), {"boxes", "labels", "masks", "scores"})


if __name__ == "__main__":
    unittest.main()