"""InstantHMR — standalone 3D human pose inference + Rerun visualization."""

from .inference import InstantHMR, HMRPrediction
from .detector import RFDETRDetector, YOLODetector
from .pipeline import PosePipeline, FrameResult
from .skeleton import JOINT_NAMES, SKELETON_EDGES, NUM_JOINTS, edges_for
from .visualizer import RerunVisualizer, OpenCVVisualizer
from .adapter import process_video_with_instanthmr, get_pipeline

__all__ = [
    "InstantHMR",
    "HMRPrediction",
    "RFDETRDetector",
    "YOLODetector",
    "PosePipeline",
    "FrameResult",
    "OpenCVVisualizer",
    "JOINT_NAMES",
    "SKELETON_EDGES",
    "NUM_JOINTS",
    "edges_for",
    "process_video_with_instanthmr",
    "get_pipeline",
]

__version__ = "0.1.0"
