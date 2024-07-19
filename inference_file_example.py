import argparse
from typing import List

import cv2
import numpy as np
from inference import get_model
from utils.general import find_in_list, load_zones_config
from utils.timers import FPSBasedTimer

import supervision as sv

# Màu sắc cho các zone
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)

# Label của các objects
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

# Code thực hiện phân tích video chính
def main(
    source_video_path: str,
    zone_configuration_path: str,
    model_id: str,
    confidence: float,
    iou: float,
    classes: List[int],
) -> None:
    # Sử dụng Yolov8 với Coco data set (https://universe.roboflow.com/microsoft/coco/dataset/5)
    model = get_model(model_id=model_id) 
    
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path) # Lấy frame từ video

    # Load các zone từ file config và tạo các PolygonZone
    polygons = load_zones_config(file_path=zone_configuration_path)
    zones = [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,),
        )
        for polygon in polygons
    ]
    
    # Bộ đo thời gian dựa trên fps của video
    timers = [FPSBasedTimer(video_info.fps) for _ in zones]

    # Xử lý từng frame trong video
    for frame in frames_generator:
        results = model.infer(frame, confidence=confidence, iou_threshold=iou)[0]
        detections = sv.Detections.from_inference(results)
        
        # Lọc ra các đối tượng trong các class cần theo dõi.
        # Trong Coco data set, class_id của người là 0
        detections = detections[find_in_list(detections.class_id, classes)] 
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()
        
        # Lặp qua từng zone để vẽ và xác định thời gian mà object ở trong zone
        for idx, zone in enumerate(zones):
            annotated_frame = sv.draw_polygon(
                scene=annotated_frame, polygon=zone.polygon, color=COLORS.by_idx(idx)
            )

            # Lọc để lấy các detections trong zone và tính thời gian mà object ở trong zone
            detections_in_zone = detections[zone.trigger(detections)]
            time_in_zone = timers[idx].tick(detections_in_zone)
            custom_color_lookup = np.full(detections_in_zone.class_id.shape, idx)

            annotated_frame = COLOR_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                custom_color_lookup=custom_color_lookup,
            )
            
            # gán label cho các object được detect
            labels = [
                f"#{tracker_id} {int(time // 60):02d}:{int(time % 60):02d}"
                for tracker_id, time in zip(detections_in_zone.tracker_id, time_in_zone)
            ]
            annotated_frame = LABEL_ANNOTATOR.annotate(
                scene=annotated_frame,
                detections=detections_in_zone,
                labels=labels,
                custom_color_lookup=custom_color_lookup,
            )

        # Hiển thị video
        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# Parse arguments và chạy hàm main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculating detections dwell time in zones, using video file."
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        required=True,
        help="Path to the zone configuration JSON file.",
    )
    parser.add_argument(
        "--source_video_path",
        type=str,
        required=True,
        help="Path to the source video file.",
    )
    parser.add_argument(
        "--model_id", type=str, default="yolov8s-640", help="Roboflow model ID."
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Confidence level for detections (0 to 1). Default is 0.3.",
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.7,
        type=float,
        help="IOU threshold for non-max suppression. Default is 0.7.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to track. If empty, all classes are tracked.",
    )
    args = parser.parse_args()

    main(
        source_video_path=args.source_video_path,
        zone_configuration_path=args.zone_configuration_path,
        model_id=args.model_id,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
    )
