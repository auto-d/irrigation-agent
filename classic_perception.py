"""Classical camera perception and offline anomaly tooling."""

from __future__ import annotations

import datetime as dt
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

from planner import Event
from services.rtsp.ingest import FFmpegVideoStream, ffprobe_stream_info


def _now() -> dt.datetime:
    """Return the current UTC timestamp."""
    return dt.datetime.now(dt.timezone.utc)


class SecurityCameraPerceptor:
    """Sample one or more frames and emit a constrained scene event."""

    def probe_raw(
        self,
        url: str | None = None,
        *,
        sample_frames: int = 1,
        save_frame_path: str | None = None,
        baseline_image_dir: str | None = None,
        baseline_output_path: str | None = None,
        baseline_visualization_path: str | None = None,
        experiment_image_dir: str | None = None,
        experiment_output_dir: str | None = None,
        score_image_path: str | None = None,
        score_image_dir: str | None = None,
        score_visualization_path: str | None = None,
    ) -> Dict[str, Any]:
        """Return inexpensive stream metadata and sampled frame statistics."""
        if experiment_image_dir:
            if not baseline_image_dir:
                raise RuntimeError("baseline image directory required when running image experiments")
            return {
                "mode": "experiment_image_directory",
                "experiments": self.experiment_labeled_image_directory(
                    baseline_image_dir=baseline_image_dir,
                    labeled_image_dir=experiment_image_dir,
                    output_dir=experiment_output_dir,
                ),
            }
        if baseline_image_dir:
            summary = self.build_baseline_from_directory(
                baseline_image_dir,
                save_baseline_path=baseline_output_path,
                visualization_path=baseline_visualization_path,
            )
            return {
                "mode": "baseline_directory",
                "baseline": summary,
            }
        if score_image_path:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still image")
            return {
                "mode": "score_image",
                "score": self.score_image_against_baseline(
                    score_image_path,
                    baseline_output_path,
                    visualization_path=score_visualization_path,
                ),
            }
        if score_image_dir:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still-image directory")
            return {
                "mode": "score_image_directory",
                "evaluation": self.evaluate_labeled_image_directory(
                    score_image_dir,
                    baseline_output_path,
                ),
            }
        if not url:
            raise RuntimeError("camera URL required")
        stream_info = ffprobe_stream_info(url)
        stats, capture_info = self._sample_frame_stats(
            url,
            width=int(stream_info["width"]),
            height=int(stream_info["height"]),
            sample_frames=sample_frames,
            save_frame_path=save_frame_path,
        )
        return {
            "stream": stream_info,
            "sample_frames": sample_frames,
            "frame_stats": stats,
            "capture": capture_info,
        }

    def perceive(
        self,
        url: str | None = None,
        *,
        sample_frames: int = 3,
        save_frame_path: str | None = None,
        baseline_image_dir: str | None = None,
        baseline_output_path: str | None = None,
        baseline_visualization_path: str | None = None,
        experiment_image_dir: str | None = None,
        experiment_output_dir: str | None = None,
        score_image_path: str | None = None,
        score_image_dir: str | None = None,
        score_visualization_path: str | None = None,
    ) -> Event:
        """Return one normalized camera scene event."""
        if experiment_image_dir:
            if not baseline_image_dir:
                raise RuntimeError("baseline image directory required when running image experiments")
            result = self.experiment_labeled_image_directory(
                baseline_image_dir=baseline_image_dir,
                labeled_image_dir=experiment_image_dir,
                output_dir=experiment_output_dir,
            )
            payload = {
                "mode": "experiment_image_directory",
                **result,
                "interpretation": "scene_anomaly_experiments",
            }
            return Event(timestamp=_now(), source="camera", type="scene_anomaly_experiments", payload=payload)
        if baseline_image_dir:
            summary = self.build_baseline_from_directory(
                baseline_image_dir,
                save_baseline_path=baseline_output_path,
                visualization_path=baseline_visualization_path,
            )
            payload = {
                "mode": "baseline_directory",
                "image_dir": summary["image_dir"],
                "image_count": summary["image_count"],
                "width": summary["width"],
                "height": summary["height"],
                "patch_size": summary["patch_size"],
                "mean_brightness_avg": summary["mean_brightness_avg"],
                "mean_brightness_std": summary["mean_brightness_std"],
                "normalization": summary["normalization"],
                "patch_rows": summary["patch_rows"],
                "patch_cols": summary["patch_cols"],
                "patch_variability_mean": summary["patch_variability_mean"],
                "patch_variability_p95": summary["patch_variability_p95"],
                "patch_variability_max": summary["patch_variability_max"],
                "feature_medians": summary["feature_medians"],
                "top_variable_patches": summary["top_variable_patches"],
                "saved_baseline_path": summary.get("saved_baseline_path"),
                "saved_baseline_visualization_path": summary.get("saved_baseline_visualization_path"),
                "interpretation": "scene_baseline_established",
            }
            return Event(timestamp=_now(), source="camera", type="scene_baseline", payload=payload)
        if score_image_path:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still image")
            result = self.score_image_against_baseline(
                score_image_path,
                baseline_output_path,
                visualization_path=score_visualization_path,
            )
            payload = {
                "mode": "score_image",
                **result,
                "interpretation": "scene_anomaly_scored",
            }
            return Event(timestamp=_now(), source="camera", type="scene_anomaly_score", payload=payload)
        if score_image_dir:
            if not baseline_output_path:
                raise RuntimeError("baseline output path required when scoring a still-image directory")
            result = self.evaluate_labeled_image_directory(score_image_dir, baseline_output_path)
            payload = {
                "mode": "score_image_directory",
                **result,
                "interpretation": "scene_anomaly_batch_evaluated",
            }
            return Event(timestamp=_now(), source="camera", type="scene_anomaly_batch", payload=payload)
        if not url:
            raise RuntimeError("camera URL required")
        stream_info = ffprobe_stream_info(url)
        stats, capture_info = self._sample_frame_stats(
            url,
            width=int(stream_info["width"]),
            height=int(stream_info["height"]),
            sample_frames=sample_frames,
            save_frame_path=save_frame_path,
        )
        mean_brightness = round(float(np.mean([frame["mean_brightness"] for frame in stats])), 2) if stats else None
        motion_score = max((frame["motion_score"] for frame in stats), default=0.0)
        payload = {
            "camera_backend": "classic",
            "width": stream_info["width"],
            "height": stream_info["height"],
            "sample_frames": len(stats),
            "saved_frame_path": capture_info.get("saved_frame_path"),
            "mean_brightness": mean_brightness,
            "motion_score": round(float(motion_score), 4),
            "interpretation": "unclassified_scene",
            "person_detected": None,
            "animal_detected": None,
            "lawn_mower_active": None,
        }
        return Event(timestamp=_now(), source="camera", type="scene_activity", payload=payload)

    def build_baseline_from_directory(
        self,
        image_dir: str,
        *,
        save_baseline_path: str | None = None,
        patch_size: int = 32,
        visualization_path: str | None = None,
    ) -> Dict[str, Any]:
        """Build a texture-series baseline from a directory of still images."""
        image_paths = self._list_image_paths(image_dir)
        if len(image_paths) < 2:
            raise RuntimeError("baseline image directory must contain at least two readable images")

        brightness_values: List[float] = []
        reference_shape: Tuple[int, int, int] | None = None
        coarse_frames: List[Dict[str, np.ndarray]] = []
        fine_frames: List[Dict[str, np.ndarray]] = []
        fine_patch_size = max(8, patch_size // 2)
        mean_bgr_accumulator: np.ndarray | None = None
        for path in image_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if reference_shape is None:
                reference_shape = frame.shape
            if frame.shape != reference_shape:
                raise RuntimeError(
                    f"baseline image {path} has shape {frame.shape}, expected {reference_shape}"
                )
            if mean_bgr_accumulator is None:
                mean_bgr_accumulator = frame.astype(np.float32)
            else:
                mean_bgr_accumulator += frame.astype(np.float32)
            brightness_values.append(float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))))
            coarse_frames.append(self._compute_patch_texture_descriptors(frame, patch_size=patch_size))
            fine_frames.append(self._compute_patch_texture_descriptors(frame, patch_size=fine_patch_size))

        if len(coarse_frames) < 2 or reference_shape is None:
            raise RuntimeError("baseline image directory must contain at least two readable images")

        height, width = reference_shape[:2]
        coarse_model = self._build_patch_texture_model(coarse_frames)
        fine_model = self._build_patch_texture_model(fine_frames)
        patch_rows, patch_cols = coarse_model["grad_mean_median"].shape
        fine_patch_rows, fine_patch_cols = fine_model["grad_mean_median"].shape

        top_variable_patches = self._top_scored_patches(
            coarse_model["patch_variability"],
            patch_size=patch_size,
            image_width=width,
            image_height=height,
            score_label="variability_score",
        )

        summary = {
            "image_dir": str(Path(image_dir).expanduser().resolve()),
            "image_count": len(coarse_frames),
            "sample_image_names": [path.name for path in image_paths[: min(10, len(image_paths))]],
            "width": width,
            "height": height,
            "patch_size": patch_size,
            "patch_rows": int(patch_rows),
            "patch_cols": int(patch_cols),
            "fine_patch_size": int(fine_patch_size),
            "fine_patch_rows": int(fine_patch_rows),
            "fine_patch_cols": int(fine_patch_cols),
            "normalization": "local_contrast_gradient_lbp",
            "mean_brightness_avg": round(float(np.mean(brightness_values)), 4),
            "mean_brightness_std": round(float(np.std(brightness_values)), 4),
            "feature_medians": {
                "gradient_mean": round(float(np.median(coarse_model["grad_mean_median"])), 4),
                "gradient_std": round(float(np.median(coarse_model["grad_std_median"])), 4),
                "edge_density": round(float(np.median(coarse_model["edge_density_median"])), 6),
                "lab_a_mean": round(float(np.median(coarse_model["lab_a_median"])), 4),
                "lab_b_mean": round(float(np.median(coarse_model["lab_b_median"])), 4),
            },
            "patch_variability_mean": round(float(np.mean(coarse_model["patch_variability"])), 4),
            "patch_variability_p95": round(float(np.percentile(coarse_model["patch_variability"], 95)), 4),
            "patch_variability_max": round(float(np.max(coarse_model["patch_variability"])), 4),
            "fine_patch_variability_mean": round(float(np.mean(fine_model["patch_variability"])), 4),
            "fine_patch_variability_p95": round(float(np.percentile(fine_model["patch_variability"], 95)), 4),
            "fine_patch_variability_max": round(float(np.max(fine_model["patch_variability"])), 4),
            "top_variable_patches": top_variable_patches,
        }

        if visualization_path:
            if mean_bgr_accumulator is None:
                raise RuntimeError("unable to render baseline visualization without readable images")
            mean_bgr_image = np.clip(mean_bgr_accumulator / float(len(coarse_frames)), 0, 255).astype(np.uint8)
            saved_baseline_visualization_path = self._write_baseline_visualization(
                visualization_path,
                mean_bgr_image=mean_bgr_image,
                coarse_variability=coarse_model["patch_variability"],
                fine_variability=fine_model["patch_variability"],
                patch_size=patch_size,
                fine_patch_size=fine_patch_size,
                title=f"Baseline: {Path(image_dir).name}",
            )
            summary["saved_baseline_visualization_path"] = saved_baseline_visualization_path

        if save_baseline_path:
            saved_path = self._write_baseline_artifact(
                save_baseline_path,
                summary=summary,
                patch_model={f"coarse_{key}": value for key, value in coarse_model.items()} | {
                    f"fine_{key}": value for key, value in fine_model.items()
                },
            )
            summary["saved_baseline_path"] = saved_path

        return summary

    def score_image_against_baseline(
        self,
        image_path: str,
        baseline_path: str,
        *,
        visualization_path: str | None = None,
    ) -> Dict[str, Any]:
        """Score one still image against a saved texture baseline."""
        baseline = self._load_baseline_artifact(baseline_path)
        model = baseline["model"]
        summary = baseline["summary"]
        patch_size = int(summary["patch_size"])
        fine_patch_size = int(summary.get("fine_patch_size", max(8, patch_size // 2)))

        frame = cv2.imread(str(Path(image_path).expanduser()), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"unable to read score image: {image_path}")
        expected_shape = (int(summary["height"]), int(summary["width"]), 3)
        if frame.shape != expected_shape:
            raise RuntimeError(f"score image shape {frame.shape} did not match baseline shape {expected_shape}")

        coarse_descriptors = self._compute_patch_texture_descriptors(frame, patch_size=patch_size)
        fine_descriptors = self._compute_patch_texture_descriptors(frame, patch_size=fine_patch_size)
        coarse_score_grid = self._score_descriptor_grid(coarse_descriptors, model, prefix="coarse")
        fine_score_grid = self._score_descriptor_grid(fine_descriptors, model, prefix="fine")
        score_grid = coarse_score_grid
        threshold = self._fine_anomaly_threshold(fine_score_grid, model)
        support_threshold = max(2.4, threshold * 0.58)
        anomalous = fine_score_grid >= threshold
        support_mask = fine_score_grid >= support_threshold
        all_regions = self._extract_anomalous_regions(
            anomalous,
            support_mask,
            fine_score_grid,
            patch_size=fine_patch_size,
            image_width=int(summary["width"]),
            image_height=int(summary["height"]),
        )
        accepted_regions, rejected_regions = self._split_object_like_regions(all_regions)
        scene_score = (0.35 * self._scene_score_from_grid(coarse_score_grid)) + (
            0.65 * self._scene_score_from_grid(fine_score_grid)
        )
        pedestrian_score = self._pedestrian_score_from_regions(accepted_regions)
        linear_change_score = self._linear_change_score_from_regions(rejected_regions)

        saved_score_visualization_path = None
        if visualization_path:
            saved_score_visualization_path = self._write_score_visualization(
                visualization_path,
                frame=frame,
                coarse_score_grid=coarse_score_grid,
                fine_score_grid=fine_score_grid,
                accepted_regions=accepted_regions,
                rejected_regions=rejected_regions,
                patch_size=patch_size,
                fine_patch_size=fine_patch_size,
                title=Path(image_path).name,
            )

        return {
            "image_path": str(Path(image_path).expanduser().resolve()),
            "baseline_path": str(Path(baseline_path).expanduser().resolve()),
            "width": int(summary["width"]),
            "height": int(summary["height"]),
            "patch_size": patch_size,
            "patch_rows": int(summary["patch_rows"]),
            "patch_cols": int(summary["patch_cols"]),
            "fine_patch_size": fine_patch_size,
            "fine_patch_rows": int(summary.get("fine_patch_rows", fine_score_grid.shape[0])),
            "fine_patch_cols": int(summary.get("fine_patch_cols", fine_score_grid.shape[1])),
            "anomaly_threshold": round(float(threshold), 4),
            "support_threshold": round(float(support_threshold), 4),
            "scene_score": round(float(scene_score), 4),
            "pedestrian_score": round(float(pedestrian_score), 4),
            "anomaly_score_mean": round(float(np.mean(score_grid)), 4),
            "anomaly_score_p95": round(float(np.percentile(score_grid, 95)), 4),
            "anomaly_score_max": round(float(np.max(score_grid)), 4),
            "anomalous_patch_count": int(np.sum(anomalous)),
            "object_like_region_count": int(len(accepted_regions)),
            "linear_region_count": int(len(rejected_regions)),
            "linear_change_score": round(float(linear_change_score), 4),
            "saved_score_visualization_path": saved_score_visualization_path,
            "top_anomalous_patches": self._top_scored_patches(
                fine_score_grid,
                patch_size=fine_patch_size,
                image_width=int(summary["width"]),
                image_height=int(summary["height"]),
                score_label="anomaly_score",
            ),
            "candidate_boxes": accepted_regions,
            "rejected_candidate_boxes": rejected_regions[:8],
        }

    def evaluate_labeled_image_directory(
        self,
        image_dir: str,
        baseline_path: str,
    ) -> Dict[str, Any]:
        """Score a labeled directory and aggregate anomaly separation by label."""
        image_paths = self._list_image_paths(image_dir)
        records: List[Dict[str, Any]] = []
        by_label: Dict[str, List[Dict[str, Any]]] = {}
        for path in image_paths:
            score = self.score_image_against_baseline(str(path), baseline_path)
            label = self._label_from_filename(path)
            record = {
                "image_name": path.name,
                "label": label,
                "scene_score": score["scene_score"],
                "pedestrian_score": score["pedestrian_score"],
                "anomaly_score_max": score["anomaly_score_max"],
                "anomaly_score_p95": score["anomaly_score_p95"],
                "anomalous_patch_count": score["anomalous_patch_count"],
                "object_like_region_count": score["object_like_region_count"],
                "linear_region_count": score["linear_region_count"],
                "linear_change_score": score["linear_change_score"],
                "candidate_box_count": len(score["candidate_boxes"]),
                "candidate_boxes": score["candidate_boxes"][:3],
            }
            records.append(record)
            by_label.setdefault(label, []).append(record)

        label_summary: Dict[str, Dict[str, Any]] = {}
        for label, items in sorted(by_label.items()):
            scene_scores = [item["scene_score"] for item in items]
            pedestrian_scores = [item["pedestrian_score"] for item in items]
            max_scores = [item["anomaly_score_max"] for item in items]
            patch_counts = [item["anomalous_patch_count"] for item in items]
            box_counts = [item["candidate_box_count"] for item in items]
            linear_scores = [item["linear_change_score"] for item in items]
            object_like_counts = [item["object_like_region_count"] for item in items]
            label_summary[label] = {
                "count": len(items),
                "scene_score_avg": round(float(np.mean(scene_scores)), 4),
                "scene_score_p95": round(float(np.percentile(scene_scores, 95)), 4),
                "scene_score_max": round(float(np.max(scene_scores)), 4),
                "pedestrian_score_avg": round(float(np.mean(pedestrian_scores)), 4),
                "pedestrian_score_p95": round(float(np.percentile(pedestrian_scores, 95)), 4),
                "pedestrian_score_max": round(float(np.max(pedestrian_scores)), 4),
                "anomaly_score_max_avg": round(float(np.mean(max_scores)), 4),
                "anomaly_score_max_p95": round(float(np.percentile(max_scores, 95)), 4),
                "anomaly_score_max_max": round(float(np.max(max_scores)), 4),
                "anomalous_patch_count_avg": round(float(np.mean(patch_counts)), 4),
                "object_like_region_count_avg": round(float(np.mean(object_like_counts)), 4),
                "candidate_box_count_avg": round(float(np.mean(box_counts)), 4),
                "linear_change_score_avg": round(float(np.mean(linear_scores)), 4),
                "top_examples": sorted(items, key=lambda item: item["pedestrian_score"], reverse=True)[:3],
            }

        threshold_summary = self._threshold_summary(records, score_key="pedestrian_score")
        return {
            "image_dir": str(Path(image_dir).expanduser().resolve()),
            "baseline_path": str(Path(baseline_path).expanduser().resolve()),
            "image_count": len(records),
            "label_summary": label_summary,
            "threshold_summary": threshold_summary,
            "records": sorted(records, key=lambda item: item["pedestrian_score"], reverse=True),
        }

    def experiment_labeled_image_directory(
        self,
        *,
        baseline_image_dir: str,
        labeled_image_dir: str,
        output_dir: str | None = None,
    ) -> Dict[str, Any]:
        """Run multiple direct residual-map experiments against labeled frames."""
        reference = self._build_background_reference(baseline_image_dir)
        image_paths = self._list_image_paths(labeled_image_dir)
        methods = self._experiment_methods(reference)
        output_path = Path(output_dir).expanduser().resolve() if output_dir else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        clear_exemplar = next((path for path in image_paths if self._label_from_filename(path) == "CLEAR"), None)
        mower_exemplar = next(
            (path for path in image_paths if "MOWER" in self._label_from_filename(path)),
            None,
        )
        person_exemplar = next((path for path in image_paths if self._label_from_filename(path) == "PERSON"), None)

        experiment_summaries: List[Dict[str, Any]] = []
        for method_name, method in methods.items():
            records: List[Dict[str, Any]] = []
            by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for path in image_paths:
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                score_map = method(frame)
                metrics = self._summarize_experiment_map(score_map)
                label = self._label_from_filename(path)
                record = {
                    "image_name": path.name,
                    "label": label,
                    **metrics,
                }
                records.append(record)
                by_label[label].append(record)

            threshold_summary = self._threshold_summary(records, score_key="top_region_score")
            label_summary = self._experiment_label_summary(by_label)
            experiment_summary = {
                "method": method_name,
                "label_summary": label_summary,
                "threshold_summary": threshold_summary,
                "records": sorted(records, key=lambda item: item["top_region_score"], reverse=True),
            }
            experiment_summaries.append(experiment_summary)

            if output_path is not None:
                for exemplar_name, exemplar_path in (
                    ("clear", clear_exemplar),
                    ("mower", mower_exemplar),
                    ("person", person_exemplar),
                ):
                    if exemplar_path is None:
                        continue
                    frame = cv2.imread(str(exemplar_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    score_map = method(frame)
                    metrics = self._summarize_experiment_map(score_map)
                    self._write_experiment_visualization(
                        output_path / f"{method_name}_{exemplar_name}.jpg",
                        frame=frame,
                        score_map=score_map,
                        components=metrics["top_regions"],
                        title=f"{method_name}: {exemplar_path.name}",
                    )

        experiment_summaries.sort(
            key=lambda item: (
                item["threshold_summary"].get("best_accuracy_threshold", {}).get("balanced_accuracy", 0.0),
                item["threshold_summary"].get("best_accuracy_threshold", {}).get("accuracy", 0.0),
                item["label_summary"].get("MOWER", {}).get("top_region_score_avg", 0.0),
            ),
            reverse=True,
        )
        result = {
            "baseline_image_dir": str(Path(baseline_image_dir).expanduser().resolve()),
            "labeled_image_dir": str(Path(labeled_image_dir).expanduser().resolve()),
            "output_dir": str(output_path) if output_path is not None else None,
            "method_count": len(experiment_summaries),
            "methods": experiment_summaries,
        }
        if output_path is not None:
            summary_path = output_path / "summary.json"
            summary_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    def evaluate_best_effort_classifier(
        self,
        *,
        baseline_image_dir: str,
        labeled_image_dir: str,
        output_dir: str | None = None,
    ) -> Dict[str, Any]:
        """Run a pragmatic scene classifier with and without nuisance suppression."""
        reference = self._build_background_reference(baseline_image_dir)
        methods = self._experiment_methods(reference)
        raw_method = methods["lab_illumination_intersection"]
        suppressed_method = methods["baseline_suppressed_combo"]

        image_paths = self._list_image_paths(labeled_image_dir)
        output_path = Path(output_dir).expanduser().resolve() if output_dir else None
        if output_path is not None:
            output_path.mkdir(parents=True, exist_ok=True)

        raw_records = self._classifier_records(image_paths, raw_method)
        suppressed_records = self._classifier_records(image_paths, suppressed_method)

        positive_threshold = self._fit_positive_threshold(suppressed_records)
        type_threshold = self._fit_type_threshold(suppressed_records, positive_threshold=positive_threshold)

        raw_records = self._apply_classifier_thresholds(
            raw_records,
            positive_threshold=positive_threshold,
            type_threshold=type_threshold,
        )
        suppressed_records = self._apply_classifier_thresholds(
            suppressed_records,
            positive_threshold=positive_threshold,
            type_threshold=type_threshold,
        )

        if output_path is not None:
            nuisance_panel_path = self._write_nuisance_baseline_visualization(
                output_path / "nuisance_baseline.jpg",
                reference=reference,
            )
            sample_dir = output_path / "samples"
            sample_dir.mkdir(parents=True, exist_ok=True)
            for path in image_paths:
                raw_record = next(item for item in raw_records if item["image_name"] == path.name)
                suppressed_record = next(item for item in suppressed_records if item["image_name"] == path.name)
                frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                raw_map = raw_method(frame)
                suppressed_map = suppressed_method(frame)
                self._write_classifier_comparison_visualization(
                    sample_dir / f"{path.stem}.jpg",
                    frame=frame,
                    raw_map=raw_map,
                    raw_record=raw_record,
                    suppressed_map=suppressed_map,
                    suppressed_record=suppressed_record,
                    title=path.name,
                )
        else:
            nuisance_panel_path = None

        result = {
            "baseline_image_dir": str(Path(baseline_image_dir).expanduser().resolve()),
            "labeled_image_dir": str(Path(labeled_image_dir).expanduser().resolve()),
            "output_dir": str(output_path) if output_path is not None else None,
            "nuisance_baseline_path": nuisance_panel_path,
            "positive_threshold": round(float(positive_threshold), 4),
            "type_threshold": round(float(type_threshold), 2),
            "without_baseline": self._classifier_summary(raw_records, score_key="plausible_score"),
            "with_baseline": self._classifier_summary(suppressed_records, score_key="plausible_score"),
            "comparison_records": [
                {
                    "image_name": raw_record["image_name"],
                    "label": raw_record["label"],
                    "without_baseline_prediction": raw_record["predicted_label"],
                    "without_baseline_score": raw_record["plausible_score"],
                    "with_baseline_prediction": suppressed_record["predicted_label"],
                    "with_baseline_score": suppressed_record["plausible_score"],
                }
                for raw_record, suppressed_record in zip(raw_records, suppressed_records)
            ],
        }
        if output_path is not None:
            (output_path / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result

    def _sample_frame_stats(
        self,
        url: str,
        *,
        width: int,
        height: int,
        sample_frames: int,
        save_frame_path: str | None = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Collect lightweight frame metrics without downstream inference."""
        stats: List[Dict[str, Any]] = []
        previous_gray = None
        saved_frame_path: str | None = None
        first_frame_shape: List[int] | None = None
        first_frame_dtype: str | None = None
        started_at = time.monotonic()
        with FFmpegVideoStream(url, width, height).start() as stream:
            for index in range(sample_frames):
                frame, error = stream.read()
                if frame is None:
                    raise RuntimeError(f"Unable to read RTSP frame: {error or 'unknown error'}")
                if index == 0:
                    first_frame_shape = list(frame.shape)
                    first_frame_dtype = str(frame.dtype)
                if index == 0 and save_frame_path:
                    saved_frame_path = self._write_debug_frame(frame, save_frame_path)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                mean_brightness = float(np.mean(gray))
                brightness_stddev = float(np.std(gray))
                contrast = brightness_stddev
                blur_laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                motion_score = 0.0
                if previous_gray is not None:
                    motion_score = float(np.mean(cv2.absdiff(gray, previous_gray)))
                previous_gray = gray
                stats.append(
                    {
                        "index": index,
                        "mean_brightness": mean_brightness,
                        "brightness_stddev": brightness_stddev,
                        "contrast": contrast,
                        "blur_laplacian_variance": blur_laplacian_variance,
                        "motion_score": motion_score,
                    }
                )
        elapsed_seconds = max(time.monotonic() - started_at, 1e-6)
        capture_info = {
            "saved_frame_path": saved_frame_path,
            "frame_shape": first_frame_shape,
            "dtype": first_frame_dtype,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "approx_fps": round(len(stats) / elapsed_seconds, 3),
        }
        return stats, capture_info

    @staticmethod
    def _list_image_paths(image_dir: str) -> List[Path]:
        path = Path(image_dir).expanduser()
        if not path.is_dir():
            raise RuntimeError(f"baseline image directory not found: {path}")
        image_paths = sorted(
            file_path
            for file_path in path.iterdir()
            if (
                file_path.is_file()
                and file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                and not SecurityCameraPerceptor._is_generated_visualization(file_path)
            )
        )
        if not image_paths:
            raise RuntimeError(f"no supported image files found in baseline directory: {path}")
        return image_paths

    @staticmethod
    def _is_generated_visualization(path: Path) -> bool:
        stem = path.stem.lower()
        return any(token in stem for token in ("baseline_viz", "score_viz"))

    @staticmethod
    def _top_scored_patches(
        score_grid: np.ndarray,
        *,
        patch_size: int,
        image_width: int,
        image_height: int,
        score_label: str,
        mask: np.ndarray | None = None,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        ranked: List[Dict[str, Any]] = []
        rows, cols = score_grid.shape
        for row in range(rows):
            for col in range(cols):
                if mask is not None and not bool(mask[row, col]):
                    continue
                x0 = col * patch_size
                y0 = row * patch_size
                x1 = min(image_width, x0 + patch_size)
                y1 = min(image_height, y0 + patch_size)
                ranked.append(
                    {
                        "row": int(row),
                        "col": int(col),
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x1),
                        "y1": int(y1),
                        score_label: round(float(score_grid[row, col]), 4),
                    }
                )
        ranked.sort(key=lambda item: item[score_label], reverse=True)
        return ranked[:limit]

    @staticmethod
    def _build_patch_texture_model(frame_descriptors: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        grad_mean_stack = np.stack([item["grad_mean"] for item in frame_descriptors], axis=0)
        grad_std_stack = np.stack([item["grad_std"] for item in frame_descriptors], axis=0)
        edge_density_stack = np.stack([item["edge_density"] for item in frame_descriptors], axis=0)
        lab_a_stack = np.stack([item["lab_a_mean"] for item in frame_descriptors], axis=0)
        lab_b_stack = np.stack([item["lab_b_mean"] for item in frame_descriptors], axis=0)
        lbp_hist_stack = np.stack([item["lbp_hist"] for item in frame_descriptors], axis=0)

        grad_mean_median = np.median(grad_mean_stack, axis=0)
        grad_std_median = np.median(grad_std_stack, axis=0)
        edge_density_median = np.median(edge_density_stack, axis=0)
        lab_a_median = np.median(lab_a_stack, axis=0)
        lab_b_median = np.median(lab_b_stack, axis=0)
        lbp_hist_mean = np.mean(lbp_hist_stack, axis=0)
        lbp_hist_std = np.std(lbp_hist_stack, axis=0)

        grad_mean_mad = np.median(np.abs(grad_mean_stack - grad_mean_median), axis=0)
        grad_std_mad = np.median(np.abs(grad_std_stack - grad_std_median), axis=0)
        edge_density_mad = np.median(np.abs(edge_density_stack - edge_density_median), axis=0)
        lab_a_mad = np.median(np.abs(lab_a_stack - lab_a_median), axis=0)
        lab_b_mad = np.median(np.abs(lab_b_stack - lab_b_median), axis=0)

        patch_variability = (
            grad_mean_mad
            + grad_std_mad
            + (edge_density_mad * 255.0)
            + (lab_a_mad * 0.5)
            + (lab_b_mad * 0.5)
            + np.mean(lbp_hist_std, axis=2)
        )
        return {
            "grad_mean_median": grad_mean_median,
            "grad_mean_mad": grad_mean_mad,
            "grad_std_median": grad_std_median,
            "grad_std_mad": grad_std_mad,
            "edge_density_median": edge_density_median,
            "edge_density_mad": edge_density_mad,
            "lab_a_median": lab_a_median,
            "lab_a_mad": lab_a_mad,
            "lab_b_median": lab_b_median,
            "lab_b_mad": lab_b_mad,
            "lbp_hist_mean": lbp_hist_mean,
            "lbp_hist_std": lbp_hist_std,
            "patch_variability": patch_variability,
        }

    @staticmethod
    def _score_descriptor_grid(
        descriptors: Dict[str, np.ndarray],
        model: Dict[str, np.ndarray],
        *,
        prefix: str,
    ) -> np.ndarray:
        grad_mean_score = np.abs(descriptors["grad_mean"] - model[f"{prefix}_grad_mean_median"]) / (
            model[f"{prefix}_grad_mean_mad"] + 1.0
        )
        grad_std_score = np.abs(descriptors["grad_std"] - model[f"{prefix}_grad_std_median"]) / (
            model[f"{prefix}_grad_std_mad"] + 1.0
        )
        edge_density_score = np.abs(descriptors["edge_density"] - model[f"{prefix}_edge_density_median"]) / (
            model[f"{prefix}_edge_density_mad"] + 0.01
        )
        lab_a_median = model.get(f"{prefix}_lab_a_median")
        lab_a_mad = model.get(f"{prefix}_lab_a_mad")
        lab_b_median = model.get(f"{prefix}_lab_b_median")
        lab_b_mad = model.get(f"{prefix}_lab_b_mad")
        if lab_a_median is None or lab_a_mad is None or lab_b_median is None or lab_b_mad is None:
            color_score = np.zeros_like(grad_mean_score)
        else:
            lab_a_score = np.abs(descriptors["lab_a_mean"] - lab_a_median) / (lab_a_mad + 2.0)
            lab_b_score = np.abs(descriptors["lab_b_mean"] - lab_b_median) / (lab_b_mad + 2.0)
            color_score = (lab_a_score + lab_b_score) / 2.0
        lbp_distance = 0.5 * np.sum(
            np.abs(descriptors["lbp_hist"] - model[f"{prefix}_lbp_hist_mean"]),
            axis=2,
        )
        return (
            grad_mean_score
            + grad_std_score
            + edge_density_score
            + (color_score * 1.5)
            + (lbp_distance * 8.0)
        ) / 5.0

    @staticmethod
    def _write_baseline_artifact(
        save_baseline_path: str,
        *,
        summary: Dict[str, Any],
        patch_model: Dict[str, np.ndarray],
    ) -> str:
        path = Path(save_baseline_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "summary": summary,
            "model": {key: np.round(value, 6).tolist() for key, value in patch_model.items()},
        }
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        return str(path.resolve())

    @staticmethod
    def _write_baseline_visualization(
        visualization_path: str,
        *,
        mean_bgr_image: np.ndarray,
        coarse_variability: np.ndarray,
        fine_variability: np.ndarray,
        patch_size: int,
        fine_patch_size: int,
        title: str,
    ) -> str:
        coarse_panel = SecurityCameraPerceptor._render_heatmap_panel(
            mean_bgr_image,
            coarse_variability,
            patch_size=patch_size,
            label="Coarse variability",
        )
        fine_panel = SecurityCameraPerceptor._render_heatmap_panel(
            mean_bgr_image,
            fine_variability,
            patch_size=fine_patch_size,
            label="Fine variability",
        )
        base_panel = mean_bgr_image.copy()
        SecurityCameraPerceptor._put_panel_label(base_panel, "Mean scene")
        combined = cv2.hconcat([base_panel, coarse_panel, fine_panel])
        SecurityCameraPerceptor._put_title(combined, title)
        return SecurityCameraPerceptor._write_visualization_image(visualization_path, combined)

    @staticmethod
    def _write_score_visualization(
        visualization_path: str,
        *,
        frame: np.ndarray,
        coarse_score_grid: np.ndarray,
        fine_score_grid: np.ndarray,
        accepted_regions: List[Dict[str, Any]],
        rejected_regions: List[Dict[str, Any]],
        patch_size: int,
        fine_patch_size: int,
        title: str,
    ) -> str:
        original_panel = frame.copy()
        SecurityCameraPerceptor._put_panel_label(original_panel, "Candidate image")
        coarse_panel = SecurityCameraPerceptor._render_heatmap_panel(
            frame,
            coarse_score_grid,
            patch_size=patch_size,
            label="Coarse anomaly heat",
        )
        fine_panel = SecurityCameraPerceptor._render_heatmap_panel(
            frame,
            fine_score_grid,
            patch_size=fine_patch_size,
            label="Fine anomaly heat",
        )
        region_panel = frame.copy()
        SecurityCameraPerceptor._draw_region_overlays(region_panel, rejected_regions, color=(0, 165, 255))
        SecurityCameraPerceptor._draw_region_overlays(region_panel, accepted_regions, color=(0, 220, 0))
        SecurityCameraPerceptor._put_panel_label(region_panel, "Regions: green=object orange=linear")
        combined = cv2.hconcat([original_panel, coarse_panel, fine_panel, region_panel])
        SecurityCameraPerceptor._put_title(combined, title)
        return SecurityCameraPerceptor._write_visualization_image(visualization_path, combined)

    @staticmethod
    def _render_heatmap_panel(
        base_image: np.ndarray,
        score_grid: np.ndarray,
        *,
        patch_size: int,
        label: str,
    ) -> np.ndarray:
        heat = SecurityCameraPerceptor._grid_to_heatmap(
            score_grid,
            height=base_image.shape[0],
            width=base_image.shape[1],
            patch_size=patch_size,
        )
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_TURBO)
        panel = cv2.addWeighted(base_image, 0.55, colored, 0.45, 0.0)
        SecurityCameraPerceptor._put_panel_label(panel, label)
        return panel

    @staticmethod
    def _grid_to_heatmap(
        score_grid: np.ndarray,
        *,
        height: int,
        width: int,
        patch_size: int,
    ) -> np.ndarray:
        rows, cols = score_grid.shape
        heat = np.zeros((height, width), dtype=np.float32)
        for row in range(rows):
            for col in range(cols):
                y0 = row * patch_size
                x0 = col * patch_size
                y1 = min(height, y0 + patch_size)
                x1 = min(width, x0 + patch_size)
                heat[y0:y1, x0:x1] = float(score_grid[row, col])
        normalized = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    @staticmethod
    def _draw_region_overlays(
        image: np.ndarray,
        regions: List[Dict[str, Any]],
        *,
        color: Tuple[int, int, int],
    ) -> None:
        for index, region in enumerate(regions[:12]):
            cv2.rectangle(
                image,
                (int(region["x0"]), int(region["y0"])),
                (int(region["x1"]), int(region["y1"])),
                color,
                2,
            )
            score = region.get("object_score", region.get("region_score", region.get("score_max", 0.0)))
            label = f"{index + 1}:{float(score):.1f}"
            cv2.putText(
                image,
                label,
                (int(region["x0"]) + 4, max(118, int(region["y0"]) + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _put_panel_label(image: np.ndarray, label: str) -> None:
        cv2.rectangle(image, (8, 64), (380, 102), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (16, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _put_title(image: np.ndarray, title: str) -> None:
        cv2.rectangle(image, (0, 0), (image.shape[1], 56), (20, 20, 20), -1)
        cv2.putText(
            image,
            title,
            (16, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _write_visualization_image(path: str, image: np.ndarray) -> str:
        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"failed to write visualization image to {output_path}")
        return str(output_path.resolve())

    @staticmethod
    def _write_experiment_visualization(
        path: Path,
        *,
        frame: np.ndarray,
        score_map: np.ndarray,
        components: List[Dict[str, Any]],
        title: str,
    ) -> str:
        original = frame.copy()
        SecurityCameraPerceptor._put_panel_label(original, "Candidate image")
        normalized = cv2.normalize(score_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        heat_panel = cv2.addWeighted(frame, 0.55, heat, 0.45, 0.0)
        SecurityCameraPerceptor._put_panel_label(heat_panel, "Residual heat")
        region_panel = frame.copy()
        SecurityCameraPerceptor._draw_region_overlays(region_panel, components, color=(0, 220, 0))
        SecurityCameraPerceptor._put_panel_label(region_panel, "Top regions")
        combined = cv2.hconcat([original, heat_panel, region_panel])
        SecurityCameraPerceptor._put_title(combined, title)
        return SecurityCameraPerceptor._write_visualization_image(str(path), combined)

    @staticmethod
    def _load_baseline_artifact(path: str) -> Dict[str, Any]:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        model = {key: np.array(value, dtype=np.float32) for key, value in payload["model"].items()}
        return {
            "summary": payload["summary"],
            "model": model,
        }

    @staticmethod
    def _build_background_reference(image_dir: str) -> Dict[str, np.ndarray]:
        image_paths = SecurityCameraPerceptor._list_image_paths(image_dir)
        frames: List[np.ndarray] = []
        for path in image_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frames.append(frame)
        if len(frames) < 2:
            raise RuntimeError("need at least two readable baseline images for experiments")
        frame_stack = np.stack(frames, axis=0).astype(np.float32)
        median_bgr = np.median(frame_stack, axis=0).astype(np.float32)
        median_lab = cv2.cvtColor(np.clip(median_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        median_gray = cv2.cvtColor(np.clip(median_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        median_blur = cv2.GaussianBlur(median_gray, (0, 0), 21)
        median_grad = SecurityCameraPerceptor._gradient_magnitude(median_gray)
        nuisance_heatmap = SecurityCameraPerceptor._build_nuisance_heatmap(
            frames=frames,
            median_bgr=median_bgr,
            median_lab=median_lab,
            median_gray=median_gray,
            median_blur=median_blur,
        )
        spatial_prior = SecurityCameraPerceptor._build_spatial_prior(
            height=int(median_gray.shape[0]),
            width=int(median_gray.shape[1]),
        )
        reliability_map = np.clip((1.0 - (0.85 * nuisance_heatmap)) * spatial_prior, 0.0, 1.0)
        return {
            "median_bgr": median_bgr,
            "median_lab": median_lab,
            "median_gray": median_gray,
            "median_blur": median_blur,
            "median_grad": median_grad,
            "nuisance_heatmap": nuisance_heatmap,
            "spatial_prior": spatial_prior,
            "reliability_map": reliability_map,
        }

    @staticmethod
    def _experiment_methods(
        reference: Dict[str, np.ndarray],
    ) -> Dict[str, Callable[[np.ndarray], np.ndarray]]:
        def _normalize_map(score: np.ndarray) -> np.ndarray:
            return cv2.normalize(score, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

        def lab_ab_delta(frame: np.ndarray) -> np.ndarray:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            diff_a = np.abs(lab[:, :, 1] - reference["median_lab"][:, :, 1])
            diff_b = np.abs(lab[:, :, 2] - reference["median_lab"][:, :, 2])
            score = diff_a + diff_b
            return cv2.GaussianBlur(score, (0, 0), 7)

        def illumination_normalized_delta(frame: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frame_norm = gray - cv2.GaussianBlur(gray, (0, 0), 21)
            ref_norm = reference["median_gray"] - reference["median_blur"]
            score = np.abs(frame_norm - ref_norm)
            return cv2.GaussianBlur(score, (0, 0), 7)

        def gradient_delta(frame: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            grad = SecurityCameraPerceptor._gradient_magnitude(gray)
            score = np.abs(grad - reference["median_grad"])
            return cv2.GaussianBlur(score, (0, 0), 5)

        def lab_gradient_combo(frame: np.ndarray) -> np.ndarray:
            lab_score = lab_ab_delta(frame)
            grad_score = gradient_delta(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frame_norm = gray - cv2.GaussianBlur(gray, (0, 0), 31)
            ref_norm = reference["median_gray"] - cv2.GaussianBlur(reference["median_gray"], (0, 0), 31)
            norm_score = np.abs(frame_norm - ref_norm)
            return cv2.GaussianBlur((0.55 * lab_score) + (0.3 * grad_score) + (0.15 * norm_score), (0, 0), 5)

        def low_freq_foreground(frame: np.ndarray) -> np.ndarray:
            frame_blur = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), 9)
            diff = np.abs(frame_blur - reference["median_bgr"])
            score = np.mean(diff, axis=2)
            return cv2.GaussianBlur(score, (0, 0), 9)

        def lab_illumination_intersection(frame: np.ndarray) -> np.ndarray:
            lab_score = _normalize_map(lab_ab_delta(frame))
            illum_score = _normalize_map(illumination_normalized_delta(frame))
            score = np.sqrt(np.maximum(lab_score * illum_score, 0.0))
            return cv2.GaussianBlur(score, (0, 0), 5)

        def lab_illumination_min(frame: np.ndarray) -> np.ndarray:
            lab_score = _normalize_map(lab_ab_delta(frame))
            illum_score = _normalize_map(illumination_normalized_delta(frame))
            score = np.minimum(lab_score, illum_score)
            return cv2.GaussianBlur(score, (0, 0), 5)

        def lab_shadow_suppressed(frame: np.ndarray) -> np.ndarray:
            lab_score = _normalize_map(lab_ab_delta(frame))
            illum_score = _normalize_map(illumination_normalized_delta(frame))
            low_freq_score = _normalize_map(low_freq_foreground(frame))
            shadow_penalty = np.clip(low_freq_score - illum_score, 0.0, 1.0)
            score = np.clip((0.8 * lab_score) + (0.35 * illum_score) - (0.45 * shadow_penalty), 0.0, 1.0)
            return cv2.GaussianBlur(score, (0, 0), 5)

        def baseline_suppressed_combo(frame: np.ndarray) -> np.ndarray:
            lab_score = _normalize_map(lab_ab_delta(frame))
            illum_score = _normalize_map(illumination_normalized_delta(frame))
            combo = np.sqrt(np.maximum(lab_score * illum_score, 0.0))
            score = combo * reference["reliability_map"]
            return cv2.GaussianBlur(score, (0, 0), 5)

        return {
            "lab_ab_delta": lab_ab_delta,
            "illumination_normalized_delta": illumination_normalized_delta,
            "gradient_delta": gradient_delta,
            "lab_gradient_combo": lab_gradient_combo,
            "low_freq_foreground": low_freq_foreground,
            "lab_illumination_intersection": lab_illumination_intersection,
            "lab_illumination_min": lab_illumination_min,
            "lab_shadow_suppressed": lab_shadow_suppressed,
            "baseline_suppressed_combo": baseline_suppressed_combo,
        }

    @staticmethod
    def _gradient_magnitude(gray: np.ndarray) -> np.ndarray:
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(grad_x, grad_y)

    @staticmethod
    def _summarize_experiment_map(score_map: np.ndarray) -> Dict[str, Any]:
        normalized = cv2.normalize(score_map, None, 0.0, 1.0, cv2.NORM_MINMAX)
        high_threshold = float(np.percentile(normalized, 99.25))
        support_threshold = float(np.percentile(normalized, 97.5))
        seed_mask = normalized >= max(high_threshold, 0.55)
        support_mask = normalized >= max(support_threshold, 0.35)
        components = SecurityCameraPerceptor._extract_map_components(seed_mask, support_mask, normalized)
        top_regions = components[:5]
        top_region_score = float(top_regions[0]["region_score"]) if top_regions else 0.0
        top_region_area = int(top_regions[0]["pixel_count"]) if top_regions else 0
        return {
            "map_p95": round(float(np.percentile(normalized, 95)), 4),
            "map_p99": round(float(np.percentile(normalized, 99)), 4),
            "high_threshold": round(max(high_threshold, 0.55), 4),
            "support_threshold": round(max(support_threshold, 0.35), 4),
            "top_region_score": round(top_region_score, 4),
            "top_region_area": top_region_area,
            "top_regions": top_regions,
        }

    @staticmethod
    def _extract_map_components(
        seed_mask: np.ndarray,
        support_mask: np.ndarray,
        score_map: np.ndarray,
    ) -> List[Dict[str, Any]]:
        rows, cols = score_map.shape
        visited = np.zeros_like(seed_mask, dtype=bool)
        components: List[Dict[str, Any]] = []
        for row in range(rows):
            for col in range(cols):
                if not seed_mask[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                pixels: List[Tuple[int, int]] = []
                while stack:
                    current_row, current_col = stack.pop()
                    pixels.append((current_row, current_col))
                    for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                            continue
                        if visited[next_row, next_col] or not support_mask[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
                pixel_count = len(pixels)
                ys = [pixel[0] for pixel in pixels]
                xs = [pixel[1] for pixel in pixels]
                values = np.array([float(score_map[y, x]) for y, x in pixels], dtype=np.float32)
                bbox_area = max(1, (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1))
                fill_ratio = pixel_count / float(bbox_area)
                region_score = float(np.mean(values) * (pixel_count ** 0.35) * (0.5 + fill_ratio))
                components.append(
                    {
                        "x0": int(min(xs)),
                        "y0": int(min(ys)),
                        "x1": int(max(xs) + 1),
                        "y1": int(max(ys) + 1),
                        "width": int(max(xs) - min(xs) + 1),
                        "height": int(max(ys) - min(ys) + 1),
                        "centroid_x": round(float(np.mean(xs)), 2),
                        "centroid_y": round(float(np.mean(ys)), 2),
                        "pixel_count": int(pixel_count),
                        "fill_ratio": round(float(fill_ratio), 4),
                        "score_max": round(float(np.max(values)), 4),
                        "score_mean": round(float(np.mean(values)), 4),
                        "region_score": round(region_score, 4),
                    }
                )
        components.sort(key=lambda item: item["region_score"], reverse=True)
        return components

    @staticmethod
    def _experiment_label_summary(
        by_label: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for label, items in sorted(by_label.items()):
            top_scores = [item["top_region_score"] for item in items]
            areas = [item["top_region_area"] for item in items]
            summary[label] = {
                "count": len(items),
                "top_region_score_avg": round(float(np.mean(top_scores)), 4),
                "top_region_score_p95": round(float(np.percentile(top_scores, 95)), 4),
                "top_region_score_max": round(float(np.max(top_scores)), 4),
                "top_region_area_avg": round(float(np.mean(areas)), 2),
                "top_examples": sorted(items, key=lambda item: item["top_region_score"], reverse=True)[:3],
            }
        return summary

    @staticmethod
    def _build_nuisance_heatmap(
        *,
        frames: List[np.ndarray],
        median_bgr: np.ndarray,
        median_lab: np.ndarray,
        median_gray: np.ndarray,
        median_blur: np.ndarray,
    ) -> np.ndarray:
        nuisance_stack: List[np.ndarray] = []
        for frame in frames:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
            diff_a = np.abs(lab[:, :, 1] - median_lab[:, :, 1])
            diff_b = np.abs(lab[:, :, 2] - median_lab[:, :, 2])
            lab_score = cv2.GaussianBlur(diff_a + diff_b, (0, 0), 7)
            lab_norm = cv2.normalize(lab_score, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frame_norm = gray - cv2.GaussianBlur(gray, (0, 0), 21)
            illum_score = cv2.GaussianBlur(np.abs(frame_norm - (median_gray - median_blur)), (0, 0), 7)
            illum_norm = cv2.normalize(illum_score, None, 0.0, 1.0, cv2.NORM_MINMAX).astype(np.float32)

            combo = np.sqrt(np.maximum(lab_norm * illum_norm, 0.0))
            nuisance_stack.append(combo)
        nuisance = np.stack(nuisance_stack, axis=0)
        nuisance_mean = np.mean(nuisance, axis=0)
        nuisance_p95 = np.percentile(nuisance, 95, axis=0)
        nuisance_hot = np.mean(nuisance >= 0.55, axis=0)
        combined = (0.45 * nuisance_mean) + (0.35 * nuisance_p95) + (0.2 * nuisance_hot)
        return cv2.GaussianBlur(np.clip(combined, 0.0, 1.0).astype(np.float32), (0, 0), 9)

    @staticmethod
    def _build_spatial_prior(*, height: int, width: int) -> np.ndarray:
        y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
        top_penalty = np.clip((0.18 - y) / 0.18, 0.0, 1.0)
        bottom_penalty = np.clip((y - 0.9) / 0.1, 0.0, 1.0)
        edge_penalty = np.clip((x - 0.92) / 0.08, 0.0, 1.0) * np.clip((y - 0.78) / 0.22, 0.0, 1.0)
        prior = 1.0 - (0.55 * top_penalty) - (0.3 * bottom_penalty) - (0.35 * edge_penalty)
        return np.clip(prior, 0.25, 1.0).astype(np.float32)

    @staticmethod
    def _write_nuisance_baseline_visualization(
        path: Path,
        *,
        reference: Dict[str, np.ndarray],
    ) -> str:
        mean_image = np.clip(reference["median_bgr"], 0, 255).astype(np.uint8)
        nuisance_heat = cv2.normalize(reference["nuisance_heatmap"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        reliability_heat = cv2.normalize(reference["reliability_map"], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        nuisance_panel = cv2.addWeighted(mean_image, 0.55, cv2.applyColorMap(nuisance_heat, cv2.COLORMAP_TURBO), 0.45, 0.0)
        reliability_panel = cv2.addWeighted(mean_image, 0.55, cv2.applyColorMap(reliability_heat, cv2.COLORMAP_VIRIDIS), 0.45, 0.0)
        SecurityCameraPerceptor._put_panel_label(mean_image, "Median background")
        SecurityCameraPerceptor._put_panel_label(nuisance_panel, "Nuisance heatmap")
        SecurityCameraPerceptor._put_panel_label(reliability_panel, "Reliability prior")
        combined = cv2.hconcat([mean_image, nuisance_panel, reliability_panel])
        SecurityCameraPerceptor._put_title(combined, "Best-Effort Baseline")
        return SecurityCameraPerceptor._write_visualization_image(str(path), combined)

    def _classifier_records(
        self,
        image_paths: List[Path],
        method: Callable[[np.ndarray], np.ndarray],
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for path in image_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            score_map = method(frame)
            metrics = self._summarize_experiment_map(score_map)
            label = self._label_from_filename(path)
            top_region = self._select_plausible_region(
                metrics["top_regions"],
                image_height=int(frame.shape[0]),
                image_width=int(frame.shape[1]),
            )
            plausible_score = 0.0
            plausible_area = 0
            plausible_height = 0
            plausible_width = 0
            plausible_y = None
            if top_region is not None:
                plausible_score = float(top_region["region_score"])
                plausible_area = int(top_region["pixel_count"])
                plausible_height = int(top_region["height"])
                plausible_width = int(top_region["width"])
                plausible_y = float(top_region["centroid_y"])
            records.append(
                {
                    "image_name": path.name,
                    "label": label,
                    "plausible_score": round(plausible_score, 4),
                    "plausible_area": plausible_area,
                    "plausible_height": plausible_height,
                    "plausible_width": plausible_width,
                    "plausible_y": plausible_y,
                    "top_regions": metrics["top_regions"],
                }
            )
        records.sort(key=lambda item: item["image_name"])
        return records

    @staticmethod
    def _select_plausible_region(
        regions: List[Dict[str, Any]],
        *,
        image_height: int,
        image_width: int,
    ) -> Dict[str, Any] | None:
        best_region: Dict[str, Any] | None = None
        best_score = -1.0
        for region in regions:
            y_norm = float(region["centroid_y"]) / max(float(image_height), 1.0)
            x_norm = float(region["centroid_x"]) / max(float(image_width), 1.0)
            width = float(region["width"])
            height = float(region["height"])
            aspect = width / max(height, 1.0)
            if y_norm < 0.12:
                vertical_weight = 0.15
            elif y_norm < 0.28:
                vertical_weight = 0.15 + (0.85 * ((y_norm - 0.12) / 0.16))
            elif y_norm <= 0.86:
                vertical_weight = 1.0
            elif y_norm <= 0.96:
                vertical_weight = 1.0 - (0.45 * ((y_norm - 0.86) / 0.10))
            else:
                vertical_weight = 0.2
            edge_weight = 1.0 - (0.35 * max(0.0, x_norm - 0.93) / 0.07)
            size_weight = 0.8 + (0.2 * min(height / 60.0, 1.0))
            shape_weight = 1.0
            if aspect > 5.0 and y_norm > 0.82:
                shape_weight *= 0.35
            if aspect > 6.5:
                shape_weight *= 0.5
            plausible_score = float(region["region_score"]) * vertical_weight * edge_weight * size_weight * shape_weight
            if plausible_score > best_score:
                best_score = plausible_score
                best_region = {**region, "plausible_score": round(plausible_score, 4)}
        return best_region

    @staticmethod
    def _fit_positive_threshold(records: List[Dict[str, Any]]) -> float:
        thresholds = sorted({float(record["plausible_score"]) for record in records})
        if not thresholds:
            return 0.0
        candidates = sorted(set(thresholds + [(left + right) / 2.0 for left, right in zip(thresholds, thresholds[1:])]))
        best_threshold = candidates[0]
        best_tuple = (-1.0, -1.0)
        for threshold in candidates:
            tp = sum(1 for record in records if record["label"] != "CLEAR" and float(record["plausible_score"]) >= threshold)
            fp = sum(1 for record in records if record["label"] == "CLEAR" and float(record["plausible_score"]) >= threshold)
            tn = sum(1 for record in records if record["label"] == "CLEAR" and float(record["plausible_score"]) < threshold)
            fn = sum(1 for record in records if record["label"] != "CLEAR" and float(record["plausible_score"]) < threshold)
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            balanced = (recall + specificity) / 2.0
            accuracy = (tp + tn) / max(1, len(records))
            candidate_tuple = (balanced, accuracy)
            if candidate_tuple > best_tuple:
                best_tuple = candidate_tuple
                best_threshold = threshold
        return float(best_threshold)

    @staticmethod
    def _fit_type_threshold(records: List[Dict[str, Any]], *, positive_threshold: float) -> float:
        positives = [
            record
            for record in records
            if record["label"] != "CLEAR" and float(record["plausible_score"]) >= positive_threshold
        ]
        if not positives:
            return 0.0
        areas = sorted({float(record["plausible_area"]) for record in positives})
        candidates = sorted(set(areas + [(left + right) / 2.0 for left, right in zip(areas, areas[1:])]))
        best_threshold = candidates[0] if candidates else 0.0
        best_accuracy = -1.0
        for threshold in candidates:
            correct = 0
            for record in positives:
                predicted = "PERSON" if float(record["plausible_area"]) < threshold else "MOWER"
                actual = "PERSON" if record["label"] == "PERSON" else "MOWER"
                if predicted == actual:
                    correct += 1
            accuracy = correct / max(1, len(positives))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        return float(best_threshold)

    @staticmethod
    def _apply_classifier_thresholds(
        records: List[Dict[str, Any]],
        *,
        positive_threshold: float,
        type_threshold: float,
    ) -> List[Dict[str, Any]]:
        classified: List[Dict[str, Any]] = []
        for record in records:
            score = float(record["plausible_score"])
            area = float(record["plausible_area"])
            if score < positive_threshold:
                predicted_label = "CLEAR"
            else:
                predicted_label = "PERSON" if area < type_threshold else "MOWER"
            classified.append({**record, "predicted_label": predicted_label})
        return classified

    @staticmethod
    def _classifier_summary(records: List[Dict[str, Any]], *, score_key: str) -> Dict[str, Any]:
        overall_accuracy = sum(1 for record in records if record["predicted_label"] == ("MOWER" if "MOWER" in record["label"] else record["label"])) / max(1, len(records))
        per_label: Dict[str, Dict[str, Any]] = {}
        for label in sorted({record["label"] for record in records}):
            items = [record for record in records if record["label"] == label]
            correct = 0
            confusion: Dict[str, int] = defaultdict(int)
            for item in items:
                normalized_label = "MOWER" if "MOWER" in item["label"] else item["label"]
                if item["predicted_label"] == normalized_label:
                    correct += 1
                confusion[item["predicted_label"]] += 1
            per_label[label] = {
                "count": len(items),
                "accuracy": round(correct / max(1, len(items)), 4),
                "avg_score": round(float(np.mean([float(item[score_key]) for item in items])), 4),
                "predictions": dict(sorted(confusion.items())),
            }
        return {
            "overall_accuracy": round(overall_accuracy, 4),
            "per_label": per_label,
        }

    @staticmethod
    def _write_classifier_comparison_visualization(
        path: Path,
        *,
        frame: np.ndarray,
        raw_map: np.ndarray,
        raw_record: Dict[str, Any],
        suppressed_map: np.ndarray,
        suppressed_record: Dict[str, Any],
        title: str,
    ) -> str:
        original = frame.copy()
        SecurityCameraPerceptor._put_panel_label(original, f"Truth: {raw_record['label']}")
        raw_heat = cv2.addWeighted(
            frame,
            0.55,
            cv2.applyColorMap(cv2.normalize(raw_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO),
            0.45,
            0.0,
        )
        SecurityCameraPerceptor._draw_region_overlays(raw_heat, raw_record["top_regions"], color=(0, 165, 255))
        SecurityCameraPerceptor._put_panel_label(
            raw_heat,
            f"Without baseline: {raw_record['predicted_label']} ({raw_record['plausible_score']:.1f})",
        )
        suppressed_heat = cv2.addWeighted(
            frame,
            0.55,
            cv2.applyColorMap(cv2.normalize(suppressed_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_TURBO),
            0.45,
            0.0,
        )
        SecurityCameraPerceptor._draw_region_overlays(suppressed_heat, suppressed_record["top_regions"], color=(0, 220, 0))
        SecurityCameraPerceptor._put_panel_label(
            suppressed_heat,
            f"With baseline: {suppressed_record['predicted_label']} ({suppressed_record['plausible_score']:.1f})",
        )
        combined = cv2.hconcat([original, raw_heat, suppressed_heat])
        SecurityCameraPerceptor._put_title(combined, title)
        return SecurityCameraPerceptor._write_visualization_image(str(path), combined)

    @staticmethod
    def _label_from_filename(path: Path) -> str:
        stem = path.stem
        match = re.search(r" - ([A-Z][A-Z _]+)$", stem)
        if not match:
            return "CLEAR"
        suffix = match.group(1).strip()
        if not suffix:
            return "CLEAR"
        return suffix.upper().replace(" ", "_")

    @staticmethod
    def _compute_patch_texture_descriptors(
        frame: np.ndarray,
        *,
        patch_size: int,
    ) -> Dict[str, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        normalized = SecurityCameraPerceptor._normalize_local_contrast(gray)
        grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)
        normalized_u8 = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        edges = cv2.Canny(normalized_u8, 50, 150).astype(np.float32) / 255.0
        lbp = SecurityCameraPerceptor._lbp_image(normalized_u8)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_a = lab[:, :, 1]
        lab_b = lab[:, :, 2]

        height, width = gray.shape
        patch_rows = max(1, height // patch_size)
        patch_cols = max(1, width // patch_size)
        effective_height = patch_rows * patch_size
        effective_width = patch_cols * patch_size

        grad_mag = grad_mag[:effective_height, :effective_width]
        edges = edges[:effective_height, :effective_width]
        lbp = lbp[:effective_height, :effective_width]
        lab_a = lab_a[:effective_height, :effective_width]
        lab_b = lab_b[:effective_height, :effective_width]

        grad_mean = grad_mag.reshape(patch_rows, patch_size, patch_cols, patch_size).mean(axis=(1, 3))
        grad_std = grad_mag.reshape(patch_rows, patch_size, patch_cols, patch_size).std(axis=(1, 3))
        edge_density = edges.reshape(patch_rows, patch_size, patch_cols, patch_size).mean(axis=(1, 3))
        lab_a_mean = lab_a.reshape(patch_rows, patch_size, patch_cols, patch_size).mean(axis=(1, 3))
        lab_b_mean = lab_b.reshape(patch_rows, patch_size, patch_cols, patch_size).mean(axis=(1, 3))
        lbp_hist = SecurityCameraPerceptor._patch_lbp_histograms(
            lbp,
            patch_rows=patch_rows,
            patch_cols=patch_cols,
            patch_size=patch_size,
        )
        return {
            "grad_mean": grad_mean.astype(np.float32),
            "grad_std": grad_std.astype(np.float32),
            "edge_density": edge_density.astype(np.float32),
            "lab_a_mean": lab_a_mean.astype(np.float32),
            "lab_b_mean": lab_b_mean.astype(np.float32),
            "lbp_hist": lbp_hist.astype(np.float32),
        }

    @staticmethod
    def _normalize_local_contrast(gray: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(gray, (0, 0), 15)
        high_pass = gray - blurred
        local_std = cv2.GaussianBlur(high_pass * high_pass, (0, 0), 15)
        local_std = np.sqrt(np.maximum(local_std, 1.0))
        return high_pass / local_std

    @staticmethod
    def _lbp_image(gray_u8: np.ndarray) -> np.ndarray:
        center = gray_u8[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ]
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = gray_u8[1 + dy : gray_u8.shape[0] - 1 + dy, 1 + dx : gray_u8.shape[1] - 1 + dx]
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)
        padded = np.zeros_like(gray_u8, dtype=np.uint8)
        padded[1:-1, 1:-1] = lbp
        return padded

    @staticmethod
    def _patch_lbp_histograms(
        lbp: np.ndarray,
        *,
        patch_rows: int,
        patch_cols: int,
        patch_size: int,
        bins: int = 16,
    ) -> np.ndarray:
        hist = np.zeros((patch_rows, patch_cols, bins), dtype=np.float32)
        bin_edges = np.linspace(0, 256, bins + 1, dtype=np.int32)
        for row in range(patch_rows):
            for col in range(patch_cols):
                patch = lbp[
                    row * patch_size : (row + 1) * patch_size,
                    col * patch_size : (col + 1) * patch_size,
                ]
                counts, _ = np.histogram(patch, bins=bin_edges)
                counts = counts.astype(np.float32)
                counts /= max(float(np.sum(counts)), 1.0)
                hist[row, col] = counts
        return hist

    @staticmethod
    def _extract_anomalous_regions(
        anomalous: np.ndarray,
        support_mask: np.ndarray,
        score_grid: np.ndarray,
        *,
        patch_size: int,
        image_width: int,
        image_height: int,
    ) -> List[Dict[str, Any]]:
        rows, cols = anomalous.shape
        visited = np.zeros_like(anomalous, dtype=bool)
        regions: List[Dict[str, Any]] = []
        for row in range(rows):
            for col in range(cols):
                if not anomalous[row, col] or visited[row, col]:
                    continue
                stack = [(row, col)]
                visited[row, col] = True
                component: List[Tuple[int, int]] = []
                while stack:
                    current_row, current_col = stack.pop()
                    component.append((current_row, current_col))
                    for delta_row, delta_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        next_row = current_row + delta_row
                        next_col = current_col + delta_col
                        if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                            continue
                        if visited[next_row, next_col] or not support_mask[next_row, next_col]:
                            continue
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))
                box_rows = [item[0] for item in component]
                box_cols = [item[1] for item in component]
                x0 = min(box_cols) * patch_size
                y0 = min(box_rows) * patch_size
                x1 = min(image_width, (max(box_cols) + 1) * patch_size)
                y1 = min(image_height, (max(box_rows) + 1) * patch_size)
                patch_count = int(len(component))
                seed_count = int(sum(1 for item in component if bool(anomalous[item[0], item[1]])))
                component_scores = np.array([float(score_grid[item[0], item[1]]) for item in component], dtype=np.float32)
                coords = np.array(component, dtype=np.float32)
                center = np.mean(coords, axis=0)
                centered = coords - center
                cov = np.cov(centered.T) if patch_count > 1 else np.eye(2, dtype=np.float32) * 1e-3
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.clip(np.sort(eigvals), 1e-3, None)
                major = float(np.sqrt(eigvals[-1]))
                minor = float(np.sqrt(eigvals[0]))
                elongation = float(major / minor)
                bbox_width_patches = int(max(box_cols) - min(box_cols) + 1)
                bbox_height_patches = int(max(box_rows) - min(box_rows) + 1)
                bbox_area_patches = int(bbox_width_patches * bbox_height_patches)
                fill_ratio = float(patch_count / max(bbox_area_patches, 1))
                aspect_ratio = float(max(bbox_width_patches, bbox_height_patches) / max(1, min(bbox_width_patches, bbox_height_patches)))
                peak_score = float(np.max(component_scores))
                mean_score = float(np.mean(component_scores))
                concentration = float(np.sum(component_scores) / max(patch_count ** 0.85, 1.0))
                seed_density = float(seed_count / max(patch_count, 1))
                principal = eigvecs[:, int(np.argmax(eigvals))]
                orientation_deg = float(np.degrees(np.arctan2(principal[0], principal[1])))
                compactness = float(fill_ratio / max(elongation, 1.0))
                size_bonus = float(min(patch_count, 6) / 6.0)
                object_score = (
                    1.8 * peak_score
                    + 1.2 * mean_score
                    + 0.9 * concentration
                    + 6.0 * fill_ratio
                    + 3.5 * compactness
                    + 4.0 * seed_density
                    + 3.0 * size_bonus
                    - 1.4 * max(0.0, elongation - 1.8)
                    - 0.6 * max(0.0, aspect_ratio - 2.0)
                    - 3.5 * max(0, 2 - patch_count)
                )
                regions.append(
                    {
                        "x0": int(x0),
                        "y0": int(y0),
                        "x1": int(x1),
                        "y1": int(y1),
                        "patch_count": patch_count,
                        "seed_count": seed_count,
                        "patch_rows": [int(value) for value in sorted(set(box_rows))],
                        "patch_cols": [int(value) for value in sorted(set(box_cols))],
                        "score_max": round(peak_score, 4),
                        "score_mean": round(mean_score, 4),
                        "score_sum": round(float(np.sum(component_scores)), 4),
                        "fill_ratio": round(fill_ratio, 4),
                        "aspect_ratio": round(aspect_ratio, 4),
                        "elongation": round(elongation, 4),
                        "compactness": round(compactness, 4),
                        "concentration": round(concentration, 4),
                        "seed_density": round(seed_density, 4),
                        "orientation_deg": round(orientation_deg, 2),
                        "object_score": round(float(object_score), 4),
                    }
                )
        regions.sort(key=lambda item: item["object_score"], reverse=True)
        return regions

    @staticmethod
    def _split_object_like_regions(
        regions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        for region in regions:
            patch_count = int(region["patch_count"])
            elongation = float(region["elongation"])
            aspect_ratio = float(region["aspect_ratio"])
            fill_ratio = float(region["fill_ratio"])
            concentration = float(region["concentration"])
            peak_score = float(region["score_max"])
            is_object_like = (
                patch_count >= 2
                and peak_score >= 3.4
                and concentration >= 4.4
                and fill_ratio >= 0.22
                and float(region["seed_density"]) >= 0.18
                and elongation <= 4.8
                and aspect_ratio <= 5.5
                and not (patch_count >= 6 and elongation >= 3.2 and fill_ratio <= 0.38)
            )
            if not is_object_like and patch_count == 1:
                is_object_like = (
                    peak_score >= 8.8
                    and concentration >= 8.8
                    and fill_ratio >= 0.85
                )
            if is_object_like:
                accepted.append(region)
            else:
                rejected.append(region)
        accepted.sort(key=lambda item: item["object_score"], reverse=True)
        rejected.sort(key=lambda item: item["object_score"], reverse=True)
        return accepted[:12], rejected[:12]

    @staticmethod
    def _scene_score_from_grid(score_grid: np.ndarray) -> float:
        flat = np.sort(score_grid.reshape(-1))
        if flat.size == 0:
            return 0.0
        top_k = max(4, min(24, flat.size // 50))
        top_mean = float(np.mean(flat[-top_k:]))
        p99 = float(np.percentile(flat, 99))
        p95 = float(np.percentile(flat, 95))
        return (0.5 * top_mean) + (0.3 * p99) + (0.2 * p95)

    @staticmethod
    def _fine_anomaly_threshold(score_grid: np.ndarray, model: Dict[str, np.ndarray]) -> float:
        baseline_floor = float(np.percentile(model["fine_patch_variability"], 95) / 4.0)
        score_floor = float(np.percentile(score_grid, 88))
        return max(3.1, baseline_floor, min(score_floor, 6.5))

    @staticmethod
    def _pedestrian_score_from_regions(regions: List[Dict[str, Any]]) -> float:
        if not regions:
            return 0.0
        top_regions = regions[:3]
        top1 = float(top_regions[0]["object_score"])
        top_mean = float(np.mean([region["object_score"] for region in top_regions]))
        concentration_mean = float(np.mean([region["concentration"] for region in top_regions]))
        compactness_mean = float(np.mean([region["compactness"] for region in top_regions]))
        return (0.45 * top1) + (0.25 * top_mean) + (0.2 * concentration_mean) + (2.0 * compactness_mean)

    @staticmethod
    def _linear_change_score_from_regions(regions: List[Dict[str, Any]]) -> float:
        if not regions:
            return 0.0
        top_regions = regions[:3]
        return float(
            np.mean(
                [
                    float(region["score_mean"]) + max(0.0, float(region["elongation"]) - 1.0)
                    for region in top_regions
                ]
            )
        )

    @staticmethod
    def _threshold_summary(records: List[Dict[str, Any]], *, score_key: str) -> Dict[str, Any]:
        labeled = [
            {
                "score": float(record[score_key]),
                "is_positive": record["label"] != "CLEAR",
            }
            for record in records
        ]
        positives = [item for item in labeled if item["is_positive"]]
        negatives = [item for item in labeled if not item["is_positive"]]
        if not positives or not negatives:
            return {"error": "need both CLEAR and obstacle-labeled images for threshold evaluation"}

        candidate_scores = sorted({item["score"] for item in labeled})
        thresholds = []
        thresholds.extend(candidate_scores)
        for left, right in zip(candidate_scores, candidate_scores[1:]):
            thresholds.append((left + right) / 2.0)
        thresholds = sorted(set(thresholds))

        best_accuracy = None
        best_f1 = None
        for threshold in thresholds:
            tp = sum(1 for item in labeled if item["is_positive"] and item["score"] >= threshold)
            fp = sum(1 for item in labeled if (not item["is_positive"]) and item["score"] >= threshold)
            tn = sum(1 for item in labeled if (not item["is_positive"]) and item["score"] < threshold)
            fn = sum(1 for item in labeled if item["is_positive"] and item["score"] < threshold)

            total = tp + tn + fp + fn
            accuracy = (tp + tn) / total if total else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            balanced_accuracy = (recall + specificity) / 2.0
            summary = {
                "threshold": round(float(threshold), 4),
                "accuracy": round(float(accuracy), 4),
                "balanced_accuracy": round(float(balanced_accuracy), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
            if best_accuracy is None or (
                summary["accuracy"], summary["balanced_accuracy"], summary["f1"]
            ) > (
                best_accuracy["accuracy"],
                best_accuracy["balanced_accuracy"],
                best_accuracy["f1"],
            ):
                best_accuracy = summary
            if best_f1 is None or (
                summary["f1"], summary["balanced_accuracy"], summary["accuracy"]
            ) > (
                best_f1["f1"],
                best_f1["balanced_accuracy"],
                best_f1["accuracy"],
            ):
                best_f1 = summary

        return {
            "score_key": score_key,
            "clear_avg": round(float(np.mean([item["score"] for item in negatives])), 4),
            "positive_avg": round(float(np.mean([item["score"] for item in positives])), 4),
            "best_accuracy_threshold": best_accuracy,
            "best_f1_threshold": best_f1,
        }

    @staticmethod
    def _write_debug_frame(frame: Any, save_frame_path: str) -> str:
        path = Path(save_frame_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Failed to write debug frame to {path}")
        return str(path.resolve())


ClassicSecurityCameraPerceptor = SecurityCameraPerceptor


def camera_perceptor_for_backend(backend: str) -> Any:
    """Backward-compatible alias for the classical backend."""
    normalized = (backend or "classic").strip().lower()
    if normalized != "classic":
        raise ValueError(f"Unknown classical camera backend: {backend}")
    return ClassicSecurityCameraPerceptor()
