"""
Frame-to-video renderer for thesis experiments.

Based on: Bench2Drive/tools/generate_video.py
Changes from original:
- CLI arguments instead of hardcoded paths
- Safer handling for missing frames/meta
- Optional commentary overlay if `meta['commentary']` exists
"""

import argparse
import json
import re
import shutil
import subprocess
from statistics import median
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm


PREFERRED_CODEC = 'avc1'
FALLBACK_CODEC = 'mp4v'
COMMENT_MAX_CHARS = 30
COMMENT_MAX_LINES = 8

# ===================== Tunable layout variables =====================
# Fixed separator gap for LEFT/RIGHT and between row2/row3 views.
VIEW_GAP_PX = 2
# Dedicated gap between FRONT row and LEFT/RIGHT row.
ROW12_GAP_PX = 2
# Size of LEFT/RIGHT/REAR views relative to front-based baseline.
SECONDARY_VIEW_SCALE = 1.6
# Front-bottom black-padding detection and trim behavior.
FRONT_TRIM_THRESHOLD = 8
FRONT_TRIM_MAX_SCAN_RATIO = 0.5
# Keep this many pixels of bottom margin after front trim.
FRONT_TRIM_KEEP_MARGIN_PX = 12
# ===================================================================

VIEW_DIRS = {
    'front': 'rgb_front',
    'left': 'rgb_left',
    'right': 'rgb_right',
    'rear': 'rgb_rear',
}
LABEL_STYLE = {
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'scale': 0.6,
    'thickness': 1,
    'color': (255, 255, 255),
    'pad_y': 20,
}


def _resize_with_letterbox(img, target_width, target_height):
    src_height, src_width = img.shape[:2]
    if src_width == target_width and src_height == target_height:
        return img

    scale = min(target_width / src_width, target_height / src_height)
    resized_width = max(1, int(round(src_width * scale)))
    resized_height = max(1, int(round(src_height * scale)))

    resized = cv2.resize(img, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized
    return canvas


def _trim_black_borders(img, threshold: int = 8):
    if img is None or img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        return img
    ys, xs = np.where(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    if y1 <= y0 or x1 <= x0:
        return img
    return img[y0:y1 + 1, x0:x1 + 1]


def _count_bottom_black_rows(img, threshold: int = FRONT_TRIM_THRESHOLD, max_scan_ratio: float = FRONT_TRIM_MAX_SCAN_RATIO):
    if img is None or img.size == 0:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = gray.shape[0]
    max_scan = max(1, int(round(h * max_scan_ratio)))
    count = 0
    for y in range(h - 1, max(-1, h - 1 - max_scan), -1):
        if float(np.mean(gray[y])) <= threshold:
            count += 1
        else:
            break
    return count


def _trim_bottom_rows(img, rows: int):
    if img is None or img.size == 0 or rows <= 0:
        return img
    h = img.shape[0]
    keep_h = max(1, h - rows)
    return img[:keep_h, :]


def _letterbox_fit(src_width: int, src_height: int, target_width: int, target_height: int):
    scale = min(target_width / src_width, target_height / src_height)
    fitted_w = max(1, int(round(src_width * scale)))
    fitted_h = max(1, int(round(src_height * scale)))
    x_offset = (target_width - fitted_w) // 2
    y_offset = (target_height - fitted_h) // 2
    return x_offset, y_offset, fitted_w, fitted_h


def _make_even(value: int):
    return value if value % 2 == 0 else value + 1


def _compute_secondary_panel_size(front_width: int, target_aspect: float):
    base_panel_w = max(1, (front_width - VIEW_GAP_PX) // 2)
    panel_w = max(1, int(round(base_panel_w * SECONDARY_VIEW_SCALE)))
    panel_h = max(1, int(round(panel_w / max(0.5, target_aspect))))
    return panel_w, panel_h


def _parse_scenario_seed(images_folder: Path):
    match = re.search(r'route(\d+)_seed(\d+)', images_folder.name)
    if match:
        return match.group(1), match.group(2)
    for parent in images_folder.parents:
        match = re.search(r'route(\d+)_seed(\d+)', parent.name)
        if match:
            return match.group(1), match.group(2)
    return None, None


def _start_ffmpeg_writer(output_video: Path, width: int, height: int, fps: int, crf: int, preset: str):
    if shutil.which('ffmpeg') is None:
        return None

    cmd = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s:v', f'{width}x{height}',
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-c:v', 'libx264',
        '-preset', preset,
        '-crf', str(crf),
        '-pix_fmt', 'yuv420p',
        str(output_video),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _open_video_writer(output_video: Path, output_width: int, height: int, fps: int, hq_crf: int, hq_preset: str):
    ffmpeg_proc = _start_ffmpeg_writer(output_video, output_width, height, fps, hq_crf, hq_preset)
    if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
        print(f'[video] using codec=libx264(crf={hq_crf},preset={hq_preset}) output={output_video}')
        return ffmpeg_proc, None

    used_codec = PREFERRED_CODEC
    fourcc = cv2.VideoWriter_fourcc(*PREFERRED_CODEC)
    video = cv2.VideoWriter(str(output_video), fourcc, fps, (output_width, height))
    if not video.isOpened():
        used_codec = FALLBACK_CODEC
        fourcc = cv2.VideoWriter_fourcc(*FALLBACK_CODEC)
        video = cv2.VideoWriter(str(output_video), fourcc, fps, (output_width, height))
    if not video.isOpened():
        raise RuntimeError(f'Failed to open output video writer: {output_video}')

    print(f'[video] ffmpeg unavailable; using codec={used_codec} output={output_video}')
    return None, video


def _draw_telemetry_panel(scene_img, telemetry_x, telemetry_width, panel_height, meta, font_scale, text_color, text_position, overlay_context=None):
    steer = float(meta.get('steer', 0.0))
    throttle = float(meta.get('throttle', 0.0))
    brake = float(meta.get('brake', 0.0))
    speed = float(meta.get('speed', 0.0))
    overlay_context = overlay_context or {}

    if telemetry_width > 0:
        cv2.rectangle(
            scene_img,
            (max(0, telemetry_x), 0),
            (max(0, telemetry_x + telemetry_width - 1), max(0, panel_height - 1)),
            (0, 0, 0),
            -1,
        )

    overlay_scale = max(0.5, font_scale * 0.85)
    text_x = telemetry_x + max(10, min(text_position[0], max(10, telemetry_width - 20)))
    text_y = max(20, text_position[1])
    line_height = max(20, int(round(26 * overlay_scale)))
    telemetry_lines = []
    scenario_id = overlay_context.get('scenario_id')
    seed = overlay_context.get('seed')
    step_idx = overlay_context.get('sim_step')
    total_steps = overlay_context.get('sim_step_total')
    sim_time_s = overlay_context.get('sim_time_s')

    if scenario_id is not None:
        telemetry_lines.append(f'scenario: {scenario_id}')
    if seed is not None:
        telemetry_lines.append(f'seed: {seed}')
    if step_idx is not None and total_steps is not None:
        telemetry_lines.append(f'step: {step_idx}/{total_steps}')
    if sim_time_s is not None:
        telemetry_lines.append(f'sim time: {sim_time_s:.2f}s')

    metric_lines = [
        f'speed: {round(speed, 2)}',
        f'steer: {round(steer, 2)}',
        f'throttle: {round(throttle, 2)}',
        f'brake: {round(brake, 2)}',
    ]
    metric_lines = telemetry_lines + metric_lines

    max_text_width = 0
    for line in metric_lines:
        (line_w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, overlay_scale, 2)
        max_text_width = max(max_text_width, line_w)

    box_height = line_height * len(metric_lines) + 14
    box_x0 = max(telemetry_x, text_x - 8)
    box_y0 = max(0, text_y - line_height)
    box_x1 = min(telemetry_x + telemetry_width, box_x0 + max_text_width + 16)
    box_y1 = min(panel_height, box_y0 + box_height)
    cv2.rectangle(scene_img, (box_x0, box_y0), (box_x1, box_y1), (0, 0, 0), -1)

    for idx, line in enumerate(metric_lines):
        y = text_y + idx * line_height
        if y < panel_height - 2:
            cv2.putText(scene_img, line, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, overlay_scale, text_color, 2, cv2.LINE_AA)

    commentary = meta.get('commentary')
    if not commentary:
        return

    comment_text = str(commentary)
    comment_lines = [comment_text[i:i + COMMENT_MAX_CHARS] for i in range(0, len(comment_text), COMMENT_MAX_CHARS)]
    comment_start = text_y + len(metric_lines) * line_height + 10
    comment_line_height = max(18, int(round(22 * overlay_scale)))
    for idx, line in enumerate(comment_lines[:COMMENT_MAX_LINES]):
        y = comment_start + idx * comment_line_height
        if y < panel_height - 2:
            cv2.putText(scene_img, line, (text_x, y), cv2.FONT_HERSHEY_SIMPLEX, max(0.55, overlay_scale * 0.75), text_color, 2, cv2.LINE_AA)


def _read_view_frame(images_folder: Path, view_name: str, stem: str):
    view_dir = images_folder / VIEW_DIRS[view_name]
    if not view_dir.exists():
        return None
    for suffix in ('.jpg', '.png'):
        candidate = view_dir / f'{stem}{suffix}'
        if candidate.exists():
            img = cv2.imread(str(candidate))
            if img is not None:
                return img
    return None


def _compose_view_grid(front_img, left_img, right_img, rear_img, fixed_small_h: Optional[int] = None, fixed_front_bottom_trim: int = 0):
    height, width = front_img.shape[:2]
    between_gap = VIEW_GAP_PX
    panel_w, default_small_h = _compute_secondary_panel_size(width, width / max(1, height))
    small_h = fixed_small_h if fixed_small_h is not None else default_small_h
    row12_gap = ROW12_GAP_PX
    row23_gap = between_gap

    front_src = _trim_bottom_rows(front_img, fixed_front_bottom_trim)
    left_src = _trim_black_borders(left_img) if left_img is not None else _trim_black_borders(np.zeros_like(front_img))
    right_src = _trim_black_borders(right_img) if right_img is not None else _trim_black_borders(np.zeros_like(front_img))
    rear_src = _trim_black_borders(rear_img) if rear_img is not None else _trim_black_borders(np.zeros_like(front_img))

    left_panel = _resize_with_letterbox(left_src, panel_w, small_h)
    right_panel = _resize_with_letterbox(right_src, panel_w, small_h)
    rear_panel = _resize_with_letterbox(rear_src, panel_w, small_h)

    left_fit = _letterbox_fit(left_src.shape[1], left_src.shape[0], panel_w, small_h)
    right_fit = _letterbox_fit(right_src.shape[1], right_src.shape[0], panel_w, small_h)
    rear_fit = _letterbox_fit(rear_src.shape[1], rear_src.shape[0], panel_w, small_h)
    row1_h = front_src.shape[0]
    front_fit = _letterbox_fit(front_src.shape[1], front_src.shape[0], width, row1_h)

    row1_y0 = 0
    row2_y0 = row1_y0 + row1_h + row12_gap
    row3_y0 = row2_y0 + small_h + row23_gap

    scene_height = _make_even(row3_y0 + small_h)
    row2_total_w = (2 * panel_w) + between_gap
    scene_width = _make_even(max(width, row2_total_w))
    scene = np.zeros((scene_height, scene_width, 3), dtype=np.uint8)

    front_x0 = (scene_width - width) // 2
    scene[0:row1_h, front_x0:front_x0 + width] = _resize_with_letterbox(front_src, width, row1_h)

    row2_x0 = (scene_width - row2_total_w) // 2
    scene[row2_y0:row2_y0 + small_h, row2_x0:row2_x0 + panel_w] = left_panel
    right_x0 = row2_x0 + panel_w + between_gap
    scene[row2_y0:row2_y0 + small_h, right_x0:right_x0 + panel_w] = right_panel

    rear_x0 = (scene_width - panel_w) // 2
    scene[row3_y0:row3_y0 + small_h, rear_x0:rear_x0 + panel_w] = rear_panel

    font = LABEL_STYLE['font']
    scale = LABEL_STYLE['scale']
    thickness = LABEL_STYLE['thickness']
    color = LABEL_STYLE['color']

    def draw_label_centered(img, text, center_x, baseline_y):
        (text_w, _), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, int(round(center_x - text_w / 2)))
        cv2.putText(img, text, (x, baseline_y), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, baseline_y), font, scale, color, thickness, cv2.LINE_AA)

    def label_baseline(y_origin, fit_rect):
        _, y_off, _, fitted_h = fit_rect
        return y_origin + y_off + max(18, min(30, int(round(fitted_h * 0.12))))

    draw_label_centered(
        scene,
        'FRONT',
        front_x0 + front_fit[0] + (front_fit[2] / 2),
        label_baseline(row1_y0, front_fit),
    )
    draw_label_centered(
        scene,
        'LEFT',
        row2_x0 + left_fit[0] + (left_fit[2] / 2),
        label_baseline(row2_y0, left_fit),
    )
    draw_label_centered(
        scene,
        'RIGHT',
        right_x0 + right_fit[0] + (right_fit[2] / 2),
        label_baseline(row2_y0, right_fit),
    )
    draw_label_centered(
        scene,
        'REAR',
        rear_x0 + rear_fit[0] + (rear_fit[2] / 2),
        label_baseline(row3_y0, rear_fit),
    )
    telemetry_x = 0
    telemetry_w = max(0, front_x0)
    return scene, telemetry_x, telemetry_w, row1_h


def _compose_frame(scene_img, front_height: int, telemetry_x: int, telemetry_width: int, meta, font_scale: float, text_color, text_position, overlay_context=None):
    composed = scene_img.copy()
    _draw_telemetry_panel(composed, telemetry_x, telemetry_width, front_height, meta, font_scale, text_color, text_position, overlay_context)
    return composed


def _write_composed_frame(ffmpeg_proc, video, frame):
    if ffmpeg_proc is not None and ffmpeg_proc.stdin is not None:
        ffmpeg_proc.stdin.write(frame.tobytes())
    else:
        video.write(frame)


def _close_writers(ffmpeg_proc, video):
    if ffmpeg_proc is not None:
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        ffmpeg_rc = ffmpeg_proc.wait()
        if ffmpeg_rc != 0:
            err = ffmpeg_proc.stderr.read().decode('utf-8', errors='replace') if ffmpeg_proc.stderr is not None else ''
            raise RuntimeError(f'ffmpeg encoding failed (rc={ffmpeg_rc}):\n{err[-2000:]}')
        return
    if video is not None:
        video.release()


def _infer_output_fps(images, requested_fps: Optional[int], sim_fps: float):
    if requested_fps is not None:
        return max(1, int(requested_fps))

    frame_indices = []
    for image_path in images:
        stem = image_path.stem
        if stem.isdigit():
            frame_indices.append(int(stem))

    if len(frame_indices) < 2:
        return max(1, int(round(sim_fps)))

    frame_indices.sort()
    deltas = [b - a for a, b in zip(frame_indices[:-1], frame_indices[1:]) if b - a > 0]
    if not deltas:
        return max(1, int(round(sim_fps)))

    stride = min(deltas)
    inferred_fps = sim_fps / stride
    return max(1, int(round(inferred_fps)))


def create_video(images_folder: Path, output_video: Path, fps: Optional[int], sim_fps: float, font_scale: float, text_color, text_position, hq_crf: int, hq_preset: str):
    rgb_dir = images_folder / VIEW_DIRS['front']
    meta_dir = images_folder / 'meta'

    images = sorted([img for img in rgb_dir.iterdir() if img.suffix.lower() in {'.jpg', '.png'}])
    if not images:
        raise RuntimeError(f'No RGB frames found in {rgb_dir}')

    output_fps = _infer_output_fps(images, fps, sim_fps)
    scenario_id, seed = _parse_scenario_seed(images_folder)

    frame_items = []
    for image_path in images:
        stem = image_path.stem
        meta_path = meta_dir / f'{stem}.json'
        if meta_path.exists():
            frame_number = int(stem) if stem.isdigit() else None
            frame_items.append((image_path, meta_path, stem, frame_number))

    total_steps = len(frame_items)
    if total_steps == 0:
        raise RuntimeError(f'No frame/meta pairs found in {images_folder}')

    numeric_steps = [item[3] for item in frame_items if item[3] is not None]
    sim_step_total = max(numeric_steps) if numeric_steps else total_steps

    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        raise RuntimeError(f'Failed to read first frame: {images[0]}')
    height, width = first_frame.shape[:2]

    first_stem = images[0].stem
    first_left = _read_view_frame(images_folder, 'left', first_stem)
    first_right = _read_view_frame(images_folder, 'right', first_stem)
    first_rear = _read_view_frame(images_folder, 'rear', first_stem)

    side_aspects = []
    for side_img in (first_left, first_right, first_rear):
        if side_img is None:
            continue
        trimmed = _trim_black_borders(side_img)
        h, w = trimmed.shape[:2]
        if h > 0:
            side_aspects.append(w / h)
    if not side_aspects:
        side_aspects = [width / max(1, height)]

    target_side_aspect = median(side_aspects)
    _, fixed_small_h = _compute_secondary_panel_size(width, target_side_aspect)

    # Instead of basing the front-bottom trim solely on the first frame,
    # scan all available front frames and choose a conservative (minimum)
    # detected black-row count. This avoids over-trimming when the first
    # frame happens to have an unusually large black margin and prevents
    # telemetry/commentary from being cropped on many frames.
    detected_counts = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        detected_counts.append(_count_bottom_black_rows(img))

    if detected_counts:
        # conservative choice: trim at most the smallest detected black-run
        detected_front_bottom_black = int(min(detected_counts))
    else:
        detected_front_bottom_black = _count_bottom_black_rows(first_frame)

    fixed_front_bottom_trim = max(0, detected_front_bottom_black - FRONT_TRIM_KEEP_MARGIN_PX)

    preview_scene, telemetry_x, telemetry_width, front_row_height = _compose_view_grid(
        first_frame,
        first_left,
        first_right,
        first_rear,
        fixed_small_h,
        fixed_front_bottom_trim,
    )
    scene_height, output_width = preview_scene.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_proc, video = _open_video_writer(output_video, output_width, scene_height, output_fps, hq_crf, hq_preset)

    written_frames = 0
    try:
        for step_idx, (image_path, meta_path, stem, frame_number) in enumerate(tqdm(frame_items, desc='Stitching frames'), start=1):

            front_img = cv2.imread(str(image_path))
            if front_img is None:
                continue

            if front_img.shape[1] != width or front_img.shape[0] != height:
                front_img = _resize_with_letterbox(front_img, width, height)

            left_img = _read_view_frame(images_folder, 'left', stem)
            right_img = _read_view_frame(images_folder, 'right', stem)
            rear_img = _read_view_frame(images_folder, 'rear', stem)

            scene_img, telemetry_x, telemetry_width, front_row_height = _compose_view_grid(
                front_img,
                left_img,
                right_img,
                rear_img,
                fixed_small_h,
                fixed_front_bottom_trim,
            )

            with meta_path.open('r', encoding='utf-8') as f:
                meta = json.load(f)

            sim_time_s = (frame_number / sim_fps) if frame_number is not None else ((step_idx - 1) / sim_fps)
            sim_step = frame_number if frame_number is not None else step_idx
            overlay_context = {
                'scenario_id': scenario_id,
                'seed': seed,
                'sim_step': sim_step,
                'sim_step_total': sim_step_total,
                'sim_time_s': sim_time_s,
            }

            composed = _compose_frame(scene_img, front_row_height, telemetry_x, telemetry_width, meta, font_scale, text_color, text_position, overlay_context)
            _write_composed_frame(ffmpeg_proc, video, composed)
            written_frames += 1
    finally:
        _close_writers(ffmpeg_proc, video)

    if written_frames == 0:
        raise RuntimeError(f'No frames were written to {output_video}. Check image/meta alignment in {images_folder}.')


def main():
    parser = argparse.ArgumentParser(description='Generate multiview MP4 from frame folders and meta JSON files.')
    parser.add_argument('--images-folder', required=True, help='Folder containing rgb_front/ and meta/ subfolders')
    parser.add_argument('--output-video', required=True)
    parser.add_argument('--fps', type=int, default=None, help='Output FPS. If omitted, infer from frame index spacing and --sim-fps')
    parser.add_argument('--sim-fps', type=float, default=20.0, help='Simulator FPS used for auto-inferred output FPS (default: 20)')
    parser.add_argument('--font-scale', type=float, default=1.0)
    parser.add_argument('--text-x', type=int, default=50)
    parser.add_argument('--text-y', type=int, default=50)
    parser.add_argument('--hq-crf', type=int, default=18, help='ffmpeg/libx264 quality: lower is better (typical 16-23)')
    parser.add_argument('--hq-preset', default='slow', help='ffmpeg/libx264 preset (ultrafast..veryslow)')
    args = parser.parse_args()

    create_video(
        images_folder=Path(args.images_folder),
        output_video=Path(args.output_video),
        fps=args.fps,
        sim_fps=args.sim_fps,
        font_scale=args.font_scale,
        text_color=(255, 255, 255),
        text_position=(args.text_x, args.text_y),
        hq_crf=args.hq_crf,
        hq_preset=args.hq_preset,
    )


if __name__ == '__main__':
    main()
