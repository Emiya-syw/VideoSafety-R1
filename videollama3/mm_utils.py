import os
from typing import Optional

import cv2
import ffmpeg
import imageio
import numpy as np
from decord import VideoReader, cpu
from PIL import Image

from .constants import MAX_FRAMES, NUM_FRAMES_PER_SECOND


def load_images(image_path):
    if isinstance(image_path, str) and os.path.isfile(image_path):
        return [Image.open(image_path).convert("RGB")]
    if isinstance(image_path, str) and os.path.isdir(image_path):
        return [
            Image.open(os.path.join(image_path, name)).convert("RGB")
            for name in sorted(os.listdir(image_path))
        ]
    if isinstance(image_path, list) and isinstance(image_path[0], str):
        return [Image.open(path).convert("RGB") for path in image_path]
    if isinstance(image_path, list) and isinstance(image_path[0], Image.Image):
        return image_path
    if isinstance(image_path, Image.Image):
        return [image_path]
    raise ValueError(f"Unsupported image path type: {type(image_path)}")


def frame_sample(duration, mode="uniform", num_frames=None, vid_fps=None, fps=None):
    if mode == "uniform":
        if num_frames is None:
            raise ValueError("num_frames is required for uniform sampling.")
        if duration <= num_frames:
            return np.arange(duration).astype(int)
        return np.linspace(0, duration - 1, num_frames, dtype=int)
    if mode == "fps":
        if vid_fps is None:
            raise ValueError("vid_fps is required for FPS sampling.")
        fps = fps if fps is not None else NUM_FRAMES_PER_SECOND
        segment_len = min(vid_fps // fps, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    raise ValueError(f"Unsupported frame sampling mode: {mode}")


def load_video_from_ids(
    video_path,
    start_time=None,
    end_time=None,
    fps=None,
    max_frames=None,
    temporal_factor=1,
):
    if start_time is not None and end_time is not None:
        start_time = max(start_time, 0.0)
        end_time = max(end_time, 0.0)
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.isdir(video_path):
        frame_files = sorted(os.listdir(video_path))
        video_fps = 3
        video_length = len(frame_files)
    elif video_path.endswith(".gif"):
        gif_reader = imageio.get_reader(video_path)
        video_fps = 25
        video_length = len(gif_reader)
    else:
        video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        video_fps = video_reader.get_avg_fps()
        video_length = len(video_reader)

    first_frame = 0 if start_time is None else max(int(start_time * video_fps) - 1, 0)
    last_frame = (
        video_length - 1
        if end_time is None
        else min(int(end_time * video_fps) - 1, video_length - 1)
    )
    frame_indices = list(range(first_frame, last_frame + 1))
    duration = len(frame_indices)
    max_frames = max_frames if max_frames is not None else MAX_FRAMES

    if fps is not None and duration / video_fps < max_frames:
        sampled = frame_sample(duration, mode="fps", vid_fps=video_fps, fps=fps)
    else:
        sampled = frame_sample(duration, mode="uniform", num_frames=max_frames)
    sampled_indices = [frame_indices[index] for index in sampled]

    if os.path.isdir(video_path):
        frames = [
            cv2.cvtColor(
                cv2.imread(os.path.join(video_path, frame_files[index])),
                cv2.COLOR_BGR2RGB,
            )
            for index in sampled_indices
        ]
    elif video_path.endswith(".gif"):
        selected = set(sampled_indices)
        frames = [
            cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            for index, frame in enumerate(gif_reader)
            if index in selected
        ]
    else:
        frames = video_reader.get_batch(sampled_indices).asnumpy()

    timestamps = [index / video_fps for index in sampled_indices]
    if temporal_factor > 1 and len(frames) % temporal_factor:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        step = 1 / (fps or video_fps)
        timestamps.extend(timestamps[-1] + step * (index + 1) for index in range(pad_length))
    return frames, timestamps


def load_video(
    video_path: str,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
    size: Optional[int] = None,
    size_divisible: int = 1,
    precise_time: bool = False,
    verbose: bool = False,
    temporal_factor: int = 1,
):
    if (
        os.path.isdir(video_path)
        or video_path.endswith(".gif")
        or (
            start_time is not None
            and end_time is not None
            and end_time - start_time < 1
        )
    ):
        return load_video_from_ids(
            video_path,
            start_time,
            end_time,
            fps=fps,
            max_frames=max_frames,
            temporal_factor=temporal_factor,
        )

    probe = ffmpeg.probe(video_path)
    duration = float(probe["format"]["duration"])
    video_stream = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )
    width, height = int(video_stream["width"]), int(video_stream["height"])

    trim_args = {}
    should_trim = start_time is not None or end_time is not None
    if start_time is not None:
        adjusted_start = max(float(video_stream["start_time"]), start_time)
        duration -= adjusted_start - start_time
        start_time = adjusted_start
    else:
        start_time = float(video_stream["start_time"])
    if end_time is not None:
        duration = min(duration, end_time - start_time)
    if should_trim:
        trim_args = {"ss": start_time, "t": duration}

    input_args = {} if precise_time else trim_args
    output_args = trim_args if precise_time else {}
    if size is not None:
        scale = size / min(width, height)
        new_width, new_height = round(width * scale), round(height * scale)
    else:
        new_width, new_height = width, height
    new_width = new_width // size_divisible * size_divisible
    new_height = new_height // size_divisible * size_divisible

    stream = ffmpeg.input(video_path, **input_args)
    if fps is not None:
        stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    if (new_width, new_height) != (width, height):
        stream = ffmpeg.filter(stream, "scale", new_width, new_height)
    stream = ffmpeg.output(
        stream,
        "pipe:",
        format="rawvideo",
        pix_fmt="rgb24",
        **output_args,
    )
    output, _ = ffmpeg.run(stream, capture_stdout=True, quiet=not verbose)
    frames = np.frombuffer(output, np.uint8).reshape(
        [-1, new_height, new_width, 3]
    ).transpose([0, 3, 1, 2])

    if fps is not None:
        timestamps = np.arange(
            start_time,
            start_time + duration + 1 / fps,
            1 / fps,
        )[: len(frames)]
    else:
        timestamps = np.linspace(start_time, start_time + duration, len(frames))

    max_frames = max_frames if max_frames is not None else MAX_FRAMES
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[indices]
        timestamps = timestamps[indices]

    if temporal_factor > 1 and len(frames) % temporal_factor:
        pad_length = temporal_factor - len(frames) % temporal_factor
        frames = np.concatenate([frames, frames[-1:].repeat(pad_length, axis=0)])
        step = 1 / (fps or 1)
        timestamps = list(timestamps)
        timestamps.extend(timestamps[-1] + step * (index + 1) for index in range(pad_length))

    return [frame for frame in frames], list(timestamps)
