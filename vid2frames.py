import ffmpeg
from tqdm import tqdm


def convert2frames(
    video_path, output_path, fps=1 / 2
) -> float:  # Default to one frame every two seconds
    probe = ffmpeg.probe(video_path)
    time = float(probe["streams"][0]["duration"])
    num_frames = int(time * fps)
    interval = 1 / fps

    offsets = [i * interval for i in range(num_frames)]

    for ix, offset in tqdm(enumerate(offsets), total=len(offsets)):
        (
            ffmpeg.input(video_path, ss=offset)
            .output(f"{output_path}/f_{int(ix * interval * 1000)}.jpg", vframes=1)
            .run(quiet=True)
        )
    return time
