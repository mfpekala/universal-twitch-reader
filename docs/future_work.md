# Future Work

This document outlines ways that this system can (and should) be improved/extended in the future.

## Avoid Creating Intermediate Files

Although the system can currently achieve "passable" quality with an RTF comfortably around 1.5, the biggest performance issue is uneccessary intermediate files for programming convenience. To isolate the problem of pre-processing from classification and processing, I had the pre-processing step write detected boxes (and their binarized versions) down to files referenced later in the pipeline. Worse still, we then make another tesseract pass on these subimages, which could've probably just happened as part of the initial pass to begin with.

With more time, I'd be interested to experiment with the quality of the text detected after one pass of an entire screenshot, vs two passes, one to identify boxes and another to extract text. It may be the case that doing this second pass yields higher quality text, in which case the intermediate files and second pass may be worth it.

## More Adaptive Binarization and Merging

The preprocessing step exposes a couple knobs:

- Confidence threshold - How confident does tesseract need to be to call something a box?
- Binarization threshold - What's the cutoff between black/white as we simplify the color space (pre-tesseract).
- Merge theshold - How close/similar do two boxes have to be to merge them?

I set these knobs by playing around with the league compilation video, the single-streamer league stream, and a little bit of the overwatch video. Ideally, there would be better ways to automatically set these parameters for best results instead of having to do trial and error.

I also wonder if different thresholds for different frames could make sense. Games often alternate rapidly between light and dark backgrounds, and a per-frame binarization threshold could avoid losing a box between transition frames.

## Set a Realtime Factor Instead of a Framerate

During pre-processing, we set the rate at which we should extract frames (in terms of frames per second). The number we really care about, however, is real-time factor. Now that the pipeline is complete, it should be possible instead for the user to set a desired RTF and have the system pick the highest framerate possible that would achieve this.

## Incorporate Text in Classification

Once boxes are determined, we extract features by looking at their metadata (position, size) as well as image features (penultimate layer of mobile net, likely coordinated most closely with things like background color or shapes).

However, we also have preliminary text information from the first pass of tesseract. In the end result, I noticed that things flagged as "HUD" (menus, items, lag, kill counts etc.) had very predictable text. By adding in the predicted text to the features we assign to each box during labelling, we'll likely get better performance and more consistent text areas.
