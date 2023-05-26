# Universal Twitch Reader

Universal Twitch Reader (UTR) is an engine which takes in a recording of a twitch stream and produces a log of time-stamped text from the stream separated into user-informed categories.

It works in a few phases.

- First, tesseract is used to find regions with high-probability of containing text
- Then, a merging algorithm is applied to group very adjacent boxes together
- Next, MobileNet is used to extract features and the boxes are clustered into categories based off targeted input from the user. Meta-data about the boxes such as position on screen and size is also used to improve semantic clustering.
- Finally, tesseract is applied on the individual boxes and changes are logged between frames and timestamped in the final output file.

The input is a list of files to analyze.

The output is a series of files that look like:

```
// A.json
{
  texts: [
    ["00:00:03", "Hellu everyone!"],
    ["00:00:04", "*Hello "],
    ["00:00:11", "glhf"],
    ...
  ]
}
```

where "A" is the name of a type of on-screen text as inputted by the user.

## Documentation

Documentation is split into files in the `docs` folder for readability.

### [Setup](docs/setup.md)

### [System Overview](docs/system_overview.md)

### [Results](docs/results.md)

### [Pitfalls / Future Work](docs/future_work.md)
