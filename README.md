
NES-Solve
==

NES-Solve is a GPU-based Nintendo Entertainment System emulator written in CUDA, with a Python API. Its intended use case is data generation for reinforcement learning. It currently does not have a general-purpose API, since it is hard-coded to my experiments around certain convolutions.

`NESSolve.run` takes as input a filename to a game ROM and a *num_instances* x *num_actions* matrix consisting of the sequence of button presses each instance should make per frame, and outputs an array of size *num_instances* x *num_actions* x 256/*kernel_width* x 240/*kernel_height* containing the kernel activations seen in each instance.

It currently supports Mapper 0 games. It will have Mapper 1 support but I'm only really interested in a couple of games so I won't be adding mappers beyond that. I plan on adding a random forest engine in order to "inline" agents.

It's not yet clear whether this GPU-based approach is a win vs spending similar effort on a SIMD-style/cache-optimized CPU-based NES emulator, but it is better given my use case and resources, since GPUs are easier to come by than high core count CPUs.

The key optimization challenge is that the NES deals with random access on memory on the scale of kilobytes to tens of kilobytes, which is too large to use any kind of straightforward memory locality. But since this is the first CUDA program I've written there is no doubt a lot of low-hanging fruit.

Frame Data
---

NES-Solve can generate frames. Ideally we would want to generate PyTorch tensors on-device, but I'm not doing standard vision-based learning at the moment so such a feature isn't implemented.

Block Score
--

Some of my experiments revolve around what I call the Block Score. See [here](http://ec2-54-176-62-21.us-west-1.compute.amazonaws.com/) for a live demonstration. The block score is the number of unique rectangles seen by a game instance. In the following screen, the sky blocks all together count as 1 since they're all the same, while the cloud counts as 4 since each of its 4 tiles are unique.

<p align="center">
    <img src="notebooks/grid.png">
</p>

The intuition behind the block score is as follows. First, it has less variance than the human-chosen game score, which is spikier and more prone to pathological scenarios. Second, humans are generally not good at designing score systems<i>no sense of irony</i>, so a more human-agnostic measure seems better to me.

Third, games are visual and designed for humans to enjoy. The general characteristics of games as entertainment implies that greater progress through a game involves greater visual entropy, so to speak. By contrast, a game's score system is often just a way of adding texture and otherwise not a significant element.

I may be totally wrong on all of these points, but I'll worry about that later. Note that the block score actually does account for the human-given score to some degree. Particular kernel sizes or offsets will allow the digital score shown on the screen to be tracked logarithmically, which is about right. And for example, in Super Mario Bros small +100 tokens appear near events, which will count if they are unique.
