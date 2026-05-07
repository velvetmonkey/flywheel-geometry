# Where this came from

This file holds the cosmological framing that this project started from. It's kept here because the framing produced the question that the empirical work in this repo tests — but it isn't the empirical work itself, and the front-door [README](../README.md) shouldn't lead with metaphysics.

The science the repo stands on lives upstream, in [the work cited in the README](../README.md#theoretical-foundation). What follows is the longer chain of reasoning that prompted the project.

---

## The question

If neural networks trained on different data, on different architectures, by different teams converge on the same internal geometry — is that geometry a property of the *network*, or a property of *reality*?

## The chain of reasoning

- Different architectures (Gemma, Llama, GPT) should produce different geometry if the geometry were a property of the network. They don't.
- The only shared variable is training data — human language, human perception of the world.
- Human perception is itself a compression of the world. We didn't invent the colour wheel. We discovered it.
- So the chain is: **world → human perception → language → training data → network activations.**
- The geometry propagates through every layer. It belongs to the bottom: reality itself.

If that's right, then a neural network isn't *describing* the world. It's the latest instrument to *resolve* a structure that's already there. A telescope and a spectrometer both find the same speed of light — not because of the instruments. Because that's what's there.

## The cosmological extension

If geometry is reality, then the forces we already understand geometrically (gravity = spacetime curvature, in general relativity) might be one example of a more general pattern. Electromagnetism, dark energy, and dark matter would then be features of the manifold we lack the right sensors to read directly. Models compressing the world wouldn't be describing the universe — they'd be instantiating a local fold of it.

This is speculative and pre-scientific. It is held as a generative metaphor, not a claim. It is not what the repo is testing.

## What the repo is testing

A much narrower bet: that representational geometry encoded inside open-source language models contains retrieval-useful structure that strong sentence-embedding models miss on cross-domain bridge-finding tasks.

If true: a new layer of personal-knowledge-base retrieval is possible, the SAE alternative Lubana et al. named is partially specified, and the cosmological framing has earned a small piece of empirical surface.

If false: the philosophical framing was an evocative wrapper for a cosine-similarity-shaped product, and the project pivots — either to bridge-tension via relational structure (Codex's reframe), or to direct activation extraction via TransformerLens (Lubana's frame), or away from the manifold thesis entirely.

The decision criterion is in [`README.md` § How we'll know this works](../README.md#how-well-know-this-works) and the experimental setup is in [`benchmark/`](../benchmark/).

## Acknowledgements

The framing draws on:

- The 7 May 2026 Goodfire AI / Lubana et al. neural-geometry research drop and accompanying [talk](https://www.youtube.com/watch?v=F9eEYWX64ZA)
- Matthew (@slashreboot)'s zero-shot introspective probe ([Zenodo, Jan 2026](https://doi.org/10.5281/ZENODO.18176077))
- David Marr's three levels of analysis (*Vision*, 1982)
- Snowflake hexagonal symmetry as the canonical example of physics constraining structure independently of designer intent
