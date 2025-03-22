# monoculus

Neural network research inspired by neuroscience and modularity, including but
not limited to self organization, ring topology, network bridging
(locality/topology aware), bridge dynamics, hot swapping bridges, co-evaluation
or competition as a form of internal feedback, runaway feedback triggering a
return to homeostasis as a form of response-to-prompt trigger, and whatever else
I find interesting.

Eventually I'd like to be able to write an agent capable of:
    - Continuing to process in the absence of input, E.g. generating its own
      stimuli via feedback from hidden state or memory, and/or modifying its
      internal representations continuously through noise.
    - Hot-swapping bridged networks, potentially with an also-hot-swappable
      variety of bridged network management modules (self-organizing or
      deterministic, idk).

I'd like to set everything up from scratch (E.g., don't use pytorch) and
preferably do so in Rust. I want everything to be modular so it's easy to swap
out one tokenizer for another, or one context window management paradigm for
another, etc. But like, everything should be modular.

## Topics To Explore

- Testing techniques
    - Targeted prompts to test things like memory, coherence, etc.

- Optimization techniques
    - exlore ways to benchmark training speed and operation speed
    - training faster
    - operating faster
    - optimize for memory or execution time
    - run on different types of hardware

- Different algorithms for dropping context
    - drop oldest context
    - drop least relevant context
    - compress / summarize context
    - non-linear attention weighting
    - recurrence or segment-level recurrence

- Weight editing
    - Rank-One Model Editing
    - Batch knowledge insertion
    - Swapping a network the main agent is bridged to with another

- Layered context windows
    - Hierarchical prompting
    - Combine short term and long term windows

- Topologies
    - Self Organizing Maps (SOMs)
    - Convolution Neural Networks (CNNs)
    - Graph Neural Networks (GNNs)
    - Recurrent Neural Networks (RNNs)

- Noise
    - Continual training
    - Dropout
    - Noisy gradients
    - inject noise on a timer as a way to simulate the passage of time
    - inject noise during internal feedback loops
    - inject noise during training
    - inject different types of noise
    - reduce noise periodically in order to simulate a return to homeostasis
      E.g., trend state towards identity matrix or base state.
    - somehow simulate neuron fatigue.
    - Some theories of consciousness (e.g., Tononi’s IIT, Friston’s FEP) argue
      that awareness arises when internal systems deviate from equilibrium, and
      must reconcile conflicting inputs, goals, or predictions. “What if we
      built a neural system that wakes up, not when given input, but when
      knocked out of balance?”

- Feedback
    - internal feedback loops
    - feedback loops that runaway until some threshold, then attempt to return
      to homeostasis or local minimum.
    - multiple interconnected networks that can influence neighbors on the other
      side of links.
    - Different algorithms for neighbor influence

- Input mediums
    - Text
    - Image
    - Video
    - Audio
    - Combination
    - Arbitrary files

- Output mediums
    - Text
    - Image
    - Video
    - Audio
    - Combination
    - Arbitrary files
    - Video game controls
