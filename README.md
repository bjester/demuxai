# demuxai

**EXPERIMENTAL: this is a work-in-progress, and for local usage. Not for production!**

A lightweight proxy that provides a singular OpenAI compatible API for all your AI services.
- Aggregates upstream APIs' models and prefixes them with a provider ID
- The provider ID is used to route the request to the upstream provider
- Before sending the request upstream, the provider ID is stripped from the request

Named after a demultiplexer, takes one input and distributes it to one or more outputs.

## Background
I wanted an API proxy that I could configure in my Jetbrains IDE, but not limit me to that provider for all settings. For example, I use a local Ollama model for instant completion, and other providers for chat or agentic tasks. Additionally, there are many free tiers available, so this abstraction layer can implement more advanced routing behaviors to take full advantage of free offerings, without interruption.

## Features
Supports the following providers:
- Ollama
- Ollama Cloud
- Github Models
- Mistral AI (and Codestral)
- Fireworks AI Serverless

Currently, demuxai has the following capabilities:
- Legacy completion endpoint
- Chat completion
- FIM completion
- Embedding generation
- Streaming responses

### Planned
The following features are planned:
- Timing statistics (WIP)
- Token statistics (WIP)
- Complete most of the API spec
- Completion ID routing back to the original provider
- Round-robin and failover providers
- Consensus providers to summarize responses from multiple upstream providers
- Split providers that route to upstream providers based off content type (i.e. python vs JS)
- Configuration like [routedns](https://github.com/folbricht/routedns)

## Alternatives
- Need a full-featured proxy or GUI? Checkout [bifrost](https://github.com/maximhq/bifrost)
- Need more providers? Checkout [LiteLLM](https://github.com/BerriAI/litellm)

## AI Disclosure
LLMs were used in the development of this project, mostly for brainstorming and bootstrapping code, particularly tests. The contribution proportion is roughly 80 / 20, human and AI code respectively.

## License
[MIT](LICENSE) :: Copyright 2026 Blaine Jester
