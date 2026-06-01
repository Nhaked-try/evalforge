# Security

## Reporting vulnerabilities

If you find a security issue, email me at nhaked.try@protonmail.com. Don't open a public issue.

## What's relevant

EvalForge runs LLM inference locally and connects to external APIs (OpenAI, Anthropic) for judge models. Potential concerns:

- **API key exposure**: Keys are loaded from env vars or `.env` files. Never commit `.env` files.
- **Prompt injection**: If you're evaluating untrusted inputs, be aware that adversarial prompts could manipulate judge models. This is a known limitation, not a bug.
- **Local model execution**: If using a local judge model, it runs with full system access. Don't run untrusted model weights.

## Supported versions

Security fixes only apply to the latest release.
