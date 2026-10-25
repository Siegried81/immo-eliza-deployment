# Security policy

## Supported versions

This is a portfolio and training project with no releases. Only the latest
commit on `main` is maintained; fixes land there and nowhere else.

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Use GitHub's private reporting instead: the **Security** tab of this repository,
then **Report a vulnerability**. If that button is not available, open an issue
titled "Security contact" with no details, and a private channel will be
arranged.

Include what you found, how to reproduce it, and what an attacker gains. You can
expect a first answer within seven days; a confirmed issue is fixed on `main`
and credited in the fix unless you prefer otherwise.

## Secrets

The API and the UI use no key or token. The only credential the project touches
is the Hugging Face write token that `scripts/publish_hf.py` needs; it is the one
`hf auth login` stores in the user's Hugging Face cache, outside the repository,
and the script never reads or prints it. `.env` files stay gitignored;
`.env.example` lists the variables with their defaults and no value. A credential that appears in this
repository's history is a vulnerability in its own right: report it the same
way, and it will be revoked rather than merely deleted.

## What is in scope here

- The prediction API is public and unauthenticated by design; it stores no
  request and no personal data.
- In scope: input that crashes the API or the model, unbounded requests that
  exhaust the free hosting tier, and model files loaded from an untrusted path
  (`joblib` executes code on load).
