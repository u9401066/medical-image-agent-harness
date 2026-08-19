# Security and privacy

Do not report patient data in a public issue. If a vulnerability could expose
medical images, identifiers, credentials, or unsafe clinical actions, use GitHub's
private vulnerability reporting for this repository.

The harness accepts only de-identified research inputs. Metadata removal alone does
not establish that pixels, overlays, graphics, or structured content are free of
identifiers. Keep data local unless an authorized policy explicitly allows a remote
model, and retain no source images by default.

Treat image metadata/OCR and external evaluation configurations as untrusted input.
Run third-party adapters and prompt-evaluation code with least privilege, isolated
credentials, and restricted network access.
