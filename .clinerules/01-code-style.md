# Code Style Rules
- Prefer multiline strings over string building.
- Never use many separate print statements for formatted output; use a single formatted string instead.
- Do not make imports inside of functions and methods, unless there's a very good reason to (e.g. torch being an optional import due to cuda requirements is a good reason, and probably the only one)
- Do not make arbitrary comments.