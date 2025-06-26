# Coding styles

## General guidelines

- Production quality is required for all code from the viewpoint of readability, maintainability, performance and security.
- Be strict on type annotations. Specifically, provide type annotations for both arguments and return types in functions. Follow the modern syntax as recommended in Python 3.10+ such as list, tuple, and dict instead of typing.List, typing.Tuple, and typing.Dict.
- Follow the principle of separation of concerns.
- Minimize dependencies between modules and design loosely coupled modules.
- Use Pydantic data model or dataclass instead of `dict` as a function's argument.
- Match "roles indicated by function, method, and class names" with "actual processing and return values"
- Don't define functions within a function unless you have a special reason.
- Pass arguments in the form of keyword arguments when calling functions and methods.
- Docstring must be written when generating a function or method or class, and docstring should be numpy-style.

## Naming Conventions

- Use descriptive names for variables, functions, methods, and classes.
- Write variable names, function names, and method names in lowercase `snake_case`.
- Write class names in `PascalCase` (do not use underscores to separate words).
- Use uppercase `SNAKE_CASE` for global variable names.
- Prefix variables and method names that are not intended for external access with an underscore.