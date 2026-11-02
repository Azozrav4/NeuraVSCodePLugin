<p align="center">
  <img src="assets/neura-logo-vscode.png" alt="Neura" width="420">
</p>

# Neura for VS Code

Language support for the [Neura](https://github.com/crimpsecondary27/neura)
numerical/machine-learning programming language. This repository contains the
VS Code extension only.

The extension adds `.neura` syntax highlighting and runs the real `neura`
compiler for live error checking, build and run. It drives the compiler binary
(`neura check --json`) as a subprocess; it does not link the Rust crates.

## Features

- Syntax highlighting for `.neura` files (`source.neura` TextMate grammar).
- Live diagnostics: errors and warnings appear inline as you type or save.
- Commands (Command Palette or editor title bar for a `.neura` file):
  - **Neura: Check File** — parse, type-check and ownership-check.
  - **Neura: Run File** — compile, optimise and execute `main`.
  - **Neura: Build Artifact** — write a hash-verified `.neurabc` artifact.
  - **Neura: Format File** — canonical formatting.
  - **Neura: Show Compiler Output** — the raw compiler log.

## Requirements

- VS Code `1.75` or newer.
- The `neura` compiler binary, built from the main
  [Neura repository](https://github.com/crimpsecondary27/neura):
  ```sh
  git clone https://github.com/crimpsecondary27/neura
  cd neura
  cargo build --release        # -> target/release/neura
  ```
  Put `neura` on your `PATH`, or point the extension at it with the
  `neura.compilerPath` setting.
- Node.js and npm, to build and package the extension.

## Build the extension

```sh
npm install
npm run compile      # tsc -p ./  ->  out/extension.js
```

## Install into VS Code

**Option A — try it in a development host (no install).** Open this folder in
VS Code and press `F5`. This compiles the extension and launches an *Extension
Development Host* window with the extension loaded; the bundled
[`.vscode/launch.json`](.vscode/launch.json) and
[`.vscode/tasks.json`](.vscode/tasks.json) wire up the pre-launch compile.

**Option B — install it permanently as a `.vsix`.**

```sh
npx @vscode/vsce package          # produces neura-0.1.0.vsix
code --install-extension neura-0.1.0.vsix
```

Or use the VS Code UI: `Extensions` view → `...` menu → *Install from VSIX…*.

**Option C — link the folder directly.** Copy or symlink this repository into
your extensions directory (`~/.vscode/extensions/` on Linux/macOS,
`%USERPROFILE%\.vscode\extensions\` on Windows) and restart VS Code. Re-run
`npm run compile` after changing the source.

## Settings

By default the extension looks for `neura` on your `PATH`, then for
`target/release/neura` or `target/debug/neura` in the workspace. To override:

| Setting | Default | Purpose |
| --- | --- | --- |
| `neura.compilerPath` | `neura` | Path to the compiler binary (bare command, absolute, or workspace-relative). |
| `neura.checkOnType` | `true` | Re-check for errors while typing (debounced). |
| `neura.checkOnSave` | `true` | Check for errors on every save. |
| `neura.compilerArgs` | `[]` | Extra flags for every invocation, e.g. `--opt=aggressive`. |

## Repository layout

```
package.json                 extension manifest (language, grammar, commands, settings)
src/extension.ts             activation, diagnostics, commands, compiler invocation
syntaxes/neura.tmLanguage.json   TextMate grammar
language-configuration.json  brackets, comments, indentation rules
assets/                      extension icon
.vscode/                     launch/tasks config for F5 debugging
```

## License

MIT
