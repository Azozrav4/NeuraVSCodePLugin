// Neura language support for Visual Studio Code.
//
// The extension drives the real `neura` compiler front end
// (`neura check FILE --json`) and maps the structured diagnostics it emits onto
// VS Code diagnostics, so parse / type / ownership errors appear as squiggles,
// in the Problems panel and in the editor gutter.
//
// It also exposes commands to run, build and format the active program by
// invoking the corresponding `neura` subcommands.

import * as vscode from "vscode";
import { spawn } from "child_process";
import type { ChildProcess } from "child_process";
import * as crypto from "crypto";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

const LANGUAGE_ID = "neura";
const DIAGNOSTIC_SOURCE = "neura";
const OUTPUT_NAME = "Neura";

/** A single diagnostic as emitted by `neura check --json` (see compiler/src/diagnostics.rs). */
interface NeuraDiagnostic {
  severity: "error" | "warning" | "note";
  code: string;
  message: string;
  span?: { line: number; column: number; start: number; end: number };
  notes?: string[];
  help?: string;
}

interface NeuraCheckResult {
  status: "ok" | "error";
  file: string;
  diagnostics: NeuraDiagnostic[];
}

interface RunResult {
  code: number | null;
  stdout: string;
  stderr: string;
}

/** Per-document bookkeeping so checks can be debounced, deduplicated and cancelled. */
interface DocumentState {
  timer?: NodeJS.Timeout;
  child?: ChildProcess;
  generation: number;
  lastHash?: string;
}

export function activate(context: vscode.ExtensionContext): void {
  const collection = vscode.languages.createDiagnosticCollection(DIAGNOSTIC_SOURCE);
  const output = vscode.window.createOutputChannel(OUTPUT_NAME);
  const states = new Map<string, DocumentState>();
  let missingCompilerReported = false;

  context.subscriptions.push(collection, output);

  const report = (line: string): void => {
    output.appendLine(line);
  };

  const stateFor = (key: string): DocumentState => {
    let state = states.get(key);
    if (!state) {
      state = { generation: 0 };
      states.set(key, state);
    }
    return state;
  };

  const warnMissingCompiler = (): void => {
    if (missingCompilerReported) {
      return;
    }
    missingCompilerReported = true;
    void vscode.window
      .showWarningMessage(
        `Neura: could not run the compiler "${compilerConfig()}". ` +
          "Build it with `cargo build -p neura-cli` or set `neura.compilerPath`.",
        "Open Settings"
      )
      .then((choice) => {
        if (choice === "Open Settings") {
          void vscode.commands.executeCommand(
            "workbench.action.openSettings",
            "neura.compilerPath"
          );
        }
      });
  };

  /** Check a document, superseding any in-flight check for it. */
  const check = (document: vscode.TextDocument, force = false): Promise<void> =>
    checkDocument(
      document,
      stateFor(document.uri.toString()),
      collection,
      report,
      warnMissingCompiler,
      force
    );

  const scheduleCheck = (document: vscode.TextDocument, delay = 300): void => {
    if (document.languageId !== LANGUAGE_ID) {
      return;
    }
    if (!getConfig().get<boolean>("checkOnType", true) && delay > 0) {
      return;
    }
    const state = stateFor(document.uri.toString());
    if (state.timer) {
      clearTimeout(state.timer);
    }
    state.timer = setTimeout(() => {
      state.timer = undefined;
      void checkDocument(document, state, collection, report, warnMissingCompiler, false);
    }, delay);
  };

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((document) => scheduleCheck(document, 0)),
    vscode.workspace.onDidSaveTextDocument((document) => {
      if (getConfig().get<boolean>("checkOnSave", true)) {
        scheduleCheck(document, 0);
      }
    }),
    vscode.workspace.onDidChangeTextDocument((event) => scheduleCheck(event.document)),
    vscode.workspace.onDidCloseTextDocument((document) => {
      const key = document.uri.toString();
      const state = states.get(key);
      if (state) {
        if (state.timer) {
          clearTimeout(state.timer);
        }
        state.child?.kill();
        states.delete(key);
      }
      collection.delete(document.uri);
    }),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("neura")) {
        // Settings may change the compiler path or flags, so drop caches.
        invalidateCompilerCache();
        for (const state of states.values()) {
          state.lastHash = undefined;
        }
        for (const document of vscode.workspace.textDocuments) {
          scheduleCheck(document, 0);
        }
      }
    })
  );

  const activeNeuraDocument = (): vscode.TextDocument | undefined => {
    const document = vscode.window.activeTextEditor?.document;
    if (!document || document.languageId !== LANGUAGE_ID) {
      void vscode.window.showWarningMessage("Neura: open a .neura file first.");
      return undefined;
    }
    return document;
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("neura.check", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      await check(document, true);
      const count = collection.get(document.uri)?.length ?? 0;
      if (count === 0) {
        void vscode.window.showInformationMessage("Neura: no problems found.");
      } else {
        void vscode.window.showWarningMessage(`Neura: ${count} problem(s) found.`);
        output.show(true);
      }
    }),

    vscode.commands.registerCommand("neura.run", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      const file = await ensureSaved(document);
      if (!file) {
        return;
      }
      // Refresh diagnostics first so compile failures are visible as squiggles.
      await check(document, true);
      const args = ["run", file, ...extraArgs()];
      report(`> neura ${args.join(" ")}`);
      const result = await runCompiler(args, cwdFor(document.uri), report, warnMissingCompiler);
      if (!result) {
        return;
      }
      if (result.stdout) {
        output.append(result.stdout.endsWith("\n") ? result.stdout : result.stdout + "\n");
      }
      if (result.stderr) {
        output.append(result.stderr.endsWith("\n") ? result.stderr : result.stderr + "\n");
      }
      report(`[run exited with code ${result.code ?? "?"}]`);
      output.show(true);
      if (result.code !== 0) {
        void vscode.window.showErrorMessage("Neura: run failed — see the Neura output channel.");
      }
    }),

    vscode.commands.registerCommand("neura.build", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      const file = await ensureSaved(document);
      if (!file) {
        return;
      }
      const args = ["build", file, ...extraArgs()];
      report(`> neura ${args.join(" ")}`);
      const result = await runCompiler(args, cwdFor(document.uri), report, warnMissingCompiler);
      if (!result) {
        return;
      }
      if (result.stdout) {
        output.append(result.stdout.endsWith("\n") ? result.stdout : result.stdout + "\n");
      }
      if (result.stderr) {
        output.append(result.stderr.endsWith("\n") ? result.stderr : result.stderr + "\n");
      }
      output.show(true);
      if (result.code !== 0) {
        void vscode.window.showErrorMessage("Neura: build failed — see the Neura output channel.");
      } else {
        void vscode.window.showInformationMessage(`Neura: built ${path.basename(file, ".neura")}.neurabc`);
      }
    }),

    vscode.commands.registerCommand("neura.format", async () => {
      const document = activeNeuraDocument();
      if (!document) {
        return;
      }
      const edits = await computeFormatEdits(document, report, warnMissingCompiler);
      if (!edits) {
        output.show(true);
        void vscode.window.showErrorMessage("Neura: format failed — see the Neura output channel.");
        return;
      }
      if (edits.length === 0) {
        void vscode.window.showInformationMessage("Neura: file is already formatted.");
        return;
      }
      const edit = new vscode.WorkspaceEdit();
      edit.set(document.uri, edits);
      if (await vscode.workspace.applyEdit(edit)) {
        void vscode.window.showInformationMessage("Neura: file formatted.");
      } else {
        void vscode.window.showErrorMessage("Neura: could not apply formatting.");
      }
    }),

    vscode.commands.registerCommand("neura.showOutput", () => output.show(true)),

    // "Format Document" runs the canonical `neura fmt` formatter.
    vscode.languages.registerDocumentFormattingEditProvider(LANGUAGE_ID, {
      provideDocumentFormattingEdits: (document) =>
        computeFormatEdits(document, report, warnMissingCompiler)
    })
  );

  // Check documents that are already open when the extension activates.
  for (const document of vscode.workspace.textDocuments) {
    scheduleCheck(document, 0);
  }

  report("Neura extension active.");
}

export function deactivate(): void {
  // Diagnostics and timers are owned by the extension context and disposed with it.
}

// ---------------------------------------------------------------------------
// Checking
// ---------------------------------------------------------------------------

async function checkDocument(
  document: vscode.TextDocument,
  state: DocumentState,
  collection: vscode.DiagnosticCollection,
  report: (line: string) => void,
  onMissingCompiler: () => void,
  force: boolean
): Promise<void> {
  if (document.languageId !== LANGUAGE_ID) {
    return;
  }

  const text = document.getText();
  const hash = crypto.createHash("md5").update(text).digest("hex");
  if (!force && state.lastHash === hash) {
    // Nothing changed since the last successful check; skip the compiler run.
    return;
  }

  // Supersede any check that is still running for this document.
  const generation = ++state.generation;
  if (state.child) {
    state.child.kill();
    state.child = undefined;
  }

  // `neura check` reads from disk, so unsaved buffers are written to a temp copy.
  let target = document.uri.fsPath;
  let temporary: string | undefined;
  try {
    if (document.isDirty || document.isUntitled) {
      temporary = writeTempSource(document);
      target = temporary;
    }

    const args = ["check", target, "--json", ...extraArgs()];
    report(`> neura ${args.join(" ")}`);
    let spawned: ChildProcess | undefined;
    const result = await runCompiler(
      args,
      cwdFor(document.uri),
      report,
      onMissingCompiler,
      (child) => {
        spawned = child;
        state.child = child;
      }
    );
    if (state.child === spawned) {
      state.child = undefined;
    }
    if (generation !== state.generation || !result) {
      // Superseded by a newer check (or the compiler could not start): keep the
      // previous diagnostics instead of applying a stale result.
      return;
    }

    const parsed = parseJsonObject<NeuraCheckResult>(result.stdout);
    if (!parsed || !Array.isArray(parsed.diagnostics)) {
      collection.delete(document.uri);
      const detail = result.stderr.trim();
      report(
        `[check] no JSON diagnostics on stdout (exit ${result.code ?? "?"}).` +
          (detail ? ` stderr: ${detail}` : "")
      );
      return;
    }

    state.lastHash = hash;
    collection.set(document.uri, parsed.diagnostics.map((d) => toDiagnostic(d, document)));
  } catch (error) {
    if (generation === state.generation) {
      collection.delete(document.uri);
      report(`[check] ${String(error)}`);
    }
  } finally {
    if (temporary) {
      try {
        fs.unlinkSync(temporary);
      } catch {
        // Best effort cleanup only.
      }
    }
  }
}

function diagnosticRange(
  document: vscode.TextDocument,
  span: NeuraDiagnostic["span"]
): vscode.Range {
  const lastLine = Math.max(0, document.lineCount - 1);
  const line = clamp((span?.line ?? 1) - 1, 0, lastLine);
  const lineText = document.lineCount > 0 ? document.lineAt(line).text : "";
  const startChar = clamp((span?.column ?? 1) - 1, 0, lineText.length);
  const width = span && span.end > span.start ? span.end - span.start : 0;
  const endChar = Math.min(startChar + width, lineText.length);
  if (endChar > startChar) {
    return new vscode.Range(line, startChar, line, endChar);
  }
  // The compiler uses zero-width spans for some errors (for example an
  // unexpected end of file). Anchor the squiggle on the nearest preceding
  // non-space character so the problem stays visible.
  return anchorRange(document, line, startChar);
}

function anchorRange(document: vscode.TextDocument, line: number, char: number): vscode.Range {
  let current = line;
  let limit = char;
  while (current >= 0) {
    const text = document.lineAt(current).text;
    let index = Math.min(limit, text.length) - 1;
    while (index >= 0 && /\s/.test(text[index])) {
      index -= 1;
    }
    if (index >= 0) {
      return new vscode.Range(current, index, current, index + 1);
    }
    current -= 1;
    limit = Number.MAX_SAFE_INTEGER;
  }
  return new vscode.Range(0, 0, 0, 1);
}

function toDiagnostic(diagnostic: NeuraDiagnostic, document: vscode.TextDocument): vscode.Diagnostic {
  const severity =
    diagnostic.severity === "error"
      ? vscode.DiagnosticSeverity.Error
      : diagnostic.severity === "warning"
        ? vscode.DiagnosticSeverity.Warning
        : vscode.DiagnosticSeverity.Information;

  const range = diagnosticRange(document, diagnostic.span);

  let message = diagnostic.message;
  if (diagnostic.help) {
    message += `\nhelp: ${diagnostic.help}`;
  }
  if (diagnostic.notes && diagnostic.notes.length > 0) {
    message += "\n" + diagnostic.notes.map((note) => `note: ${note}`).join("\n");
  }

  const result = new vscode.Diagnostic(range, message, severity);
  result.code = diagnostic.code;
  result.source = DIAGNOSTIC_SOURCE;
  return result;
}

// ---------------------------------------------------------------------------
// Compiler process plumbing
// ---------------------------------------------------------------------------

function runCompiler(
  args: string[],
  cwd: string | undefined,
  report: (line: string) => void,
  onMissingCompiler: () => void,
  onSpawn?: (child: ChildProcess) => void
): Promise<RunResult | undefined> {
  const binary = resolveCompilerPath();
  return new Promise<RunResult | undefined>((resolve) => {
    let child;
    try {
      child = spawn(binary, args, { cwd, windowsHide: true });
    } catch (error) {
      report(`[compiler] failed to launch "${binary}": ${String(error)}`);
      onMissingCompiler();
      resolve(undefined);
      return;
    }

    onSpawn?.(child);

    let stdout = "";
    let stderr = "";
    let settled = false;
    const finish = (value: RunResult | undefined): void => {
      if (!settled) {
        settled = true;
        resolve(value);
      }
    };

    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });
    child.on("error", (error: NodeJS.ErrnoException) => {
      report(`[compiler] "${binary}" could not be executed: ${error.message}`);
      if (error.code === "ENOENT") {
        onMissingCompiler();
      }
      finish(undefined);
    });
    child.on("close", (code) => finish({ code, stdout, stderr }));
  });
}

/**
 * Resolving the compiler probes the filesystem, so cache the result until the
 * configuration changes (see `invalidateCompilerCache`).
 */
let compilerCache: { configured: string; resolved: string } | undefined;

function invalidateCompilerCache(): void {
  compilerCache = undefined;
}

function resolveCompilerPath(): string {
  const configured = compilerConfig();
  if (compilerCache?.configured === configured) {
    return compilerCache.resolved;
  }
  const resolved = probeCompilerPath(configured);
  compilerCache = { configured, resolved };
  return resolved;
}

function probeCompilerPath(configured: string): string {
  const candidates: string[] = [];
  const looksLikePath =
    path.isAbsolute(configured) || configured.includes("/") || configured.includes("\\");
  if (looksLikePath) {
    candidates.push(configured);
  }
  for (const folder of vscode.workspace.workspaceFolders ?? []) {
    candidates.push(path.join(folder.uri.fsPath, "target", "release", executableName()));
    candidates.push(path.join(folder.uri.fsPath, "target", "debug", executableName()));
  }
  candidates.push(configured);
  for (const candidate of candidates) {
    try {
      if (fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
        return candidate;
      }
    } catch {
      // Ignore probe failures and continue with the next candidate.
    }
  }
  // Fall back to the bare command; the OS resolves it on PATH at spawn time.
  return configured;
}

function executableName(): string {
  return process.platform === "win32" ? "neura.exe" : "neura";
}

function compilerConfig(): string {
  const value = getConfig().get<string>("compilerPath", "neura").trim();
  return value.length > 0 ? value : "neura";
}

function extraArgs(): string[] {
  const args = getConfig().get<string[]>("compilerArgs", []);
  return Array.isArray(args) ? args.filter((a) => typeof a === "string") : [];
}

function getConfig(): vscode.WorkspaceConfiguration {
  return vscode.workspace.getConfiguration("neura");
}

function cwdFor(uri: vscode.Uri): string | undefined {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (folder) {
    return folder.uri.fsPath;
  }
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? path.dirname(uri.fsPath);
}

function writeTempSource(document: vscode.TextDocument): string {
  const directory = path.join(os.tmpdir(), "neura-vscode");
  fs.mkdirSync(directory, { recursive: true });
  const digest = crypto.createHash("md5").update(document.uri.toString()).digest("hex").slice(0, 12);
  const base = document.isUntitled
    ? "untitled"
    : path.basename(document.uri.fsPath, path.extname(document.uri.fsPath));
  const file = path.join(directory, `${base || "untitled"}-${digest}.neura`);
  fs.writeFileSync(file, document.getText(), "utf8");
  return file;
}

async function ensureSaved(document: vscode.TextDocument): Promise<string | undefined> {
  if (document.isUntitled) {
    const saved = await document.save();
    if (!saved) {
      void vscode.window.showWarningMessage("Neura: save the file before running this command.");
      return undefined;
    }
  } else if (document.isDirty) {
    await document.save();
  }
  return document.uri.fsPath;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Format the document with `neura fmt`. The current buffer is written to a
 * temporary file (so unsaved edits are formatted too) and the formatted text is
 * returned as a whole-document replacement, leaving the file on disk untouched.
 */
async function computeFormatEdits(
  document: vscode.TextDocument,
  report: (line: string) => void,
  onMissingCompiler: () => void
): Promise<vscode.TextEdit[] | undefined> {
  const original = document.getText();
  const temporary = writeTempSource(document);
  try {
    const args = ["fmt", temporary];
    report(`> neura ${args.join(" ")}`);
    const result = await runCompiler(args, cwdFor(document.uri), report, onMissingCompiler);
    if (!result) {
      return undefined;
    }
    if (result.stdout) {
      report(result.stdout.trimEnd());
    }
    if (result.code !== 0) {
      if (result.stderr) {
        report(result.stderr.trimEnd());
      }
      return undefined;
    }
    const formatted = fs.readFileSync(temporary, "utf8");
    if (formatted === original) {
      return [];
    }
    const fullRange = new vscode.Range(
      document.positionAt(0),
      document.positionAt(original.length)
    );
    return [vscode.TextEdit.replace(fullRange, formatted)];
  } catch (error) {
    report(`[format] ${String(error)}`);
    return undefined;
  } finally {
    try {
      fs.unlinkSync(temporary);
    } catch {
      // Best effort cleanup only.
    }
  }
}

/** Extract the last JSON object found on its own line of `text`. */
function parseJsonObject<T>(text: string): T | undefined {
  const lines = text.split(/\r?\n/);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i].trim();
    if (!line.startsWith("{")) {
      continue;
    }
    try {
      return JSON.parse(line) as T;
    } catch {
      // Not valid JSON on this line; keep scanning upwards.
    }
  }
  return undefined;
}
