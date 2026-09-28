"""One-time, exact-fingerprint migration of compatible pre-alignment retrieval."""

import ast
import hashlib
import json
import subprocess


def compatible_retrieval(before: bytes, after: bytes) -> bool:
    names = {'_read_cache', '_append_cache', '_cache_key', '_retrieve_all'}

    def signature(source):
        tree = ast.parse(source)
        nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign))
                 or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names]
        present = {node.name for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if present != names:
            raise ValueError('missing retrieval function')
        return [ast.dump(node, include_attributes=False) for node in nodes]

    try:
        return signature(before) == signature(after)
    except (SyntaxError, ValueError):
        return False


def migrate_legacy(root, target, scope) -> bool:
    if target.exists():
        return False
    try:
        previous = subprocess.run(
            ['git', 'show', 'abe481e:scripts/eval_ch04.py'], cwd=root,
            capture_output=True, check=True, timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    current = (root / 'scripts/eval_ch04.py').read_bytes()
    if not compatible_retrieval(previous, current):
        return False
    # Git stores LF; Windows checkout may have used CRLF. Only these exact
    # historical encodings are candidates, with CURRENT authority/config/code.
    lf = previous.replace(b'\r\n', b'\n')
    for original in (lf, lf.replace(b'\n', b'\r\n')):
        legacy_scope = [*scope]
        legacy_scope[2] = [original.hex(), *scope[2][1:]]
        digest = hashlib.sha256(json.dumps(legacy_scope, ensure_ascii=False, default=str).encode()).hexdigest()[:20]
        legacy = target.with_name(f'retrieval-{digest}.jsonl')
        if legacy.is_file() and legacy != target:
            content = legacy.read_bytes()
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open('xb') as output:
                    output.write(content)
            except FileExistsError:
                return False
            return True
    return False
