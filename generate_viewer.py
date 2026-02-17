import re
import json

def generate_html(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
        return

    chunks = re.split(r'// -----// (IR Dump After .*?) //----- //', content)
    chunks = [chunk.replace("IR Dump After ", "").strip() for chunk in chunks]

    
    passes = []
    for i in range(1, len(chunks), 2):
        pass_title = chunks[i].strip()
        ir_content = chunks[i+1] if i+1 < len(chunks) else ""
        passes.append({"title": pass_title, "ir": ir_content})

    html_template = r"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>MLIR Pass Viewer</title>
        <style>
            body { font-family: 'Segoe UI', system-ui, sans-serif; display: flex; height: 100vh; margin: 0; background: #f4f4f4; }
            #sidebar { width: 350px; background: #2c3e50; color: white; overflow-y: auto; padding: 15px; flex-shrink: 0; }
            #content { flex-grow: 1; overflow-y: auto; padding: 20px; background: white; scroll-behavior: smooth; }
            .pass-item { padding: 10px; cursor: pointer; border-bottom: 1px solid #34495e; font-size: 12px; font-family: monospace; }
            .pass-item:hover { background: #34495e; }
            .pass-item.active { background: #3498db; font-weight: bold; }
            
            details { border: 1px solid #ddd; border-radius: 4px; margin-bottom: 15px; background: #fff; scroll-margin-top: 70px; }
            summary { padding: 10px; background: #f8f9fa; cursor: pointer; font-family: monospace; color: #e67e22; font-weight: bold; display: flex; align-items: center; border-bottom: 1px solid #eee; }
            summary:hover { background: #eee; }
            
            pre { 
                margin: 0; 
                padding: 10px 0; 
                overflow-x: auto; 
                white-space: pre; 
                font-family: 'Consolas', 'Monaco', monospace; 
                font-size: 13px; 
                line-height: 1.4; 
                background: #fff;
            }
            
            .line { display: block; padding: 0 15px; min-height: 1.4em; }
            .line.highlighted { 
                background-color: #e1ffdc; 
                border-left: 4px solid #2ecc71; 
                padding-left: 11px; 
            }
            
            .controls { position: sticky; top: -20px; z-index: 10; background: #f4f4f4; margin-bottom: 15px; padding: 10px; border-radius: 5px; display: flex; align-items: center; gap: 10px; border: 1px solid #ddd; }
            button { cursor: pointer; padding: 6px 12px; background: #3498db; color: white; border: none; border-radius: 3px; font-size: 12px; }
            .pin-btn { margin-right: 10px; background: white; border: 1px solid #ccc; color: #666; padding: 2px 6px; font-size: 10px; border-radius: 3px; font-weight: normal; }
            .pin-btn.active { background: #f1c40f; color: #000; border-color: #f39c12; font-weight: bold; }
            .pinned-label { font-size: 12px; color: #f39c12; font-weight: bold; }
            .char-added {
                background-color: #b6f2b6;
                font-weight: bold;
            }
            .char-removed {
                background-color: #f7b6b6;
                text-decoration: line-through;
            }
            .line.deleted {
                background-color: #ffecec;
                border-left: 4px solid #e74c3c;
                padding-left: 11px;
                color: #999;
            }

            .line.added {
                background-color: #e1ffdc;
                border-left: 4px solid #2ecc71;
                padding-left: 11px;
            }
        </style>
    </head>
    <body>
        <div id="sidebar">
            <h3>Compiler Passes</h3>
            SIDEBAR_PLACEHOLDER
        </div>
        <div id="content">
            <div class="controls">
                <button onclick="toggleAll(true)">Expand All</button>
                <button onclick="toggleAll(false)">Collapse All</button>
                <button onclick="toggleDiffMode()" id="diff-toggle"></button>
                <span id="pin-status" class="pinned-label"></span>
            </div>
            <div id="display-area">Select a pass from the sidebar.</div>
        </div>

        <script src="https://unpkg.com/diff@5.2.0/dist/diff.min.js"></script>
        <script>
            const DIFF_MODES = ["Simple", "Library", "Char", "Op-Aware"];

            let diffMode = localStorage.getItem("mlir-diff-mode") || "Simple";
            if (!DIFF_MODES.includes(diffMode)) {
                diffMode = "Simple";
            }

            const diffBtn = document.getElementById("diff-toggle");
            if (diffBtn) {
                diffBtn.innerText = "Diff: " + diffMode;
            }

            const data = DATA_PLACEHOLDER;
            let collapsedFunctions = new Set();
            let pinnedFunction = null;
            let currentIndex = -1;
            
            function escapeHtml(text) {
                return text ? text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;") : "";
            }

            window.addEventListener('keydown', function(e) {
                if (e.key === "ArrowDown" || e.key === "ArrowRight") {
                    if (currentIndex < data.length - 1) showPass(currentIndex + 1);
                    else showPass(0);
                    e.preventDefault();
                } else if (e.key === "ArrowUp" || e.key === "ArrowLeft") {
                    if (currentIndex > 0) showPass(currentIndex - 1);
                    else if(currentIndex === 0) showPass(data.length - 1);
                    else showPass(0);
                    e.preventDefault();
                }
            });

            function handleToggle(funcName, isOpen) {
                if (isOpen) collapsedFunctions.delete(funcName);
                else collapsedFunctions.add(funcName);
            }

            function togglePin(e, funcName) {
                e.stopPropagation();
                pinnedFunction = (pinnedFunction === funcName) ? null : funcName;
                updatePinStatus();
                if (currentIndex !== -1) showPass(currentIndex);
            }

            function updatePinStatus() {
                const status = document.getElementById('pin-status');
                status.innerText = pinnedFunction ? `Tracking: ${pinnedFunction}` : "";
            }

            function toggleAll(expand) {
                document.querySelectorAll('details').forEach(d => {
                    d.open = expand;
                    handleToggle(d.getAttribute('data-func'), expand);
                });
            }

            function toggleDiffMode() {
                const idx = DIFF_MODES.indexOf(diffMode);
                diffMode = DIFF_MODES[(idx + 1) % DIFF_MODES.length];
                localStorage.setItem("mlir-diff-mode", diffMode);
                document.getElementById('diff-toggle').innerText =
                    "Diff: " + diffMode;
                if (currentIndex !== -1) showPass(currentIndex);
            }


            function fastDiff(currentBody, prevBody) {
                const currentLines = currentBody.split('\n');
                if (currentLines[currentLines.length-1] === "") currentLines.pop();
                const prevLinesSet = prevBody ? new Set(prevBody.split('\n').map(l => l.trim())) : null;
                
                return currentLines.map(line => {
                    const trimmedLine = line.trim();
                    const isNew = prevLinesSet && trimmedLine.length > 0 && !prevLinesSet.has(trimmedLine);
                    const cssClass = isNew ? 'line highlighted' : 'line';
                    return `<span class="${cssClass}">${escapeHtml(line) || ' '}</span>`;
                }).join('');
            }

            function extractMlirOp(line) {
                const m = line.match(/=\s*([a-zA-Z0-9_.]+)\b/);
                return m ? m[1] : null;
            }

            function extractMlirOpKey(line) {
                const opMatch = line.match(/=\s*([a-zA-Z0-9_.]+)\b/);
                if (!opMatch) return null;

                const op = opMatch[1];

                if (op.endsWith(".call")) {
                    const calleeMatch = line.match(/@[\w\d_.]+/);
                    if (calleeMatch) {
                        return `${op} ${calleeMatch[0]}`;
                    }
                }

                return op;
            }

            function inlineCharDiff(oldLine, newLine) {
                const parts = Diff.diffChars(oldLine, newLine);
                let html = "";

                parts.forEach(p => {
                    const text = escapeHtml(p.value);
                    if (p.added) {
                        html += `<span class="char-added">${text}</span>`;
                    } else if (p.removed) {
                        html += `<span class="char-removed">${text}</span>`;
                    } else {
                        html += text;
                    }
                });
                return html;
            }

            function opAwareDiff(currentBody, prevBody) {
                if (!prevBody) {
                    return currentBody.split('\n').map(
                        l => `<span class="line">${escapeHtml(l) || ' '}</span>`
                    ).join('');
                }

                const diff = Diff.diffLines(prevBody, currentBody);
                let html = "";

                let removedBuffer = [];

                diff.forEach(part => {
                    const lines = part.value.split('\n');
                    if (lines[lines.length - 1] === "") lines.pop();

                    if (part.removed) {
                        removedBuffer = lines;
                        return;
                    }

                    if (part.added && removedBuffer.length) {
                        const count = Math.max(lines.length, removedBuffer.length);

                        for (let i = 0; i < count; i++) {
                            const oldLine = removedBuffer[i];
                            const newLine = lines[i];

                            const oldOp = oldLine ? extractMlirOpKey(oldLine) : null;
                            const newOp = newLine ? extractMlirOpKey(newLine) : null;

                            if (oldOp && newOp && oldOp !== newOp) {
                                if (oldLine) {
                                    html += `<span class="line deleted">${escapeHtml(oldLine)}</span>`;
                                }
                                if (newLine) {
                                    html += `<span class="line added">${escapeHtml(newLine)}</span>`;
                                }
                            }
                            else if (oldLine && newLine && oldOp && newOp && oldOp === newOp) {
                                html += `<span class="line added">${inlineCharDiff(oldLine, newLine)}</span>`;
                            }
                            else {
                                if (oldLine) {
                                    html += `<span class="line deleted">${escapeHtml(oldLine)}</span>`;
                                }
                                if (newLine) {
                                    html += `<span class="line added">${escapeHtml(newLine)}</span>`;
                                }
                            }
                        }

                        removedBuffer = [];
                        return;
                    }

                    if (removedBuffer.length) {
                        for (const l of removedBuffer) {
                            html += `<span class="line deleted">${escapeHtml(l)}</span>`;
                        }
                        removedBuffer = [];
                    }

                    for (const line of lines) {
                        if (part.added) {
                            html += `<span class="line added">${escapeHtml(line) || ' '}</span>`;
                        } else {
                            html += `<span class="line">${escapeHtml(line) || ' '}</span>`;
                        }
                    }
                });

                return html;
            }

            function charDiff(currentBody, prevBody) {
                if (!prevBody) {
                    return currentBody.split('\n').map(
                        l => `<span class="line">${escapeHtml(l) || ' '}</span>`
                    ).join('');
                }

                const diff = Diff.diffLines(prevBody, currentBody);
                let html = "";

                let lastRemovedLines = [];

                diff.forEach(part => {
                    const lines = part.value.split('\n');
                    if (lines[lines.length - 1] === "") lines.pop();

                    if (part.removed) {
                        lastRemovedLines = lines;
                        return;
                    }

                    if (part.added && lastRemovedLines.length) {
                        const count = Math.max(lines.length, lastRemovedLines.length);
                        for (let i = 0; i < count; i++) {
                            const oldLine = lastRemovedLines[i] || "";
                            const newLine = lines[i] || "";
                            html += `<span class="line highlighted">${inlineCharDiff(oldLine, newLine) || ' '}</span>`;
                        }
                        lastRemovedLines = [];
                        return;
                    }

                    lastRemovedLines = [];
                    for (const line of lines) {
                        if (part.added) {
                            html += `<span class="line highlighted">${escapeHtml(line) || ' '}</span>`;
                        } else {
                            html += `<span class="line">${escapeHtml(line) || ' '}</span>`;
                        }
                    }
                });

                return html;
            }

            function libraryDiff(currentBody, prevBody) {
                console.log("Calculating diff with library...");
                if (!prevBody) {
                    return currentBody.split('\n').map(
                        l => `<span class="line">${escapeHtml(l) || ' '}</span>`
                    ).join('');
                }

                const diff = Diff.diffLines(prevBody, currentBody);
                let html = "";

                diff.forEach(part => {
                    const lines = part.value.split('\n');
                    if (lines[lines.length - 1] === "") lines.pop();

                    for (const line of lines) {
                        if (part.added) {
                            html += `<span class="line highlighted">${escapeHtml(line) || ' '}</span>`;
                        } else if (part.removed) {
                            html += `<span class="line" style="background:#ffecec;color:#999;">- ${escapeHtml(line)}</span>`;
                        } else {
                            html += `<span class="line">${escapeHtml(line) || ' '}</span>`;
                        }
                    }
                });

                return html;
            }

            function getDiffedBody(currentBody, prevBody) {
                if(diffMode === "Char") {
                    return charDiff(currentBody, prevBody);
                } else if(diffMode === "Op-Aware") {
                    return opAwareDiff(currentBody, prevBody);
                } else if(diffMode === "Library") {
                    return libraryDiff(currentBody, prevBody);
                } else {
                    return fastDiff(currentBody, prevBody);
                }
            }

            function parseFunctions(ir) {
                const funcs = {};
                const stdRegex = /((?:func\.func|llvm\.func|module|tt\.func).*?(@[\w\d_.]+).*?\{)([\s\S]*?)(?=\n\s*(?:func\.func|llvm\.func|module|tt\.func|"func\.func")|$)/g;
                const quoteRegex = /("func\.func"\(.*sym_name\s*=\s*"([\w\d_.]+)".*?\{)([\s\S]*?)(?=\n\s*(?:func\.func|llvm\.func|module|tt\.func|"func\.func")|$)/g;

                let match;
                while ((match = stdRegex.exec(ir)) !== null) {
                    funcs[match[2]] = { header: match[1], body: match[3] };
                }
                while ((match = quoteRegex.exec(ir)) !== null) {
                    const name = match[2].startsWith('@') ? match[2] : '@' + match[2];
                    funcs[name] = { header: match[1], body: match[3] };
                }
                return funcs;
            }

            function showPass(index) {
                currentIndex = index;
                const item = data[index];
                const prevItem = index > 0 ? data[index - 1] : null;
                const currentFuncs = parseFunctions(item.ir);
                const prevFuncs = prevItem ? parseFunctions(prevItem.ir) : {};
                
                const items = document.querySelectorAll('.pass-item');
                items.forEach((el, i) => el.classList.toggle('active', i === index));
                if (items[index]) items[index].scrollIntoView({ block: 'nearest', behavior: 'smooth' });
                
                let htmlOutput = `<h2>${escapeHtml(item.title)}</h2>`;
                let foundAny = false;

                const sortedFuncNames = Object.keys(currentFuncs).sort();

                for (const name of sortedFuncNames) {
                    const parts = currentFuncs[name];
                    foundAny = true;
                    const isOpen = !collapsedFunctions.has(name);
                    const isPinned = pinnedFunction === name;
                    const prevBody = prevFuncs[name] ? prevFuncs[name].body : null;
                    
                    htmlOutput += `
                        <details ${isOpen || isPinned ? "open" : ""} id="func-${name.replace(/[@."]/g,'')}" data-func="${name}" ontoggle="handleToggle('${name}', this.open)">
                            <summary>
                                <button class="pin-btn ${isPinned ? 'active' : ''}" onclick="togglePin(event, '${name}')">
                                    ${isPinned ? 'Pinned' : 'Pin'}
                                </button>
                                ${escapeHtml(parts.header)}
                            </summary>
                            <pre>${getDiffedBody(parts.body, prevBody)}</pre>
                        </details>`;
                }

                if (!foundAny) {
                    htmlOutput += `<pre class="line" style="padding:15px;">${escapeHtml(item.ir)}</pre>`;
                }

                document.getElementById('display-area').innerHTML = htmlOutput;

                if (pinnedFunction && currentFuncs[pinnedFunction]) {
                    const targetId = `func-${pinnedFunction.replace(/[@."]/g,'')}`;
                    const target = document.getElementById(targetId);
                    if (target) target.scrollIntoView();
                } else {
                    document.getElementById('content').scrollTop = 0;
                }
            }
        </script>
    </body>
    </html>
    """

    sidebar_html = "".join([f'<div class="pass-item" onclick="showPass({i})">{p["title"]}</div>' for i, p in enumerate(passes)])
    full_html = html_template.replace("SIDEBAR_PLACEHOLDER", sidebar_html).replace("DATA_PLACEHOLDER", json.dumps(passes))

    with open(output_file, 'w') as f:
        f.write(full_html)
    print(f"Viewer generated: {output_file}")

if __name__ == "__main__":
    generate_html('log.txt', 'mlir_viewer.html')