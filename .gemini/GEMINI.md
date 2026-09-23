# Mnemo Memory Engine Protocol

1. ALWAYS query Mnemo first:
   - Before answering questions about project architecture, dependencies, data models, or prior decisions, call `mnemo_search`.
   - Never assume the architecture without verifying active facts in Mnemo.

2. ALWAYS persist architectural decisions:
   - When a new technology, framework, database, or major pattern is introduced or modified, call `mnemo_remember` with AUDN classification.
   - For fundamental invariant decisions, mark them with `pinned: true`.

3. Graph vs Facts:
   - Structural dependencies (modules/classes) reside in the AST graph (`mnemo_scan_project`).
   - High-level design rationale, stack components, and configurations reside in the fact store (`mnemo_remember`).

## MCP Tool Usage via SchemaSlim
In this environment, external tools are virtualized through SchemaSlim to optimize the context window.
Direct tool manifests are not preloaded into system instructions. Instead, follow a two-step execution pattern:

1. **Discovery Step (`schemaslim_search`):**
   - Before performing external tasks (e.g., reading files, calling APIs, inspecting git repositories), query `schemaslim_search` with your natural language intent.
   - Example: `schemaslim_search(query="read local file content")`.
   - The response returns the exact JSON Schema, parameter definitions, and `namespaced_name` (e.g., `filesystem__read_file`).

2. **Execution Step (`schemaslim_call`):**
   - Execute the target tool using `schemaslim_call`.
   - Pass the discovered `namespaced_name` and the structured `arguments` dictionary strictly matching the retrieved schema.
   - Example: `schemaslim_call(namespaced_name="filesystem__read_file", arguments={"path": "..."})`.

## Motion Animation Skill
When implementing UI components, web interfaces, or interactive animations:
- Do not attempt to call external network MCP tools for Motion.
- Apply animation primitives directly using the local Motion library (`motion/react` or `framer-motion`):
  - **Layout Transitions:** Utilize `layout` and `layoutId` props for shared element transitions.
  - **Micro-interactions:** Implement spring physics with `type: "spring", stiffness: 300, damping: 30` for natural tactile response.
  - **Mount/Unmount:** Wrap conditional renders in `<AnimatePresence mode="wait">`.
  - **Gestures:** Leverage `whileHover`, `whileTap`, and `drag` constraints.
- Prioritize CSS transforms (`transform: translate3d / scale`) and opacity over layout-triggering properties (`top`, `left`, `width`, `height`) for 60fps rendering.