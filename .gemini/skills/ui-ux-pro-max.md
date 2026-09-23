# UI-UX-PRO-MAX & Motion Design Guidelines

## 1. Container Discipline (Anti-"Card Soup")
- Do NOT wrap every block of text in a bordered card or background container.
- Use visual cards ONLY for clickable, selectable, or interactive elements (e.g., CLI Terminal, Calculator Widget, Code Copy Box).
- Expose metrics, architectural flow steps, and text content directly on the canvas using typography hierarchy and vertical rhythm.

## 2. Spatial Scale (Strict Spacing Hierarchy)
- **Micro (Inside elements):** 8px (`p-2`, `gap-2`) to 16px (`p-4`).
- **Macro (Between related components):** 24px (`gap-6`, `space-y-6`).
- **Sectional (Canvas breathing room):** 64px to 96px (`py-16` to `py-24`).

## 3. Engineering Micro-copy (Action + Result + Metric)
- Headlines must state: [Action] + [Quantified Result] + [Benchmark].
- Example: "Compress MCP tool context by 85% with a local hybrid vector proxy."

## 4. Responsive Verification
- Validate layout on 390px (Mobile portrait), 768px (Tablet), and 1440px (Desktop).
- No unexpected horizontal scrolling (`w-full max-w-screen overflow-x-hidden`).
- Touch targets must be minimum 44x44px.

## 5. Complete Interaction States & Motion
Every interactive element must provide:
1. **Idle**: Subdued slate/zinc palette.
2. **Hover/Focus**: Smooth CSS transition or Motion spring (`stiffness: 300, damping: 25`).
3. **Active/Loading**: Micro-interaction feedback.
4. **Success/Copied**: Instant icon/text state change with a spring pop.

## 6. Integration with 21st.dev Components
- When implementing components inspired by 21st.dev:
  - Strip excessive gradient mesh backgrounds.
  - Adapt them to the SchemaSlim monochrome palette (pure white, zinc-400, zinc-900, slate borders, emerald/green-400 for success).
  - Use `motion/react` layout transitions instead of raw CSS hacks.