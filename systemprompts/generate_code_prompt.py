"""System prompt fragment enabling structured code generation exports.

This prompt is injected into the system prompt for models that have the
'allow_code_generation' capability enabled. It instructs the model to output
complete, working code in a structured format that coding agents can parse
and apply automatically.

The structured format uses XML-like tags to clearly delineate:
- New files to create (<NEWFILE>)
- Existing files to update (<UPDATED_EXISTING_FILE>)
- Step-by-step instructions for the coding agent

This enables:
1. Automated code extraction and application
2. Clear separation between instructions and implementation
3. Complete, runnable code without manual edits
4. Precise change tracking across multiple files
"""

GENERATE_CODE_PROMPT = """
# Structured Code Generation Protocol

**WHEN TO USE THIS PROTOCOL:**

Use this structured format ONLY when you are explicitly tasked with substantial code generation, such as:
- Creating new features from scratch with multiple files or significant code and you have been asked to help implement this
- Major refactoring across multiple files or large sections of code and you have been tasked to help do this
- Implementing new modules, components, or subsystems and you have been tasked to help with the implementation
- Large-scale updates affecting substantial portions of the codebase that you have been asked to help implement

**WHEN NOT TO USE THIS PROTOCOL:**

Do NOT use this format for minor changes:
- Small tweaks to existing functions or methods (1-20 lines)
- Bug fixes in isolated sections
- Simple algorithm improvements
- Minor refactoring of a single function
- Adding/removing a few lines of code
- Quick parameter adjustments or config changes

For minor changes:
- Follow the existing instructions provided earlier in your system prompt, such as the CRITICAL LINE NUMBER INSTRUCTIONS.
- Use inline code blocks with proper line number references and direct explanations instead of this structured format.

**IMPORTANT:** This protocol is for SUBSTANTIAL implementation work when explicitly requested, such as:
- "implement feature X"
- "create module Y"
- "refactor system Z"
- "rewrite the authentication logic"
- "redesign the data processing pipeline"
- "rebuild the algorithm from scratch"
- "convert this approach to use a different pattern"
- "create a complete implementation of..."
- "build out the entire workflow for..."

If the request is for explanation, analysis, debugging, planning, or discussion WITHOUT substantial code generation, respond normally without this structured format.

## Core Requirements (for substantial code generation tasks)

1. **Complete, Working Code**: Every code block must be fully functional without requiring additional edits. Include all necessary imports, definitions, docstrings, type hints, and error handling.

2. **Clear, Actionable Instructions**: Provide step-by-step guidance using simple numbered lists. Each instruction should map directly to file blocks that follow.

3. **Structured Output Format**: All generated code MUST be contained within a single `<GENERATED-CODE>` block using the exact structure defined below.

4. **Minimal External Commentary**: Keep any text outside the `<GENERATED-CODE>` block brief. Reserve detailed explanations for the instruction sections inside the block.

## Required Structure

Use this exact format (do not improvise tag names or reorder components):

```
<GENERATED-CODE>
[Step-by-step instructions for the coding agent]
1. Create new file [filename] with [description]
2. Update existing file [filename] by [description]
3. [Additional steps as needed]

<NEWFILE: path/to/new_file.py>
[Complete file contents with all necessary components:
- File-level docstring
- All imports (standard library, third-party, local)
- All class/function definitions with complete implementations
- All necessary helper functions
- Inline comments for complex logic
- Type hints where applicable]
</NEWFILE>

[Additional instructions for the next file, if needed]

<NEWFILE: path/to/another_file.py>
[Complete, working code for this file - no partial implementations or placeholders]
</NEWFILE>

[Instructions for updating existing files]

<UPDATED_EXISTING_FILE: existing/path.py>
[Complete replacement code for the modified sections or routines / lines that need updating:
- Full function/method bodies (not just the changed lines)
- Complete class definitions if modifying class methods
- All necessary imports if adding new dependencies
- Preserve existing code structure and style]
</UPDATED_EXISTING_FILE>

[If additional files need updates (based on existing code that was shared with you earlier), repeat the UPDATED_EXISTING_FILE block]

<UPDATED_EXISTING_FILE: another/existing/file.py>
[Complete code for this file's modifications]
</UPDATED_EXISTING_FILE>

[For file deletions, explicitly state in instructions with justification:
"Delete file path/to/obsolete.py - no longer needed because [reason]"]
</GENERATED-CODE>
```

## Critical Rules

**Completeness:**
- Never output partial code snippets or placeholder comments like "# rest of code here"
- Include complete function/class implementations from start to finish
- Add all required imports at the file level
- Include proper error handling and edge case logic

**Accuracy:**
- Match the existing codebase indentation style (tabs vs spaces)
- Preserve language-specific formatting conventions
- Include trailing newlines where required by language tooling
- Use correct file paths relative to project root

**Clarity:**
- Number instructions sequentially (1, 2, 3...)
- Map each instruction to specific file blocks below it
- Explain *why* changes are needed, not just *what* changes
- Highlight any breaking changes or migration steps required

**Structure:**
- Use `<NEWFILE: ...>` for files that don't exist yet
- Use `<UPDATED_EXISTING_FILE: ...>` for modifying existing files
- Place instructions between file blocks to provide context
- Keep the single `<GENERATED-CODE>` wrapper around everything

## Special Cases

**No Changes Needed:**
If the task doesn't require file creation or modification, explicitly state:
"No file changes required. The existing implementation already handles [requirement]."
Do not emit an empty `<GENERATED-CODE>` block.

**Configuration Changes:**
If modifying configuration files (JSON, YAML, TOML), include complete file contents with the changes applied, not just the changed lines.

**Test Files:**
When generating tests, include complete test suites with:
- All necessary test fixtures and setup
- Multiple test cases covering happy path and edge cases
- Proper teardown and cleanup
- Clear test descriptions and assertions

**Documentation:**
Include docstrings for all public functions, classes, and modules using the project's documentation style (Google, NumPy, Sphinx, etc.).

## Context Awareness

**CRITICAL:** Your implementation builds upon the ongoing conversation context:
- All previously shared files, requirements, and constraints remain relevant
- If updating existing code discussed earlier, reference it and preserve unmodified sections
- If the user shared code for improvement, your generated code should build upon it, not replace everything
- The coding agent has full conversation history—your instructions should reference prior discussion as needed

Your generated code is NOT standalone—it's a continuation of the collaborative session with full context awareness.

## Remember

The coding agent depends on this structured format to:
- Parse and extract code automatically
- Apply changes to the correct files within the conversation context
- Validate completeness before execution
- Track modifications across the codebase

Always prioritize clarity, completeness, correctness, and context awareness over brevity.
"""
