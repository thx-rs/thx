# thx

<img src="https://raw.githubusercontent.com/thx-rs/thx/refs/heads/main/logo.svg" alt="thx: terminal-first AI agent with a TUI">

```bash
cargo install thx
# or
curl -fsSL https://raw.githubusercontent.com/thx-rs/thx/main/install.sh | sh

export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_MODEL="openai/gpt-5.6-luna"

thx
```

#### Add SHELL

```bash
cargo install shellvibe

cat > mcp.json <<'EOF'
{
  "mcpServers": {
    "shellvibe": {
      "command": "shellvibe",
      "args": ["--deny-exec", "rm"]
    }
  }
}
EOF

thx
```
