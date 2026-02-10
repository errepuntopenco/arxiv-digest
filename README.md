# arxiv-digest

A daily arXiv paper digest tailored to your research interests. Outputs a markdown snippet you can drop into Obsidian, Logseq, or any note-taking app.

## What it does

- Searches arXiv categories you care about for new papers
- Filters by configurable keywords with priority scoring
- Tracks papers by your collaborators and other authors of interest
- Checks Semantic Scholar for recent citations to your work
- Writes a concise markdown digest to your daily note

## Quick start

```bash
git clone https://github.com/youruser/arxiv-digest.git
cd arxiv-digest
pip install -r requirements.txt

# Create your config
cp config_example.yaml config.yaml
# Edit config.yaml with your categories, keywords, and collaborators

# Run
python arxiv_digest.py --dry-run       # preview
python arxiv_digest.py                  # write to file
```

## Configuration

All settings live in `config.yaml` (git-ignored). See `config_example.yaml` for the full annotated template.

### Key sections

| Section | What it controls |
|---------|-----------------|
| `author` | Your Semantic Scholar ID for citation tracking |
| `output` | Where the markdown file goes and its filename pattern |
| `category_groups` | arXiv categories, keywords, and filtering thresholds |
| `collaborators` | Close collaborators — their papers always appear |
| `others` | Broader community — shown if not already in another section |

### Keyword filtering

Each category group has `high_priority` and `secondary` keywords with configurable thresholds:

```yaml
keywords:
  high_priority:
    - "effective field theory"
    - "black hole"
  secondary:
    - "review"
    - "survey"
  min_high: 2        # need ≥2 high-priority hits
  min_high_alt: 1    # OR ≥1 high-priority ...
  min_secondary: 2   # ... AND ≥2 secondary hits
```

Optional `include_if` / `exclude_if` lists gate papers before keyword scoring.

### Author tracking

List name variants as they appear on arXiv (usually `Lastname, F` or `Lastname, Firstname`):

```yaml
collaborators:
  "Alice Doe":
    - "Doe, A"
    - "Doe, Alice"
```

### Citation tracking

Set your Semantic Scholar author ID to track citations to your most-cited papers:

```yaml
author:
  semantic_scholar_id: "34252686"
```

Find your ID at [semanticscholar.org](https://www.semanticscholar.org/) — it's the number in your profile URL.

## Usage

```bash
python arxiv_digest.py                          # today's digest
python arxiv_digest.py --date 2026-02-10        # specific date
python arxiv_digest.py --dry-run                # print without writing
python arxiv_digest.py --verbose                # debug logging
python arxiv_digest.py --config my_config.yaml  # custom config path
```

## Scheduling (macOS)

To run automatically on weekday mornings, create a Launch Agent:

```bash
# Create the plist
cat > ~/Library/LaunchAgents/com.user.arxiv-digest.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.arxiv-digest</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/.venv/bin/python</string>
        <string>/path/to/arxiv_digest.py</string>
        <string>--config</string>
        <string>/path/to/config.yaml</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
        <!-- Monday–Friday at 06:00 -->
        <dict><key>Weekday</key><integer>1</integer>
              <key>Hour</key><integer>6</integer>
              <key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>2</integer>
              <key>Hour</key><integer>6</integer>
              <key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>3</integer>
              <key>Hour</key><integer>6</integer>
              <key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>4</integer>
              <key>Hour</key><integer>6</integer>
              <key>Minute</key><integer>0</integer></dict>
        <dict><key>Weekday</key><integer>5</integer>
              <key>Hour</key><integer>6</integer>
              <key>Minute</key><integer>0</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>/tmp/arxiv-digest-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/arxiv-digest-stderr.log</string>
</dict>
</plist>
EOF

# Load it
launchctl load ~/Library/LaunchAgents/com.user.arxiv-digest.plist
```

If your Mac is asleep at 6 AM, the job runs as soon as you open the lid.

## Output format

```markdown
- 06:00 arxiv
    - hep-th
        - [Paper Title](https://arxiv.org/abs/...) — Author1, Author2 et al. - Keywords: EFT, black hole
    - collaborators
        - [Paper Title](https://arxiv.org/abs/...) — Author1, Author2 - Alice Doe
    - citations
        - [Citing Paper](https://arxiv.org/abs/...) - cites "Your Paper Title"
```

## How it works

1. Queries the arXiv API for each category group
2. Scores papers by keyword matches (weighted: title hits count more)
3. Deduplicates cross-listed papers (first matching section wins)
4. Scans all fetched papers for collaborator/other author names
5. Checks Semantic Scholar for recent citations to your top-cited papers
6. Writes everything to a dated markdown file

API responses are cached in `~/.cache/arxiv_digest/` to avoid redundant queries during development.

## License

MIT
