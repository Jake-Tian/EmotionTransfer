# Guide: Uploading EmotionTransfer to GitHub

## Step 1: Initialize Git Repository

```bash
cd /research/d7/gds/yztian25/EmotionTransfer
git init
```

## Step 2: Check What Will Be Committed

Before committing, verify that sensitive files are excluded:

```bash
# Check what files would be added (should NOT include .csv, models/, LLaMA-Factory/, etc.)
git status
```

The `.gitignore` file should exclude:
- All `.csv` files
- `LLaMA-Factory/` directory
- `models/` directory
- API keys and secrets
- Log files

## Step 3: Add Files to Git

```bash
# Add all files (respecting .gitignore)
git add .

# Verify what's staged
git status
```

## Step 4: Create Initial Commit

```bash
git commit -m "Initial commit: EmotionTransfer project with SFT, DPO, KTO, and ORPO training"
```

## Step 5: Create GitHub Repository

1. Go to [GitHub.com](https://github.com) and sign in
2. Click the **"+"** icon in the top right → **"New repository"**
3. Fill in:
   - **Repository name**: `EmotionTransfer` (or your preferred name)
   - **Description**: "Fine-tuning Qwen2.5-0.5B-Instruct for emotion transfer tasks"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Click **"Create repository"**

## Step 6: Connect Local Repository to GitHub

After creating the repository, GitHub will show you commands. Use these:

```bash
# Add the remote repository (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/EmotionTransfer.git

# Or if you prefer SSH (if you have SSH keys set up):
# git remote add origin git@github.com:YOUR_USERNAME/EmotionTransfer.git

# Verify the remote was added
git remote -v
```

## Step 7: Push to GitHub

```bash
# Rename default branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

If prompted for credentials:
- **Username**: Your GitHub username
- **Password**: Use a Personal Access Token (not your GitHub password)
  - Generate one at: Settings → Developer settings → Personal access tokens → Tokens (classic)
  - Or use GitHub CLI for easier authentication

## Step 8: Verify Upload

1. Go to your repository on GitHub
2. Check that:
   - README.md is visible
   - .gitignore is present
   - No `.csv` files are visible
   - No `models/` or `LLaMA-Factory/` directories are visible
   - All Python scripts are present

## Troubleshooting

### If you need to update the remote URL:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/EmotionTransfer.git
```

### If you need to force push (be careful!):
```bash
git push -u origin main --force
```

### If you want to check what files are being ignored:
```bash
git status --ignored
```

## Future Updates

After making changes:

```bash
# Check what changed
git status

# Add changed files
git add .

# Commit changes
git commit -m "Description of your changes"

# Push to GitHub
git push
```

## Important Reminders

✅ **DO** commit:
- Python scripts
- Shell scripts (.sh files)
- README.md
- .gitignore
- Configuration files (without secrets)

❌ **DON'T** commit:
- CSV files (already in .gitignore)
- Model checkpoints (already in .gitignore)
- API keys (already in .gitignore)
- Large data files
- Personal credentials

