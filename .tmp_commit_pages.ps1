$ErrorActionPreference='Stop'
Set-Location 'C:\Users\donald\abi'
Write-Host "Git version:"; git --version
Write-Host "Git remote:"; git remote -v
Write-Host "Git status (short):"; git status -s

Write-Host "GH CLI version:"; gh --version
Write-Host "GH auth status:"; gh auth status

$login = gh api user -q '.login'
$id = gh api user -q '.id'
if (-not $login) { throw "Unable to retrieve GitHub login. Run 'gh auth login'." }
if (-not $id) { throw "Unable to retrieve GitHub user id. Run 'gh auth login'." }
$email = "$id+$login@users.noreply.github.com"

# Configure git identity using GH account
Write-Host "Configuring git identity as $login <$email>"
git config user.name $login
git config user.email $email

# Stage all changes as requested
Write-Host "Staging all changes..."
git add .

# Commit if there are staged changes
$staged = git diff --cached --name-only
if ($staged) {
  $msg = "docs: branch-based Pages readiness; normalize doc paths and generator slashes"
  Write-Host "Committing: $msg"
  git commit -m $msg
} else {
  Write-Host "No changes staged; skipping commit."
}

# Push to main
Write-Host "Pushing to origin main..."
git push origin main

# Configure GitHub Pages to deploy from branch main:/docs
function Set-Pages {
  param([string]$branch='main',[string]$path='/docs')
  Write-Host "Setting GitHub Pages source to $():$()"
  gh api repos/:owner/:repo/pages -X PUT -f 'source[branch]'=$branch -f 'source[path]'=$path | Out-Host
}

try {
  Set-Pages -branch 'main' -path '/docs'
} catch {
  Write-Warning "PUT failed; enabling Pages, then setting source..."
  gh api -X POST repos/:owner/:repo/pages -f 'source[branch]'='main' -f 'source[path]'='/docs' | Out-Host
  Set-Pages -branch 'main' -path '/docs'
}

Write-Host "Current Pages settings:"
gh api repos/:owner/:repo/pages | Out-Host
