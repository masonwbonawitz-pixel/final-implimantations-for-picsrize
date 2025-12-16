#!/bin/bash

# Script to set up FOR_HOSTING folder as a separate GitHub Pages repository

echo "🚀 Setting up GitHub Pages repository for FOR_HOSTING folder..."
echo ""

# Check if .git already exists
if [ -d ".git" ]; then
    echo "⚠️  Warning: .git folder already exists!"
    echo "   This folder is already a git repository."
    echo "   Do you want to continue? (This will add/update files)"
    read -p "   Continue? (y/n): " answer
    if [ "$answer" != "y" ]; then
        echo "   Cancelled."
        exit 1
    fi
else
    echo "📦 Initializing new git repository..."
    git init
fi

echo ""
echo "📝 Adding all files..."
git add .

echo ""
echo "✅ Files ready to commit!"
echo ""
echo "📋 Next steps:"
echo ""
echo "1. Create a NEW repository on GitHub (don't use AlbumCover2):"
echo "   - Go to: https://github.com/new"
echo "   - Name it something like: 'riz-album-pages' or 'album-cover-site'"
echo "   - Make it PUBLIC (required for free GitHub Pages)"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. Commit and push:"
echo "   git commit -m 'Initial commit for GitHub Pages'"
echo "   git branch -M main"
echo "   git remote add origin git@github.com:masonwbonawitz-pixel/YOUR-NEW-REPO-NAME.git"
echo "   git push -u origin main"
echo ""
echo "3. Configure GitHub Pages:"
echo "   - Go to your NEW repo's Settings > Pages"
echo "   - Source: Deploy from a branch"
echo "   - Branch: main"
echo "   - Folder: / (root)"
echo "   - Save"
echo ""
echo "4. Your site will be at:"
echo "   https://masonwbonawitz-pixel.github.io/YOUR-NEW-REPO-NAME/"
echo ""

