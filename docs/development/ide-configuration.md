# IDE Configuration Guide

## Overview

This guide provides detailed configuration instructions for popular IDEs and editors used with the Tracktion project. Proper IDE configuration enhances productivity through intelligent code completion, debugging, and integrated testing.

## Visual Studio Code (Recommended)

### Essential Extensions

#### Python Development
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-python.debugpy",
    "ms-python.autopep8"
  ]
}
```

#### General Development
```json
{
  "recommendations": [
    "ms-vscode.vscode-json",
    "redhat.vscode-yaml",
    "ms-vscode.makefile-tools",
    "eamodio.gitlens",
    "github.vscode-pull-request-github",
    "ms-vscode.vscode-docker",
    "bradlc.vscode-tailwindcss"
  ]
}
```

#### Documentation and Markdown
```json
{
  "recommendations": [
    "yzhang.markdown-all-in-one",
    "davidanson.vscode-markdownlint",
    "bierner.markdown-mermaid"
  ]
}
```

### Workspace Settings

Create `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.terminal.activateEnvironment": true,

  // Ruff configuration
  "ruff.enable": true,
  "ruff.organizeImports": true,
  "ruff.fixAll": true,

  // MyPy configuration
  "mypy-type-checker.args": [
    "--config-file=pyproject.toml"
  ],

  // Python formatting
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true,
      "source.fixAll": true
    }
  },

  // File associations
  "files.associations": {
    "*.env.example": "properties",
    "docker-compose*.yml": "dockercompose",
    "alembic.ini": "ini"
  },

  // Exclude patterns
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    "**/.pytest_cache": true,
    "**/.mypy_cache": true,
    "**/node_modules": true,
    ".venv": false
  },

  // Testing configuration
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests/"
  ],
  "python.testing.unittestEnabled": false,

  // Editor settings
  "editor.rulers": [120],
  "editor.tabSize": 4,
  "editor.insertSpaces": true,
  "editor.trimAutoWhitespace": true,
  "files.trimTrailingWhitespace": true,

  // Git settings
  "git.enableSmartCommit": false,
  "git.confirmSync": false,
  "git.autofetch": true,

  // Docker settings
  "docker.showStartPage": false,

  // Terminal settings
  "terminal.integrated.defaultProfile.linux": "bash",
  "terminal.integrated.defaultProfile.osx": "zsh",
  "terminal.integrated.env.linux": {
    "UV_PROJECT_ENVIRONMENT": ".venv"
  },
  "terminal.integrated.env.osx": {
    "UV_PROJECT_ENVIRONMENT": ".venv"
  }
}
```

### Launch Configuration

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Analysis Service",
      "type": "python",
      "request": "launch",
      "program": "services/analysis_service/src/main.py",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "python": ".venv/bin/python"
    },
    {
      "name": "Tracklist Service",
      "type": "python",
      "request": "launch",
      "program": "services/tracklist_service/src/main.py",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "python": ".venv/bin/python"
    },
    {
      "name": "File Watcher",
      "type": "python",
      "request": "launch",
      "program": "services/file_watcher/src/main.py",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env",
      "cwd": "${workspaceFolder}",
      "python": ".venv/bin/python"
    },
    {
      "name": "Debug Tests",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "tests/",
        "-v",
        "--no-cov"
      ],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env",
      "python": ".venv/bin/python"
    }
  ]
}
```

### Tasks Configuration

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Install Dependencies",
      "type": "shell",
      "command": "uv",
      "args": ["sync"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "uv",
      "args": ["run", "pytest", "tests/", "-v"],
      "group": "test",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "Run Pre-commit",
      "type": "shell",
      "command": "uv",
      "args": ["run", "pre-commit", "run", "--all-files"],
      "group": "build",
      "presentation": {
        "echo": true,
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "Start Docker Services",
      "type": "shell",
      "command": "docker-compose",
      "args": ["up", "-d"],
      "group": "build"
    },
    {
      "label": "Stop Docker Services",
      "type": "shell",
      "command": "docker-compose",
      "args": ["down"],
      "group": "build"
    }
  ]
}
```

### Snippets

Create `.vscode/python.json` for custom snippets:

```json
{
  "FastAPI Route": {
    "prefix": "route",
    "body": [
      "@router.${1:get}(\"/${2:endpoint}\")",
      "async def ${3:function_name}(",
      "    ${4:request}: Request,",
      "    db: AsyncSession = Depends(get_db_session)",
      ") -> ${5:ResponseModel}:",
      "    \"\"\"",
      "    ${6:Description of the endpoint}",
      "    \"\"\"",
      "    ${7:# Implementation}",
      "    return ${8:result}"
    ]
  },
  "Async Test Function": {
    "prefix": "atest",
    "body": [
      "@pytest.mark.asyncio",
      "async def test_${1:function_name}():",
      "    \"\"\"${2:Test description}\"\"\"",
      "    ${3:# Test implementation}",
      "    assert ${4:condition}"
    ]
  },
  "Logger Setup": {
    "prefix": "logger",
    "body": [
      "import structlog",
      "",
      "logger = structlog.get_logger(__name__)"
    ]
  }
}
```

## PyCharm Professional

### Project Setup

1. **Open Project**: File → Open → Select tracktion directory
2. **Python Interpreter**: File → Settings → Project → Python Interpreter
   - Add New Environment → Existing Environment
   - Select `.venv/bin/python`

### Essential Settings

#### Code Quality
```
File → Settings → Tools → External Tools

Add tools:
- Name: Ruff Check
  Program: $ProjectFileDir$/.venv/bin/uv
  Arguments: run ruff check $FilePath$
  Working directory: $ProjectFileDir$

- Name: MyPy Check
  Program: $ProjectFileDir$/.venv/bin/uv
  Arguments: run mypy $FilePath$
  Working directory: $ProjectFileDir$
```

#### Run Configurations

Create run configurations for each service:
```
Run → Edit Configurations → + → Python

Analysis Service:
- Script path: services/analysis_service/src/main.py
- Environment variables: Load from .env
- Working directory: Project root
- Python interpreter: Project interpreter
```

#### Database Integration
```
Database → + → Data Source → PostgreSQL
- Host: localhost
- Port: 5432
- Database: tracktion_dev
- Username: tracktion
- Password: [from .env]
```

### Plugins
- **Requirements**: For requirements.txt syntax
- **Docker**: For Docker integration
- **Database Tools**: For database management (Professional only)
- **Git Integration**: Enhanced git features (Professional only)

## Vim/Neovim

### Required Plugins (using vim-plug)

```vim
call plug#begin()

" Python support
Plug 'davidhalter/jedi-vim'
Plug 'nvim-treesitter/nvim-treesitter', {'do': ':TSUpdate'}
Plug 'neoclide/coc.nvim', {'branch': 'release'}

" Linting and formatting
Plug 'dense-analysis/ale'
Plug 'psf/black', { 'branch': 'stable' }

" Git integration
Plug 'tpope/vim-fugitive'
Plug 'airblade/vim-gitgutter'

" File navigation
Plug 'preservim/nerdtree'
Plug 'junegunn/fzf.vim'

" General
Plug 'vim-airline/vim-airline'
Plug 'tpope/vim-commentary'

call plug#end()
```

### Configuration (.vimrc/.config/nvim/init.vim)

```vim
" Python configuration
let g:python3_host_prog = './.venv/bin/python'

" ALE configuration for linting
let g:ale_linters = {
\   'python': ['ruff', 'mypy'],
\}
let g:ale_fixers = {
\   'python': ['ruff'],
\}
let g:ale_fix_on_save = 1

" CoC configuration
let g:coc_global_extensions = ['coc-python', 'coc-json', 'coc-yaml']

" Key mappings
nnoremap <leader>t :terminal uv run pytest tests/<CR>
nnoremap <leader>p :terminal uv run pre-commit run --all-files<CR>
nnoremap <leader>r :terminal uv run python %<CR>

" File type settings
autocmd FileType python setlocal tabstop=4 shiftwidth=4 expandtab
autocmd FileType yaml setlocal tabstop=2 shiftwidth=2 expandtab
autocmd FileType json setlocal tabstop=2 shiftwidth=2 expandtab

" Project-specific settings
set number
set relativenumber
set colorcolumn=120
set expandtab
set tabstop=4
set shiftwidth=4
```

## Sublime Text

### Package Installation

Install Package Control, then install:
- **Anaconda**: Python IDE features
- **SublimeLinter**: Linting framework
- **SublimeLinter-ruff**: Ruff integration
- **GitGutter**: Git integration
- **BracketHighlighter**: Bracket matching
- **DocBlockr**: Documentation generation

### Project Settings

Create `tracktion.sublime-project`:

```json
{
  "folders": [
    {
      "path": ".",
      "folder_exclude_patterns": [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules"
      ]
    }
  ],
  "settings": {
    "python_interpreter": "./.venv/bin/python",
    "rulers": [120],
    "tab_size": 4,
    "translate_tabs_to_spaces": true,
    "trim_trailing_white_space_on_save": true,

    "SublimeLinter.linters.ruff.args": ["--config", "pyproject.toml"],

    "anaconda_linting": false,
    "anaconda_linting_behaviour": "save-only",

    "build_systems": [
      {
        "name": "uv test",
        "cmd": ["uv", "run", "pytest", "tests/", "-v"],
        "working_dir": "$project_path",
        "shell": true
      }
    ]
  }
}
```

## Emacs

### Package Configuration (using use-package)

```elisp
;; Python development
(use-package python-mode
  :ensure t
  :config
  (setq python-shell-interpreter "./.venv/bin/python"))

(use-package lsp-mode
  :ensure t
  :hook ((python-mode . lsp))
  :commands lsp
  :config
  (setq lsp-python-executable-cmd "./.venv/bin/python"))

(use-package lsp-pyright
  :ensure t
  :hook (python-mode . (lambda ()
                          (require 'lsp-pyright)
                          (lsp))))

(use-package flycheck
  :ensure t
  :init (global-flycheck-mode))

;; Git integration
(use-package magit
  :ensure t
  :bind ("C-x g" . magit-status))

;; Project management
(use-package projectile
  :ensure t
  :init
  (projectile-mode +1)
  :bind (:map projectile-mode-map
              ("C-c p" . projectile-command-map)))
```

### Directory Local Variables

Create `.dir-locals.el`:

```elisp
((python-mode . ((python-shell-interpreter . "./.venv/bin/python")
                 (flycheck-checker . python-ruff)
                 (fill-column . 120))))
```

## General IDE Tips

### Code Navigation
- **Go to Definition**: Essential for understanding service interactions
- **Find References**: Track function usage across services
- **Symbol Search**: Quick navigation in large codebases
- **File Search**: Efficient project-wide file location

### Debugging Features
- **Breakpoint Management**: Strategic debugging points
- **Variable Inspection**: Runtime state examination
- **Call Stack**: Trace execution flow
- **Interactive Console**: Runtime code evaluation

### Git Integration
- **Diff Viewing**: Compare changes before commits
- **Blame/Annotate**: Track code authorship
- **Branch Management**: Visual git workflow
- **Merge Conflict Resolution**: Integrated resolution tools

### Testing Integration
- **Test Runner**: Integrated pytest execution
- **Test Discovery**: Automatic test detection
- **Coverage Reports**: Visual test coverage
- **Test Debugging**: Debug failing tests

## Common Workflow Shortcuts

### VS Code
- `Ctrl+Shift+P`: Command Palette
- `Ctrl+P`: Quick File Open
- `F5`: Start Debugging
- `Ctrl+Shift+\``: Split Terminal
- `Ctrl+```: Toggle Terminal

### General Development
- `F12`: Go to Definition
- `Shift+F12`: Find All References
- `Ctrl+Shift+F`: Search in Files
- `Ctrl+Shift+R`: Refactor Symbol

## Troubleshooting

### Python Interpreter Issues
```bash
# Verify interpreter path
which python
ls -la .venv/bin/python

# Reset interpreter in IDE
# VS Code: Ctrl+Shift+P -> Python: Select Interpreter
```

### Extension/Plugin Issues
```bash
# Clear extension cache
rm -rf ~/.vscode/extensions/cache
# Restart IDE
```

### Linting Configuration Issues
```bash
# Verify configuration files
cat pyproject.toml | grep -A 10 "\[tool.ruff\]"
uv run ruff --version
```

### Performance Issues
- **Exclude large directories** from indexing
- **Disable unnecessary extensions**
- **Increase memory allocation** for Java-based IDEs
- **Use project-specific settings** instead of global

## Next Steps

After IDE configuration:
1. **Test debugging**: Set breakpoints and debug a service
2. **Verify linting**: Check that code quality tools work
3. **Test git integration**: Make a commit using IDE
4. **Configure shortcuts**: Set up personal key bindings
5. **Review team conventions**: Align with team coding standards

## Team Recommendations

### Standardization
- **Shared settings**: Consider team-wide VS Code settings
- **Extension recommendations**: Maintain `.vscode/extensions.json`
- **Formatting consistency**: Ensure ruff configuration works across IDEs
- **Debugging configurations**: Share launch configurations for consistency
