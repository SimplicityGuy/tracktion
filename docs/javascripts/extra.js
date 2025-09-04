// Enhanced search functionality for Tracktion Documentation

document.addEventListener('DOMContentLoaded', function() {
    // Enhanced search with keyboard shortcuts
    setupSearchShortcuts();

    // Add search filters
    setupSearchFilters();

    // Add search analytics
    setupSearchAnalytics();

    // Add quick search suggestions
    setupSearchSuggestions();
});

// Keyboard shortcuts for search
function setupSearchShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + K to focus search
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.querySelector('[data-md-component="search-query"]');
            if (searchInput) {
                searchInput.focus();
            }
        }

        // Escape to clear search
        if (e.key === 'Escape') {
            const searchInput = document.querySelector('[data-md-component="search-query"]');
            if (searchInput && document.activeElement === searchInput) {
                searchInput.value = '';
                searchInput.blur();
            }
        }
    });
}

// Search filters for different content types
function setupSearchFilters() {
    const searchContainer = document.querySelector('[data-md-component="search"]');
    if (!searchContainer) return;

    // Create filter buttons
    const filterContainer = document.createElement('div');
    filterContainer.className = 'search-filters';
    filterContainer.innerHTML = `
        <div class="filter-buttons">
            <button class="filter-btn active" data-filter="all">All</button>
            <button class="filter-btn" data-filter="api">API</button>
            <button class="filter-btn" data-filter="tutorial">Tutorials</button>
            <button class="filter-btn" data-filter="setup">Setup</button>
            <button class="filter-btn" data-filter="troubleshoot">Troubleshooting</button>
        </div>
    `;

    // Insert filters after search input
    const searchQuery = searchContainer.querySelector('[data-md-component="search-query"]');
    if (searchQuery && searchQuery.parentNode) {
        searchQuery.parentNode.insertBefore(filterContainer, searchQuery.nextSibling);
    }

    // Add filter functionality
    const filterButtons = filterContainer.querySelectorAll('.filter-btn');
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            // Filter search results
            const filter = this.dataset.filter;
            filterSearchResults(filter);
        });
    });
}

// Filter search results based on content type
function filterSearchResults(filter) {
    const searchResults = document.querySelectorAll('[data-md-component="search-result"]');

    searchResults.forEach(result => {
        if (filter === 'all') {
            result.style.display = '';
            return;
        }

        const resultLink = result.querySelector('a');
        if (!resultLink) return;

        const href = resultLink.getAttribute('href') || '';
        const title = resultLink.textContent.toLowerCase();

        let shouldShow = false;

        switch (filter) {
            case 'api':
                shouldShow = href.includes('/api/') || title.includes('api');
                break;
            case 'tutorial':
                shouldShow = href.includes('/tutorials/') || title.includes('tutorial');
                break;
            case 'setup':
                shouldShow = href.includes('/setup/') || href.includes('/getting-started/') ||
                           title.includes('setup') || title.includes('install');
                break;
            case 'troubleshoot':
                shouldShow = href.includes('/troubleshooting') || href.includes('/operations/') ||
                           title.includes('troubleshoot') || title.includes('error');
                break;
        }

        result.style.display = shouldShow ? '' : 'none';
    });
}

// Search analytics and suggestions
function setupSearchAnalytics() {
    const searchInput = document.querySelector('[data-md-component="search-query"]');
    if (!searchInput) return;

    let searchTimeout;
    searchInput.addEventListener('input', function() {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            const query = this.value.trim();
            if (query.length > 2) {
                trackSearch(query);
            }
        }, 500);
    });
}

// Track search queries (for analytics)
function trackSearch(query) {
    // Store search queries in localStorage for analytics
    const searches = JSON.parse(localStorage.getItem('search_queries') || '[]');
    const timestamp = new Date().toISOString();

    searches.push({
        query: query,
        timestamp: timestamp,
        page: window.location.pathname
    });

    // Keep only last 100 searches
    if (searches.length > 100) {
        searches.splice(0, searches.length - 100);
    }

    localStorage.setItem('search_queries', JSON.stringify(searches));
}

// Quick search suggestions based on common queries
function setupSearchSuggestions() {
    const searchInput = document.querySelector('[data-md-component="search-query"]');
    if (!searchInput) return;

    const suggestions = [
        'installation',
        'configuration',
        'API endpoints',
        'authentication',
        'troubleshooting',
        'deployment',
        'testing',
        'docker setup',
        'environment variables',
        'error codes',
        'performance tuning',
        'security',
        'migration guide',
        'development setup'
    ];

    // Create suggestion dropdown
    const suggestionDropdown = document.createElement('div');
    suggestionDropdown.className = 'search-suggestions';
    suggestionDropdown.style.display = 'none';

    searchInput.parentNode.appendChild(suggestionDropdown);

    // Show suggestions on focus
    searchInput.addEventListener('focus', function() {
        if (this.value.length === 0) {
            showSuggestions(suggestions, suggestionDropdown, this);
        }
    });

    // Hide suggestions on blur
    searchInput.addEventListener('blur', function() {
        setTimeout(() => {
            suggestionDropdown.style.display = 'none';
        }, 200);
    });

    // Filter suggestions as user types
    searchInput.addEventListener('input', function() {
        const query = this.value.toLowerCase();
        if (query.length === 0) {
            showSuggestions(suggestions, suggestionDropdown, this);
        } else if (query.length > 1) {
            const filtered = suggestions.filter(s =>
                s.toLowerCase().includes(query)
            );
            showSuggestions(filtered, suggestionDropdown, this);
        } else {
            suggestionDropdown.style.display = 'none';
        }
    });
}

// Display search suggestions
function showSuggestions(suggestions, dropdown, input) {
    if (suggestions.length === 0) {
        dropdown.style.display = 'none';
        return;
    }

    dropdown.innerHTML = suggestions.slice(0, 8).map(suggestion =>
        `<div class="suggestion-item" data-suggestion="${suggestion}">${suggestion}</div>`
    ).join('');

    dropdown.style.display = 'block';

    // Add click handlers
    dropdown.querySelectorAll('.suggestion-item').forEach(item => {
        item.addEventListener('click', function() {
            input.value = this.dataset.suggestion;
            input.focus();
            dropdown.style.display = 'none';

            // Trigger search
            const event = new Event('input', { bubbles: true });
            input.dispatchEvent(event);
        });
    });
}

// Add search result highlighting
function highlightSearchResults() {
    const urlParams = new URLSearchParams(window.location.search);
    const searchQuery = urlParams.get('q');

    if (searchQuery) {
        // Highlight search terms in page content
        const content = document.querySelector('[data-md-component="main"]');
        if (content) {
            highlightText(content, searchQuery);
        }
    }
}

// Highlight text utility
function highlightText(element, searchText) {
    const regex = new RegExp(`(${escapeRegex(searchText)})`, 'gi');
    const walker = document.createTreeWalker(
        element,
        NodeFilter.SHOW_TEXT,
        null,
        false
    );

    const textNodes = [];
    let node;

    while (node = walker.nextNode()) {
        textNodes.push(node);
    }

    textNodes.forEach(textNode => {
        if (regex.test(textNode.textContent)) {
            const highlighted = textNode.textContent.replace(regex, '<mark>$1</mark>');
            const span = document.createElement('span');
            span.innerHTML = highlighted;
            textNode.parentNode.replaceChild(span, textNode);
        }
    });
}

// Escape regex special characters
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Initialize search result highlighting
highlightSearchResults();
