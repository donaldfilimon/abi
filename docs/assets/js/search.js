// Advanced search functionality for GitHub Pages
(function() {
  'use strict';

  let searchIndex = [];
  let searchWorker;

  // Initialize search with web worker for better performance
  function initializeAdvancedSearch() {
    // Load search index
    fetch('/generated/search_index.json')
      .then(response => response.json())
      .then(data => {
        searchIndex = data;
        setupSearchInterface();
      })
      .catch(error => {
        console.warn('Search functionality unavailable:', error);
      });
  }

  function setupSearchInterface() {
    const searchInput = document.getElementById('search-input');
    if (!searchInput) return;

    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
      // Ctrl/Cmd + K to focus search
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
      }
      
      // Escape to clear search
      if (e.key === 'Escape' && document.activeElement === searchInput) {
        searchInput.value = '';
        hideSearchResults();
      }
    });

    // Add search suggestions
    searchInput.addEventListener('focus', function() {
      if (this.value.trim() === '') {
        showSearchSuggestions();
      }
    });
  }

  function showSearchSuggestions() {
    const suggestions = [
      'database API',
      'neural networks',
      'SIMD operations',
      'performance guide',
      'plugin system',
      'vector search',
      'machine learning'
    ];

    const searchResults = document.getElementById('search-results');
    if (!searchResults) return;

    searchResults.innerHTML = suggestions.map(suggestion =>
      `<div class="search-result-item suggestion" onclick="searchFor('${suggestion}')">
        <div class="search-result-title">💡 ${suggestion}</div>
        <div class="search-result-excerpt">Search suggestion</div>
      </div>`
    ).join('');
    
    searchResults.classList.remove('hidden');
  }

  function searchFor(query) {
    const searchInput = document.getElementById('search-input');
    if (searchInput) {
      searchInput.value = query;
      searchInput.dispatchEvent(new Event('input'));
    }
  }

  function hideSearchResults() {
    const searchResults = document.getElementById('search-results');
    if (searchResults) {
      searchResults.classList.add('hidden');
    }
  }

  // Fuzzy search implementation
  function fuzzySearch(query, items) {
    const fuse = new Fuse(items, {
      keys: ['title', 'excerpt'],
      threshold: 0.4,
      distance: 100,
      includeScore: true
    });
    
    return fuse.search(query).map(result => result.item);
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAdvancedSearch);
  } else {
    initializeAdvancedSearch();
  }

})();
