// Advanced search functionality for GitHub Pages
(function() {
  'use strict';

  const baseUrl = (document.body && document.body.dataset.baseurl) || '';

  function withBase(path) {
    if (!path) return baseUrl || '';
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;
    if (!baseUrl || baseUrl === '/') {
      return normalizedPath;
    }

    return `${baseUrl.replace(/\/$/, '')}${normalizedPath}`;
  }

  let searchIndex = [];

  // Initialize search with web worker for better performance
  function initializeAdvancedSearch() {
    // Load search index
    fetch(withBase('generated/search_index.json'))
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

    searchResults.innerHTML = '';

    suggestions.forEach(suggestion => {
      const item = document.createElement('div');
      item.className = 'search-result-item suggestion';

      const title = document.createElement('div');
      title.className = 'search-result-title';
      title.textContent = `💡 ${suggestion}`;

      const excerpt = document.createElement('div');
      excerpt.className = 'search-result-excerpt';
      excerpt.textContent = 'Search suggestion';

      item.appendChild(title);
      item.appendChild(excerpt);

      item.addEventListener('click', function() {
        searchFor(suggestion);
      });

      searchResults.appendChild(item);
    });

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
      searchResults.innerHTML = '';
    }
  }

  // Fuzzy search implementation
  function fuzzySearch(query, items) {
    const normalizedQuery = query.trim().toLowerCase();
    if (normalizedQuery.length === 0) return [];

    return items
      .map(item => {
        const haystack = `${item.title} ${item.excerpt}`.toLowerCase();
        const index = haystack.indexOf(normalizedQuery);
        return index === -1 ? null : { item, score: index };
      })
      .filter(Boolean)
      .sort((a, b) => a.score - b.score)
      .map(result => result.item);
  }

  window.searchFor = searchFor;

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAdvancedSearch);
  } else {
    initializeAdvancedSearch();
  }

})();
