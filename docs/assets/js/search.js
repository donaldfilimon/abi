// Advanced search functionality for GitHub Pages
(function() {
  'use strict';

  const baseUrl = resolveBaseUrl();
  let searchIndex = [];

  function resolveBaseUrl() {
    const fromWindow = typeof window !== 'undefined' && typeof window.__DOCS_BASEURL === 'string'
      ? window.__DOCS_BASEURL
      : '';
    const fromBody = document.body ? (document.body.getAttribute('data-baseurl') || '') : '';
    const raw = fromBody || fromWindow || '';
    if (!raw || raw === '/') return '';
    return raw.endsWith('/') ? raw.slice(0, -1) : raw;
  }

  function buildUrl(path) {
    if (!path) return baseUrl || '';
    const normalized = path.startsWith('/') ? path : `/${path}`;
    return `${baseUrl}${normalized}`;
  }

  function initializeAdvancedSearch() {
    const existingData = Array.isArray(window.__ABI_SEARCH_DATA) ? window.__ABI_SEARCH_DATA : null;
    if (existingData) {
      searchIndex = existingData;
      setupSearchInterface();
      return;
    }

    fetch(buildUrl('/generated/search_index.json'))
      .then(response => response.json())
      .then(data => {
        searchIndex = Array.isArray(data) ? data : [];
        if (searchIndex.length > 0) {
          window.__ABI_SEARCH_DATA = searchIndex;
        }
        setupSearchInterface();
      })
      .catch(error => {
        console.warn('Search functionality unavailable:', error);
        setupSearchInterface();
      });
  }

  function setupSearchInterface() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');
    if (!searchInput || !searchResults) return;

    document.addEventListener('keydown', function(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
      }

      if (e.key === 'Escape' && document.activeElement === searchInput) {
        searchInput.value = '';
        hideSearchResults(searchResults);
      }
    });

    searchInput.addEventListener('focus', function() {
      if (this.value.trim() === '') {
        showSearchSuggestions(searchResults);
      }
    });

    searchInput.addEventListener('input', function() {
      if (this.value.trim() === '') {
        showSearchSuggestions(searchResults);
      }
    });

    searchResults.addEventListener('mousedown', function(event) {
      if (event.target.closest('.search-result-item')) {
        event.preventDefault();
      }
    });
  }

  function hideSearchResults(container) {
    if (container) {
      container.classList.add('hidden');
    }
  }

  function showSearchSuggestions(container) {
    if (!container) return;

    const suggestions = buildSuggestionList();
    container.innerHTML = suggestions.map(suggestion => `
      <div class="search-result-item suggestion" data-suggestion="${escapeHtml(suggestion)}">
        <div class="search-result-title">💡 ${escapeHtml(suggestion)}</div>
        <div class="search-result-excerpt">Press Enter to search</div>
      </div>
    `).join('');

    container.classList.remove('hidden');
  }

  function buildSuggestionList() {
    if (!Array.isArray(searchIndex) || searchIndex.length === 0) {
      return [
        'database API',
        'neural networks',
        'SIMD operations',
        'performance guide',
        'plugin system',
        'vector search',
        'machine learning'
      ];
    }

    const titles = [];
    for (const item of searchIndex) {
      if (item && item.title && !titles.includes(item.title)) {
        titles.push(item.title);
      }
      if (titles.length >= 7) break;
    }
    return titles;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAdvancedSearch);
  } else {
    initializeAdvancedSearch();
  }

})();
n