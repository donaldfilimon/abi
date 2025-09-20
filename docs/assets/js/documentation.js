// Enhanced GitHub Pages Documentation JavaScript
(function() {
  'use strict';

  function buildUrl(path) {
    const baseUrl = document.body ? (document.body.dataset.baseurl || '') : '';

    if (!path) {
      return baseUrl || '/';
    }

    if (/^https?:\/\//i.test(path)) {
      return path;
    }

    const normalizedBase = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
    const normalizedPath = path.startsWith('/') ? path : `/${path}`;

    return `${normalizedBase}${normalizedPath}` || normalizedPath;
  }

  function normalizeDocPath(file) {
    if (!file) {
      return buildUrl('/');
    }

    let normalized = file.trim();

    if (normalized.endsWith('.md')) {
      normalized = normalized.slice(0, -3);
    }

    if (!normalized.startsWith('/')) {
      normalized = `/${normalized}`;
    }

    if (!normalized.endsWith('/') && !normalized.endsWith('.html')) {
      normalized = `${normalized}/`;
    }

    return buildUrl(normalized);
  }

  // Generate table of contents
  function generateTOC() {
    const content = document.querySelector('.documentation-content .content');
    const tocList = document.getElementById('toc-list');

    if (!content || !tocList) return;

    tocList.innerHTML = '';

    const headings = content.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) {
      const toc = document.getElementById('toc');
      if (toc) toc.style.display = 'none';
      return;
    }

    headings.forEach((heading, index) => {
      const id = heading.id || `heading-${index}`;
      heading.id = id;

      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = `#${id}`;
      a.textContent = heading.textContent;
      a.className = `toc-${heading.tagName.toLowerCase()}`;

      li.appendChild(a);
      tocList.appendChild(li);
    });
  }

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

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function escapeAttribute(text) {
    return escapeHtml(text).replace(/`/g, '&#96;');
  }

  function escapeRegExp(text) {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }

  // Search functionality
  function initializeSearch() {
    const searchInput = document.getElementById('search-input');
    const searchResults = document.getElementById('search-results');

    if (!searchInput || !searchResults) return;

    let searchData = [];
    
    // Load search index
    fetch(buildUrl('/generated/search_index.json'))
      .then(response => response.json())
      .then(data => {
        searchData = data;
      })
      .catch(error => {
        console.warn('Search index not available:', error);
      });

    let searchTimeout;
    searchInput.addEventListener('input', function() {
      clearTimeout(searchTimeout);
      const query = this.value.trim().toLowerCase();

      if (query.length < 2) {
        searchResults.classList.add('hidden');
        searchResults.innerHTML = '';
        return;
      }

      searchTimeout = setTimeout(() => {
        const results = searchData.filter(item =>
          item.title.toLowerCase().includes(query) ||
          item.excerpt.toLowerCase().includes(query)
        ).slice(0, 10);

        displaySearchResults(results, query);
      }, 200);
    });

    searchResults.addEventListener('click', function(event) {
      const target = event.target.closest('.search-result-item');
      if (!target) return;

      if (target.dataset.file) {
        navigateToPage(target.dataset.file);
        searchResults.classList.add('hidden');
        return;
      }

      if (target.dataset.suggestion) {
        applySuggestion(target.dataset.suggestion, searchInput);
        searchResults.classList.add('hidden');
      }
    });

    // Hide search results when clicking outside
    document.addEventListener('click', function(e) {
      if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
        searchResults.classList.add('hidden');
      }
    });
  }

  function displaySearchResults(results, query) {
    const searchResults = document.getElementById('search-results');
    if (!searchResults) return;

    searchResults.innerHTML = '';

    if (results.length === 0) {
      searchResults.innerHTML = '<div class="search-result-item" data-empty="true">No results found</div>';
    } else {
      const safeQuery = query ? escapeRegExp(query) : '';
      searchResults.innerHTML = results.map(result => {
        const file = escapeAttribute(result.file);
        const title = highlightText(result.title, safeQuery);
        const excerpt = highlightText(result.excerpt, safeQuery);
        return `
          <div class="search-result-item" data-file="${file}">
            <div class="search-result-title">${title}</div>
            <div class="search-result-excerpt">${excerpt}</div>
          </div>
        `;
      }).join('');
    }

      const emptyState = document.createElement('div');
      emptyState.className = 'search-result-item';
      emptyState.textContent = 'No results found';
      searchResults.appendChild(emptyState);
      searchResults.classList.remove('hidden');
      return;
    }

    results.forEach(result => {
      const item = document.createElement('div');
      item.className = 'search-result-item';

      const title = document.createElement('div');
      title.className = 'search-result-title';
      title.innerHTML = highlightText(result.title, query);

      const excerpt = document.createElement('div');
      excerpt.className = 'search-result-excerpt';
      excerpt.innerHTML = highlightText(result.excerpt, query);

      item.appendChild(title);
      item.appendChild(excerpt);

      item.addEventListener('click', () => navigateToPage(result.file));

      searchResults.appendChild(item);
    });

    searchResults.classList.remove('hidden');
  }

  function highlightText(text, escapedQuery) {
    const safeText = escapeHtml(text);
    if (!escapedQuery) return safeText;
    const regex = new RegExp(`(${escapedQuery})`, 'gi');
    return safeText.replace(regex, '<strong>$1</strong>');
  }

  function navigateToPage(file) {
    window.location.href = normalizeDocPath(file);
  }

  // Smooth scrolling for anchor links
  function initializeSmoothScrolling() {
    document.addEventListener('click', function(e) {
      if (e.target.tagName === 'A' && e.target.getAttribute('href').startsWith('#')) {
        e.preventDefault();
        const targetId = e.target.getAttribute('href').substring(1);
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
          targetElement.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });

          // Update URL without triggering navigation
          history.pushState(null, null, `#${targetId}`);
        }
      }
    });
  }

  // Copy code functionality
  function addCopyButtons() {
    const codeBlocks = document.querySelectorAll('pre code');

    codeBlocks.forEach(function(codeBlock) {
      const pre = codeBlock.parentElement;
      const button = document.createElement('button');
      button.textContent = 'Copy';
      button.className = 'copy-button';
      button.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background: var(--color-canvas-subtle);
        border: 1px solid var(--color-border-default);
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        cursor: pointer;
        color: var(--color-fg-default);
      `;

      pre.style.position = 'relative';
      pre.appendChild(button);

      button.addEventListener('click', function() {
        if (!navigator.clipboard || !navigator.clipboard.writeText) {
          return;
        }

        navigator.clipboard.writeText(codeBlock.textContent).then(function() {
          button.textContent = 'Copied!';
          setTimeout(function() {
            button.textContent = 'Copy';
          }, 2000);
        });
      });
    });
  }

  // Performance monitoring
  function trackPerformance() {
    if ('performance' in window) {
      window.addEventListener('load', function() {
        setTimeout(function() {
          const entries = performance.getEntriesByType('navigation');
          if (!entries || entries.length === 0) return;
          const perfData = entries[0];
          const loadTime = perfData.loadEventEnd - perfData.loadEventStart;

          if (loadTime > 0) {
            console.log(`Page load time: ${loadTime}ms`);
          }
        }, 0);
      });
    }

    window.addEventListener('load', function() {
      setTimeout(function() {
        const navigationEntries = performance.getEntriesByType
          ? performance.getEntriesByType('navigation')
          : [];

        const perfData = navigationEntries && navigationEntries.length > 0
          ? navigationEntries[0]
          : performance.timing;

        if (!perfData) {
          return;
        }

        const start = perfData.loadEventStart || perfData.domComplete || 0;
        const end = perfData.loadEventEnd || perfData.domComplete || 0;
        const loadTime = end - start;

        if (loadTime > 0) {
          console.log(`Page load time: ${loadTime}ms`);
        }
      }, 0);
    });
  }

  // Initialize all functionality when DOM is ready
  function initialize() {
    generateTOC();
    initializeSearch();
    initializeSmoothScrolling();
    addCopyButtons();
    trackPerformance();

    // Add performance badges to relevant sections
    const performanceMarkers = Array.from(document.querySelectorAll('code')).filter(code => {
      const text = code.textContent || '';
      return text.includes('~') || text.includes('ms') || text.includes('μs');
    });
    performanceMarkers.forEach(function(marker) {
      if (!marker || !marker.textContent) {
        return;
      }

      if (!/[~≈]|\bms\b|µs|μs/.test(marker.textContent)) {
        return;
      }

      if (!marker.parentElement || marker.parentElement.querySelector('.performance-badge')) {
        return;
      }

      const badge = document.createElement('span');
      badge.className = 'performance-badge';
      badge.textContent = 'PERF';
      marker.parentElement.insertBefore(badge, marker.nextSibling);
    });

  }

  // DOM ready check
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }

  window.navigateToPage = navigateToPage;

})();
