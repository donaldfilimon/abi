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

    const headings = content.querySelectorAll('h2, h3, h4');
    if (headings.length === 0) {
      document.getElementById('toc').style.display = 'none';
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

    if (results.length === 0) {
      searchResults.innerHTML = '<div class="search-result-item">No results found</div>';
    } else {
      searchResults.innerHTML = results.map(result => 
        `<div class="search-result-item" onclick="navigateToPage('${result.file}')">
          <div class="search-result-title">${highlightText(result.title, query)}</div>
          <div class="search-result-excerpt">${highlightText(result.excerpt, query)}</div>
        </div>`
      ).join('');
    }
    
    searchResults.classList.remove('hidden');
  }

  function highlightText(text, query) {
    if (!query) return text;
    const regex = new RegExp(`(${query})`, 'gi');
    return text.replace(regex, '<strong>$1</strong>');
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
          const perfData = performance.getEntriesByType('navigation')[0];
          const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
          
          if (loadTime > 0) {
            console.log(`Page load time: ${loadTime}ms`);
          }
        }, 0);
      });
    }
  }

  // Initialize all functionality when DOM is ready
  function initialize() {
    generateTOC();
    initializeSearch();
    initializeSmoothScrolling();
    addCopyButtons();
    trackPerformance();
    
    // Add performance badges to relevant sections
    const performanceMarkers = document.querySelectorAll('code:contains("~"), code:contains("ms"), code:contains("μs")');
    performanceMarkers.forEach(function(marker) {
      if (marker.textContent.includes('~')) {
        const badge = document.createElement('span');
        badge.className = 'performance-badge';
        badge.textContent = 'PERF';
        marker.parentElement.insertBefore(badge, marker.nextSibling);
      }
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
